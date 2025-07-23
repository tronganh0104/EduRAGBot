import pickle
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from retrieval.config import CONFIG

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    from langchain_cohere import CohereRerank
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

class QuerySystem:
    def __init__(self, config):
        self.config = config
        print("Đang khởi tạo")
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        self.reranker = self._load_reranker()
        self.index = faiss.read_index(str(self.config['index_path']))
        with open(self.config['docs_path'], "rb") as f:
            self.docs = pickle.load(f)

    def _load_reranker(self):
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if COHERE_AVAILABLE and cohere_api_key:
            print("Có thể áp dụng Cohere Reranker.")
            return CohereRerank(
                cohere_api_key=cohere_api_key,
                model="rerank-multilingual-v3.0",
                top_n=self.config["reranker_top_n"]
            )
        else:
            print(f"Không tìm thấy COHERE_API_KEY. Sử dụng CrossEncoder: {self.config['reranker_model']}")
            return CrossEncoder(self.config['reranker_model'])
        
    def query(self, text: str) -> list[Document]:
        query_vector = self.embedding_model.encode(text)
        query_vector_2d = np.array([query_vector], dtype='float32')
        
        distances, indices = self.index.search(query_vector_2d, k=self.config["retriever_k"])
        retrieved_docs_data = [self.docs[i] for i in indices[0]]
        
        retrieved_docs_lc = [
            Document(page_content=doc['content'], metadata=doc['metadata'])
            for doc in retrieved_docs_data
        ]
        
        print(f"\nCÂU HỎI: '{text}'")
        print(f"Đã truy vấn được {len(retrieved_docs_lc)} tài liệu ban đầu.")

        # ---RERANKING---
        if not self.reranker:
            print("Không có reranker, trả về kết quả truy vấn cơ bản.")
            return retrieved_docs_lc

        print("Đang thực hiện Reranking...")
        if COHERE_AVAILABLE and isinstance(self.reranker, type(CohereRerank)):
            reranked_docs = self.reranker.compress_documents(
                documents=retrieved_docs_lc,
                query=text
            )
        else:
            reranker_input = [[text, doc.page_content] for doc in retrieved_docs_lc]
            scores = self.reranker.predict(reranker_input)
            
            doc_score_pairs = list(zip(retrieved_docs_lc, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = [doc for doc, score in doc_score_pairs[:self.config["reranker_top_n"]]]

        return reranked_docs

if __name__ == "__main__":
    if not (CONFIG['index_path'].exists() and CONFIG['docs_path'].exists()):
        print(f"LỖI: Không tìm thấy file tại '{CONFIG['index_path']}' hoặc '{CONFIG['docs_path']}'.")
    else:
        query_system = QuerySystem(CONFIG)
        user_query = "Có các hình thức dạy học nào"
        final_results = query_system.query(user_query)
        
        print(f"\nKết quả cuối cùng sau khi Reranking (top {len(final_results)}):")
        for i, doc in enumerate(final_results):
            print(f"  ------------ Document {i+1} -------------")
            print(f"  Nội dung: {doc.page_content}")