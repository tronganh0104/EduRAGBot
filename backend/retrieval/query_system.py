import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import CONFIG

class QuestionClassifier:
    def __init__(self, model_path="retrieval/models/question_classifier"):
        print(f"Đang tải mô hình phân loại từ: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def classify(self, question: str) -> str:
        inputs = self.tokenizer(
            question, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return self.id2label[prediction]

class AdvancedQuerySystem:
    def __init__(self, config):
        self.config = config
        self.classifier = QuestionClassifier()
        print("Đang khởi tạo retrieval")
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        self.reranker = CrossEncoder(self.config['reranker_model'])
        self.index = faiss.read_index(self.config['index_path'])
        with open(self.config['docs_path'], "rb") as f:
            self.docs = pickle.load(f)
        self.corpus = [doc['content'] for doc in self.docs]
        tokenized_corpus = [word_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.content_to_doc_map = {doc['content']: doc for doc in self.docs}
        print("Ready")
        
    def _dense_retrieval(self, question: str, k: int) -> list:
        query_vector = self.embedding_model.encode(question)
        query_vector_2d = np.array([query_vector], dtype='float32')
        distances, indices = self.index.search(query_vector_2d, k=k)
        return [self.docs[i] for i in indices[0]]

    def _bm25_retrieval(self, question: str, k: int) -> list:
        tokenized_query = word_tokenize(question)
        top_k_contents = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return [self.content_to_doc_map[content] for content in top_k_contents]

    def _rerank(self, question: str, candidate_docs: list[Document], top_n: int) -> list[Document]:        
        reranker_input = [[question, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(reranker_input)
        doc_score_pairs = list(zip(candidate_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_n]]
        return reranked_docs

    def query(self, question: str) -> list[Document]:                
        question_type = self.classifier.classify(question)
        print(f"Loại câu hỏi được xác định: {question_type}")
        
        # Chiến lược truy vấn
        dense_k = self.config['default_dense_k']
        bm25_k = self.config['default_bm25_k']
        top_n = self.config['default_top_n']

        if question_type == "Definition" or question_type == "Factoid":            
            dense_k = 8
            bm25_k = 12  # Ưu tiên BM25
            top_n = 1    # Chỉ cần 1 chunk là đủ
        elif question_type == "List":            
            dense_k = 12
            bm25_k = 12
            top_n = 5    # Lấy nhiều chunk hơn chút vì cần list chunk ra                        
        elif question_type == "Inference":            
            dense_k = 15  # Tăng mạnh k để suy luận tốt hơn
            bm25_k = 15
            top_n = 5    
        # Loại Y/N thì để tham số mặc định

        # HYBRID SEARCH        
        dense_results = self._dense_retrieval(question, k=dense_k)
        bm25_results = self._bm25_retrieval(question, k=bm25_k)
        
        all_results_dict = {doc_data['content']: doc_data for doc_data in dense_results + bm25_results}
        candidate_docs_data = list(all_results_dict.values())
        
        candidate_docs_lc = [
            Document(page_content=doc['content'], metadata=doc['metadata']) 
            for doc in candidate_docs_data
        ]        

        # RERANKING
        final_docs = self._rerank(question, candidate_docs_lc, top_n)
        
        return final_docs

if __name__ == "__main__":
    if not (os.path.exists(CONFIG['index_path']) and os.path.exists(CONFIG['docs_path'])):
        print(f"LỖI: Không tìm thấy file tại '{CONFIG['index_path']}' hoặc '{CONFIG['docs_path']}'.")
    else:    
        query_system = AdvancedQuerySystem(CONFIG)
        
        test_queries = {
            "Definition": "Học phần điều kiện là gì ?",
            "Yes/No": "Sinh viên có được phép học cùng lúc hai chương trình không?",
            "List": "Liệt kê các hạng tốt nghiệp của sinh viên?",
            "Inference": "Làm thế nào để một sinh viên đang học chương trình chuẩn có thể chuyển sang học chương trình tài năng?",
            "Factoid": "Điểm chữ F tương ứng với thang điểm số mấy?"
        }
        for q_type, q_text in test_queries.items():
            final_results = query_system.query(q_text)                
            print(f"\nKết quả cuối cùng cho câu hỏi '{q_text}' (Loại được test: {q_type}):")
            for i, doc in enumerate(final_results):
                print(f"  --- Document {i+1} ---")
                print(f"  Nội dung: {doc.page_content[:350]}...")



