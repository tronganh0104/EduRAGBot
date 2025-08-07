import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

CONFIG = {
    "question_classifier_path": "C:/Users/Admin/Downloads/model/models/question_classifier",    
    "index_path": "C:/Users/Admin/Downloads/model/vector_store_data_v3/index.faiss",
    "docs_path": "C:/Users/Admin/Downloads/model/vector_store_data_v3/docs.pkl",
    "question_encoder_path": "C:/Users/Admin/Downloads/model/models/dpr-phobert-augmented",    
    "reranker_model": "BAAI/bge-reranker-base",    
    "default_retriever_k": 10,
    "default_reranker_top_n": 3
}

class QuestionClassifier:
    def __init__(self, model_path):        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def classify(self, question: str) -> str:
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return self.id2label[prediction]

class DprAdaptiveQuerySystem:
    def __init__(self, config):
        self.config = config        
        self.question_encoder = SentenceTransformer(config['question_encoder_path'])
        self.reranker = CrossEncoder(config['reranker_model'])
        self.index = faiss.read_index(config['index_path'])
        with open(config['docs_path'], "rb") as f:
            self.docs = pickle.load(f)
        self.classifier = QuestionClassifier(config['question_classifier_path'])        
        
    def _rerank(self, question: str, candidate_docs: list[Document], top_n: int) -> list[Document]:        
        reranker_input = [[question, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(reranker_input)
        doc_score_pairs = list(zip(candidate_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_score_pairs[:top_n]]

    def query(self, question: str) -> list[Document]:        
        question_type = self.classifier.classify(question)        
        retriever_k = self.config['default_retriever_k']
        reranker_top_n = self.config['default_reranker_top_n']

        if question_type in ["Definition", "Factoid"]:
            print("Chiến lược: Tập trung (Focused) - Tìm kiếm chính xác.")
            retriever_k = 5   
            reranker_top_n = 1 
        elif question_type == "Yes/No":
            print("Chiến lược: Cân bằng (Balanced) - Tìm quy định và ngữ cảnh.")
            retriever_k = 7
            reranker_top_n = 2
        elif question_type == "List":
            print("Chiến lược: Bao phủ (Comprehensive) - Tìm đủ các mục.")
            retriever_k = 10
            reranker_top_n = 4 
        elif question_type == "Inference":
            print("Chiến lược: Tổng hợp (Synthesizing) - Tối đa hóa ngữ cảnh liên quan.")
            retriever_k = 12  
            reranker_top_n = 4
                
        query_vector = self.question_encoder.encode(question)
        query_vector_2d = np.array([query_vector], dtype='float32')
        distances, indices = self.index.search(query_vector_2d, k=retriever_k)
        candidate_docs_data = [self.docs[i] for i in indices[0]]
        candidate_docs_lc = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in candidate_docs_data]
        print(f"DPR Retrieval tìm thấy {len(candidate_docs_lc)} ứng viên.")

        final_docs = self._rerank(question, candidate_docs_lc, top_n=reranker_top_n)
        
        return final_docs

if __name__ == '__main__':
    if not os.path.exists(CONFIG['question_classifier_path']):
        print(f"LỖI: Không tìm thấy thư mục model phân loại tại '{CONFIG['question_classifier_path']}'.")
    elif not (os.path.exists(CONFIG['index_path']) and os.path.exists(CONFIG['docs_path'])):
        print(f"LỖI: Không tìm thấy file Vector Store tại '{CONFIG['index_path']}' hoặc '{CONFIG['docs_path']}'.")
    else:    
        query_system = DprAdaptiveQuerySystem(CONFIG)
        
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
                print(f"  Nội dung: {doc.page_content}")
                print(f"  Metadata: {doc.metadata}")