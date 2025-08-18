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
    "index_path": "C:/Users/Admin/Downloads/model/vector_store_data_v4/index.faiss",
    "docs_path": "C:/Users/Admin/Downloads/model/vector_store_data_v4/docs.pkl",
    "question_encoder_path": "C:/Users/Admin/Downloads/model/models/dpr-phobert-augmented",
    "question_classifier_path": "C:/Users/Admin/Downloads/model/models/question_classifier",
    "reranker_model": "BAAI/bge-reranker-base",
    
    "strategy_config": {
        "Definition": {"k": 5, "top_n": 1},
        "Factoid":    {"k": 5, "top_n": 3},
        "Yes/No":     {"k": 7, "top_n": 4},
        "List":       {"k": 8, "top_n": 4},
        "Inference":  {"k": 12, "top_n": 5}
    },
    "relevance_similarity_threshold": 0.3726
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
        print(f"\n====================\nBắt đầu xử lý câu hỏi: '{question}'")                
        question_type = self.classifier.classify(question)
        print(f"Loại câu hỏi được xác định: {question_type}")
        
        strategy = self.config['strategy_config'].get(question_type, self.config['strategy_config']['Inference'])
        retriever_k = strategy['k']
        reranker_top_n = strategy['top_n']

        print(f"Chiến lược được chọn: retriever_k={retriever_k}, reranker_top_n={reranker_top_n}")
        
        query_vector = self.question_encoder.encode(question)
        query_vector_2d  = np.array([query_vector], dtype='float32')

        faiss.normalize_L2(query_vector_2d)
        scores, indices = self.index.search(query_vector_2d, k=retriever_k)        

        best_score = scores[0][0]
        threshold = self.config['relevance_similarity_threshold']

        print(f"Độ tương đồng của kết quả cao nhất: {best_score:.4f} (Ngưỡng yêu cầu: > {threshold})")

        if best_score < threshold: 
            print("CẢNH BÁO: Kết quả tốt nhất không đủ độ tương đồng.")
            print("=> KẾT LUẬN: Không tìm thấy tài liệu liên quan trong kho tri thức.")
            return []

        candidate_docs_data = [self.docs[i] for i in indices[0]]
        candidate_docs_lc = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in candidate_docs_data]        

        # RERANKING
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
            # --- Các câu hỏi thông tin cơ bản ---
            "Definition (Rất dễ)": "Học phần điều kiện là gì?",
            "Factoid (Dễ)": "Điểm chữ F tương ứng với thang điểm số mấy?",
            
            # --- Các câu hỏi yêu cầu tìm kiếm chính xác ---
            "Yes/No (Trung bình)": "Sinh viên có được phép học cùng lúc hai chương trình không?",
            "List (Trung bình)": "Liệt kê các trường hợp sinh viên bị buộc thôi học?",
            
            # --- Các câu hỏi suy luận và phức tạp ---
            "Inference (Khó)": "Phân biệt giữa việc chuyển ngành và chuyển trường?",
            "Inference (Rất khó)": "Một sinh viên năm thứ ba chương trình chất lượng cao bị điểm F một học phần nâng cao, cần làm gì?",
            
            # --- Các câu hỏi kiểm tra giới hạn và lớp kiểm duyệt ---
            "Borderline (Kiểm tra ngưỡng)": "Quy định về việc miễn giảm học phí cho sinh viên có hoàn cảnh khó khăn là gì?",
            "Out-of-Domain (Dễ từ chối)": "Công thức nấu món phở bò Hà Nội?",
            "Out-of-Domain (Gây nhiễu)": "Thủ tục đăng ký thi bằng lái xe máy cho sinh viên?",
        }

        for q_type, q_text in test_queries.items():
            final_results = query_system.query(q_text)
            
            print(f"\nKết quả cuối cùng cho câu hỏi '{q_text}' (Loại được test: {q_type}):")
            if not final_results:
                print("--> Hệ thống đã xác định câu hỏi nằm ngoài phạm vi và không trả về kết quả nào (ĐÚNG).")
            else:
                for i, doc in enumerate(final_results):
                    print(f"  --- Document {i+1} ---")
                    print(f"  Nội dung: {doc.page_content}")
                    print(f"  Metadata: {doc.metadata}")