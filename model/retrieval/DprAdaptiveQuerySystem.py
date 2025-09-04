import os
import pickle
import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from typing import List, Dict, Any

# class QuestionClassifier:    
#     def __init__(self, model_path):        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()
#         self.id2label = self.model.config.id2label

#     def classify(self, question: str) -> str:
#         inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         prediction = torch.argmax(outputs.logits, dim=1).item()
#         return self.id2label[prediction]

class DprAdaptiveQuerySystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config                
        self.question_encoder = SentenceTransformer(config['question_encoder_path'])
        self.reranker = CrossEncoder(config['reranker_model'])
        self.index = faiss.read_index(config['index_path'])
        
        # Tải dữ liệu chunk "con" (để truy xuất)
        with open(config['docs_path'], "rb") as f:
            self.child_docs = pickle.load(f)
            
        # Tải dữ liệu văn bản "cha" (để trả về ngữ cảnh)
        parent_docs_path = config.get('parent_docs_path')
        if parent_docs_path and os.path.exists(parent_docs_path):
            with open(parent_docs_path, "rb") as f:
                self.parent_docs = pickle.load(f)            
        else:
            self.parent_docs = None            

    def _rerank(self, question: str, candidate_docs: List[Document], top_n: int) -> List[Document]:        
        if not candidate_docs:
            return []
        print(f"   - Đang Reranking trên {len(candidate_docs)} ứng viên để lấy top {top_n}...")
        reranker_input = [[question, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(reranker_input)
        
        doc_score_pairs = list(zip(candidate_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_score_pairs[:top_n]]

    def query(self, question: str, pre_classified_intent: str = None) -> List[Document]:        
        print(f"\nBắt đầu xử lý câu hỏi: '{question}'")                
        question_type = pre_classified_intent if pre_classified_intent else "Inference"
        print(f"Thuộc loại '{question_type}'")
        
        strategy = self.config['strategy_config'].get(question_type, self.config['strategy_config']['Inference'])
        retriever_k = strategy['k']
        reranker_top_n = strategy['top_n']        
        
        query_vector = self.question_encoder.encode(question)
        query_vector_2d = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_vector_2d)
        scores, indices = self.index.search(query_vector_2d, k=retriever_k)
                
        best_score = scores[0][0]
        threshold = self.config['relevance_similarity_threshold']
        print(f"   - Độ tương đồng cao nhất: {best_score:.4f} (Ngưỡng: > {threshold})")
        
        if best_score < threshold:
            print("   -KẾT QUẢ KHÔNG ĐỦ TIN CẬY => Từ chối trả lời.")
            return []
        
        candidate_child_docs_data = [self.child_docs[i] for i in indices[0]]
        candidate_child_docs_lc = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in candidate_child_docs_data]
        reranked_child_docs = self._rerank(question, candidate_child_docs_lc, top_n=reranker_top_n)
                        
        if question_type in ["Definition", "Factoid"]:            
            return reranked_child_docs
        elif self.parent_docs is not None:            
            final_parent_docs_dict = {}
            for child_doc in reranked_child_docs:
                parent_id = child_doc.metadata.get('parent_article_number')
                if parent_id is not None and parent_id not in final_parent_docs_dict:
                    parent_content = self.parent_docs.get(parent_id)
                    if parent_content:
                        parent_metadata = {
                            "source_document": child_doc.metadata.get('source_document'),
                            "article_number": parent_id
                        }
                        final_parent_docs_dict[parent_id] = Document(page_content=parent_content, metadata=parent_metadata)
            return list(final_parent_docs_dict.values())
        else:
            # Trường hợp dự phòng nếu không có chunk "cha"
            return reranked_child_docs

if __name__ == '__main__':    
    try:
        with open("retrieval/configs/qcdt_config.json", 'r', encoding='utf-8') as f:
            qcdt_config = json.load(f)
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file config. Vui lòng tạo file configs/qcdt_config.json")
        qcdt_config = None

    if qcdt_config:        
        chuyen_gia_dao_tao = DprAdaptiveQuerySystem(qcdt_config)
        
        # Giả lập câu hỏi
        cau_hoi = "Liệt kê các hạng tốt nghiệp của sinh viên?"
        y_dinh_da_phan_loai = "List"
                
        ket_qua = chuyen_gia_dao_tao.query(cau_hoi, pre_classified_intent=y_dinh_da_phan_loai)                
        print(f"\nKết quả cuối cùng trả về:")
        if not ket_qua:
            print("--> Không tìm thấy thông tin liên quan.")
        else:
            for i, doc in enumerate(ket_qua):
                print(f"  --- Document {i+1} ---")            
                metadata = doc.metadata
                source = metadata.get('source_document', 'Không rõ nguồn')
                article_title = metadata.get('article_title', 'Không rõ tiêu đề')
                print(f"  Nguồn: {source}")
                print(f"  Tiêu đề Điều: {article_title}")
                print(f"  Nội dung: {doc.page_content}")