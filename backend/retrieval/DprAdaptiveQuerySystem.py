import os
import pickle
import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class QuestionClassifier:    
    def __init__(self, model_path):
        print(f"   - Đang tải mô hình phân loại từ: {model_path}")
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
    def __init__(self, config: Dict[str, Any]):
        self.config = config                        
        self.question_encoder = SentenceTransformer(config['question_encoder_path'])            
        self.reranker = CrossEncoder(config['reranker_model'])            
        self.index = faiss.read_index(config['index_path'])                        
        with open(config['docs_path'], "rb") as f:
            self.docs = pickle.load(f)                
        self.classifier = QuestionClassifier(config['question_classifier_path'])          

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
        question_type = pre_classified_intent if pre_classified_intent else self.classifier.classify(question)
        print(f"   - Ý định được xác định: '{question_type}'")                
        strategy = self.config['strategy_config'].get(question_type, self.config['strategy_config']['Inference'])
        retriever_k = strategy['k']
        reranker_top_n = strategy['top_n']        
        query_vector = self.question_encoder.encode(question)
        query_vector_2d = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_vector_2d)
        scores, indices = self.index.search(query_vector_2d, k=retriever_k)
                
        best_score = scores[0][0]
        threshold = self.config.get('relevance_similarity_threshold', 0.5) # Dùng .get để an toàn
        print(f"   - Độ tương đồng cao nhất: {best_score:.4f} (Ngưỡng: > {threshold})")
        
        if best_score < threshold:
            print("KẾT QUẢ KHÔNG ĐỦ TIN CẬY => Từ chối trả lời.")
            return []

        #reranking
        candidate_docs_data = [self.docs[i] for i in indices[0]]
        candidate_docs_lc = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in candidate_docs_data]
        final_docs = self._rerank(question, candidate_docs_lc, top_n=reranker_top_n)
        
        return final_docs