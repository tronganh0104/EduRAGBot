import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.MainQueryRouter import MainQueryRouterLLM

MAIN_CONFIG = {
    "qcdt": {
        "config_path": "retrieval/configs/qcdt_config.json",
        "name": "Đào tạo"
    },
    "qcctsv": {
        "config_path": "retrieval/configs/qcctsv_config.json",
        "name": "Công tác Sinh viên"
    },
    "qctdkt": {
        "config_path": "retrieval/configs/qctdkt_config.json",
        "name": "Thi đua Khen thưởng"
    },
    "tuyensinh": {
        "config_path": "retrieval/configs/tuyensinh_config.json",
        "name": "Tuyển sinh"
    }
}

class RAGSystem:
    def __init__(self, config: Dict[str, Any]):                
        self.router = MainQueryRouterLLM(config)        

    def ask(self, question: str) -> List[Dict[str, Any]]:        
        if not question or not isinstance(question, str):
            print("LỖI: Câu hỏi không hợp lệ.")
            return []
        
        retrieved_docs = self.router.query(question)                
        if not retrieved_docs:
            return []
        
        formatted_results = []
        for doc in retrieved_docs:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return formatted_results
