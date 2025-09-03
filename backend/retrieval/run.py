import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
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

if __name__ == '__main__':    
    rag_system = RAGSystem(MAIN_CONFIG)    
    while True:        
        user_question = input("\nVui lòng nhập câu hỏi của bạn (hoặc gõ 'q' để thoát): ")
        
        if user_question.lower() == 'q':
            print("Tạm biệt!")
            break                    
        final_context = rag_system.ask(user_question)
                
        print("\n" + "*"*80)
        print("Dưới đây là các thông tin liên quan nhất mà hệ thống tìm thấy:")
        print("*"*80)
        
        if not final_context:
            print("   -> Không tìm thấy thông tin phù hợp trong kho tri thức.")
        else:
            for i, doc_data in enumerate(final_context):
                print(f"\n---------- NGUỒN THÔNG TIN #{i+1} ----------")
                print(f"Metadata: {doc_data['metadata']}")
                print(f"\nNội dung:\n{doc_data['content']}")
                print("-"*(40 + len(str(i+1))))