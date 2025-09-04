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
                metadata = doc_data.get('metadata', {})
                content = doc_data.get('content', 'N/A')
                source = metadata.get('source_document', '')

                print(f"\n---------- NGUỒN THÔNG TIN #{i+1} ----------")                                
                if 'tuyển sinh' in source.lower():                    
                    print(metadata)

                else:                    
                    article_title = metadata.get('article_title', 'Không rõ')
                    article_number = metadata.get('parent_article_number') or metadata.get('article_number')
                    clause_number = metadata.get('clause_number')

                    location_str = f"Điều {article_number}"
                    if clause_number:
                        location_str += f", Khoản {clause_number}"
                    
                    print(f"Nguồn: {source}")
                    print(f"   - Trích từ: {location_str} - {article_title}")
                
                print(f"\nNội dung:\n{content}")
                print("-"*(40 + len(str(i+1))))