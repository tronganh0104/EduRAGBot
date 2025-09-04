import os
import sys
import json
import google.generativeai as genai
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.DprAdaptiveQuerySystem import DprAdaptiveQuerySystem

ROUTER_CONFIG = {
    "qcdt": { "config_path": "../retrieval/configs/qcdt_config.json" },
    "qcctsv": { "config_path": "../retrieval/configs/qcctsv_config.json" },
    "qctdkt": { "config_path": "../retrieval/configs/qctdkt_config.json" },
    "tuyensinh": { "config_path": "../retrieval/configs/tuyensinh_config.json" }
}

class MainQueryRouterLLM:
    def __init__(self, config: Dict[str, Any]):
        self.config = config                
        self.query_systems: Dict[str, DprAdaptiveQuerySystem] = {}
        for topic_key, topic_config in self.config.items():
            try:
                with open(topic_config["config_path"], 'r', encoding='utf-8') as f:
                    specific_config = json.load(f)
                
                # Ánh xạ topic_key sang tên đầy đủ
                topic_name = self._get_topic_name(topic_key)                                
                self.query_systems[topic_name] = DprAdaptiveQuerySystem(specific_config)
            except Exception as e:
                print(f"LỖI khi khởi tạo chuyên gia cho '{topic_key}': {e}")
        
        # Khởi tạo LLM để phân loại
        try:
            api_key = "AIzaSyD5SBdzXkqz4BqwmcS-et6bY5d34ChIAAg"
            if not api_key:
                raise ValueError("Biến môi trường GOOGLE_API_KEY chưa được thiết lập.")
            genai.configure(api_key=api_key)
            self.router_llm = genai.GenerativeModel('gemini-1.5-flash')            
        except Exception as e:
            print(f"LỖI: Không thể khởi tạo mô hình Gemini. Lỗi: {e}")
            self.router_llm = None

    def _get_topic_name(self, topic_key: str) -> str:
        """Ánh xạ key ngắn sang tên đầy đủ."""
        mapping = {
            "qcdt": "Đào tạo",
            "qcctsv": "Công tác Sinh viên",
            "qctdkt": "Thi đua Khen thưởng",
            "tuyensinh": "Tuyển sinh"
        }
        return mapping.get(topic_key, topic_key)

    def _analyze_question(self, question: str) -> Dict[str, str]:
        """Sử dụng LLM để phân loại cả chủ đề và ý định."""
        if not self.router_llm:
            return {"topic": "Đào tạo", "intent": "Inference"} # Mặc định nếu LLM lỗi

        prompt = f"""Bạn là một hệ thống phân tích và định tuyến truy vấn thông minh cho chatbot của một trường đại học. Hãy phân tích câu hỏi của người dùng và trả về một đối tượng JSON.

Đối tượng JSON phải có 2 key:
1. "topic": Chủ đề của câu hỏi. Giá trị phải là MỘT trong các chuỗi sau: ["Đào tạo", "Công tác Sinh viên", "Thi đua Khen thưởng", "Tuyển sinh"].
2. "intent": Ý định của câu hỏi. Giá trị phải là MỘT trong các chuỗi sau: ["Definition", "List", "Yes/No", "Factoid", "Inference"].

Ví dụ:
Câu hỏi: "Điều kiện để được xét học bổng khuyến khích là gì?"
{{
  "topic": "Thi đua Khen thưởng",
  "intent": "Definition"
}}

Câu hỏi: "Sinh viên có được phép nghỉ học tạm thời không?"
{{
  "topic": "Đào tạo",
  "intent": "Yes/No"
}}

Hãy đảm bảo chỉ trả về đối tượng JSON hợp lệ và không có bất kỳ giải thích nào khác.

Câu hỏi: "{question}"
"""
        
        try:
            response = self.router_llm.generate_content(prompt)            
            json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            analysis_result = json.loads(json_text)
            
            if "topic" in analysis_result and "intent" in analysis_result:
                return analysis_result
        except (json.JSONDecodeError, Exception) as e:
            print(f"Lỗi khi phân tích câu hỏi bằng LLM: {e}")
        
        return {"topic": "Đào tạo", "intent": "Inference"}

    def query(self, question: str):
        # Phân tích câu hỏi bằng LLM
        analysis = self._analyze_question(question)
        route_topic = analysis.get("topic")
        intent = analysis.get("intent")
        
        print(f"\n==> LLM đã phân tích: Chủ đề='{route_topic}', Ý định='{intent}'")                
        selected_system = self.query_systems.get(route_topic)
        
        if not selected_system:
            print(f"Cảnh báo: Không tìm thấy chuyên gia cho chủ đề '{route_topic}'. Dùng chuyên gia 'Đào tạo' mặc định.")
            selected_system = self.query_systems["Đào tạo"]
                
        return selected_system.query(question, pre_classified_intent=intent)