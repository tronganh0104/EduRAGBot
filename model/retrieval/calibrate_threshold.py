# file: retrieval/calibrate_threshold.py
import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.AdvancedQuerySystem import DprAdaptiveQuerySystem, CONFIG

# --- DANH SÁCH CÂU HỎI KIỂM THỬ ---
test_queries = [
    # === Loại 1: In-Domain ===
    {"id": "IN-1", "type": "In-Domain", "question": "Học phần điều kiện là gì?"},
    {"id": "IN-2", "type": "In-Domain", "question": "Sinh viên bị cảnh cáo học vụ trong trường hợp nào?"},
    {"id": "IN-3", "type": "In-Domain", "question": "Điểm trung bình chung tích lũy tối thiểu để tốt nghiệp là bao nhiêu?"},
    {"id": "IN-4", "type": "In-Domain", "question": "Có được phép đăng ký học lại để cải thiện điểm D không?"},
    {"id": "IN-5", "type": "In-Domain", "question": "Thời gian tối đa được phép học tại trường là mấy năm?"},
    {"id": "IN-6", "type": "In-Domain", "question": "Phân biệt giữa học phần bắt buộc và học phần tự chọn."},
    {"id": "IN-7", "type": "In-Domain", "question": "Liệt kê các hạng tốt nghiệp của sinh viên?"},
    {"id": "IN-8", "type": "In-Domain", "question": "Sinh viên được phép đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ chính?"},
    {"id": "IN-9", "type": "In-Domain", "question": "Tín chỉ là gì?"},
    {"id": "IN-10", "type": "In-Domain", "question": "Nếu bị điểm I ở một học phần, sinh viên cần làm gì ở học kỳ tiếp theo?"},

    # === Loại 2: Out-of-Domain ===
    {"id": "OUT-1", "type": "Out-of-Domain", "question": "Công thức nấu món phở bò Hà Nội?"},
    {"id": "OUT-2", "type": "Out-of-Domain", "question": "Vua Lý Thái Tổ dời đô về Thăng Long năm nào?"},
    {"id": "OUT-3", "type": "Out-of-Domain", "question": "Ai là tác giả của truyện Kiều?"},
    {"id": "OUT-4", "type": "Out-of-Domain", "question": "Làm thế nào để cài đặt hệ điều hành Windows 11?"},
    {"id": "OUT-5", "type": "Out-of-Domain", "question": "Giá vàng SJC hôm nay là bao nhiêu?"},
    {"id": "OUT-6", "type": "Out-of-Domain", "question": "Thời tiết ngày mai ở thành phố Hồ Chí Minh như thế nào?"},
    {"id": "OUT-7", "type": "Out-of-Domain", "question": "Thủ đô của nước Úc là gì?"},
    {"id": "OUT-8", "type": "Out-of-Domain", "question": "Đội tuyển bóng đá nào vô địch World Cup 2022?"},
    {"id": "OUT-9", "type": "Out-of-Domain", "question": "Các bước để pha một ly cà phê phin ngon?"},
    {"id": "OUT-10", "type": "Out-of-Domain", "question": "Lãi suất tiết kiệm của ngân hàng Vietcombank hiện nay là bao nhiêu?"},

    # === Loại 3: Borderline ===
    {"id": "BOR-1", "type": "Borderline", "question": "Sinh viên có hoàn cảnh khó khăn có được miễn giảm học phí không?"},
    {"id": "BOR-2", "type": "Borderline", "question": "Quy định về trang phục khi đi thi của sinh viên?"},
    {"id": "BOR-3", "type": "Borderline", "question": "Thời gian tối đa được mượn một cuốn sách ở thư viện là bao lâu?"},
    {"id": "BOR-4", "type": "Borderline", "question": "Có được cộng điểm rèn luyện nếu tham gia nghiên cứu khoa học không?"},
    {"id": "BOR-5", "type": "Borderline", "question": "Thủ tục xin cấp lại thẻ sinh viên bị mất như thế nào?"},
    {"id": "BOR-6", "type": "Borderline", "question": "Lịch nghỉ Tết Nguyên Đán năm 2026 của trường là khi nào?"},
    {"id": "BOR-7", "type": "Borderline", "question": "Phòng đào tạo của trường nằm ở tòa nhà nào?"},
    {"id": "BOR-8", "type": "Borderline", "question": "Vấn đề học tập của sinh viên?"},
    {"id": "BOR-9", "type": "Borderline", "question": "Trách nhiệm của giảng viên trong việc đảm bảo chất lượng giảng dạy?"},
    {"id": "BOR-10", "type": "Borderline", "question": "Học chương trình tài năng có tốt hơn chương trình chuẩn không?"},
]


def run_calibration():    
    query_system = DprAdaptiveQuerySystem(CONFIG)
    
    print("\nBắt đầu quá trình hiệu chỉnh ngưỡng...")
    print("ID\tLoại\tĐiểm Tương đồng Cao nhất\tCâu hỏi")
    print("--\t----\t--------------------------\t--------")
    
    for item in test_queries:
        question = item['question']
                
        query_vector = query_system.question_encoder.encode(question)
        query_vector_2d = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_vector_2d)
                
        scores, _ = query_system.index.search(query_vector_2d, k=1)
        best_score = scores[0][0]
                
        print(f"{item['id']}\t{item['type']}\t{best_score:.4f}\t\t\t{question}")

if __name__ == '__main__':
    run_calibration()