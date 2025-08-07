import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm

CONFIG = {
    "input_data_path": "data/dpr_dataset_with_negatives.json",
    "augmented_data_path": "data/dpr_dataset_augmented.json",
    "num_paraphrases_per_question": 2 
}

def initialize_llm():
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Biến môi trường GOOGLE_API_KEY chưa được thiết lập. Vui lòng thiết lập API key của bạn.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Đã khởi tạo thành công mô hình Gemini 1.5 Flash.")
        return model
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo mô hình Gemini. Lỗi: {e}")
        return None

def load_original_dataset(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Đã đọc thành công {len(dataset)} điểm dữ liệu từ '{path}'.")
        return dataset
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{path}'. Hãy chắc chắn bạn đã chạy các script trước đó.")
        return None

def augment_dataset(model, original_dataset, num_paraphrases):
    if not model or not original_dataset:
        return None

    augmented_dataset = []
    print(f"\nBắt đầu quá trình Data Augmentation cho {len(original_dataset)} câu hỏi...")

    for data_point in tqdm(original_dataset, desc="Augmenting Questions"):
        original_question = data_point['question']
        augmented_dataset.append(data_point)
        prompt = f"""Bạn là một chuyên gia về ngôn ngữ tiếng Việt.
Nhiệm vụ của bạn là đọc câu hỏi gốc sau đây và viết lại nó theo {num_paraphrases} cách khác nhau.

QUY TẮC BẮT BUỘC:
1. Các câu hỏi viết lại phải giữ nguyên hoàn toàn ý nghĩa và ý định cốt lõi của câu hỏi gốc.
2. Chỉ trả về các câu hỏi đã được viết lại, mỗi câu hỏi trên một dòng.
3. Không thêm số thứ tự, gạch đầu dòng, hoặc bất kỳ lời giải thích nào.

Câu hỏi gốc: "{original_question}"
"""
        try:
            response = model.generate_content(prompt)
            paraphrased_questions = response.text.strip().split('\n')
            for pq in paraphrased_questions:
                pq_cleaned = pq.strip()
                if pq_cleaned and pq_cleaned != original_question:
                    new_data_point = {
                        "question": pq_cleaned,
                        "positive_passage": data_point['positive_passage'],
                        "hard_negatives": data_point.get('hard_negatives', [])
                    }
                    augmented_dataset.append(new_data_point)
            time.sleep(1.5)
        except Exception as e:
            print(f"\nLỗi khi xử lý câu hỏi '{original_question}': {e}")
            continue
            
    return augmented_dataset

if __name__ == '__main__':
    llm_model = initialize_llm()
    if llm_model:
        original_data = load_original_dataset(CONFIG["input_data_path"])
        if original_data:
            final_augmented_data = augment_dataset(llm_model, original_data, CONFIG["num_paraphrases_per_question"])
            if final_augmented_data:
                output_path = CONFIG["augmented_data_path"]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_augmented_data, f, ensure_ascii=False, indent=4)
                print("\n=============================================")
                print(f"Hoàn tất Data Augmentation.")
                print(f"   - Từ {len(original_data)} câu hỏi gốc")
                print(f"   - Đã tạo ra một bộ dữ liệu mới có {len(final_augmented_data)} điểm dữ liệu.")
                print(f"   - Dữ liệu đã được lưu vào: '{output_path}'")
