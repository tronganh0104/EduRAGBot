import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from functools import lru_cache

# Cấu hình các mô hình
MODEL_ID_MAP = {
    "Qwen3 4B pretrain": "F:/Workspace/EduRAGBot/backend/models/qwen3-4b-legal-pretrain",  # Thay bằng đường dẫn cục bộ
    "Qwen3 1.7B": "Qwen/Qwen2-1.5B-Instruct",  # Thay bằng mô hình hợp lệ trên Hugging Face
    "Qwen3 4B finetune": "F:/Workspace/EduRAGBot/backend/models/qwen3-4b-finetune-ver3"  # Thay bằng đường dẫn cục bộ
}

@lru_cache(maxsize=3)
def get_model_and_tokenizer(model_name):
    """Tải mô hình và tokenizer từ đường dẫn hoặc Hugging Face."""
    model_id = MODEL_ID_MAP.get(model_name, None)
    if model_id is None:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. Các mô hình hợp lệ: {list(MODEL_ID_MAP.keys())}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Nếu là mô hình fine-tune với LoRA, tải adapter
    if model_name == "Qwen3 4B finetune":
        lora_adapter_path = model_id  # Đường dẫn đến adapter LoRA
        model = PeftModel.from_pretrained(base_model, lora_adapter_path, is_trainable=False)
    else:
        model = base_model
    
    return model, tokenizer

def generate_answer(question, context="", model_name="Qwen3 4B finetune"):
    """
    Sinh câu trả lời dựa trên câu hỏi và ngữ cảnh sử dụng mô hình fine-tune.
    
    Args:
        question (str): Câu hỏi cần trả lời.
        context (str): Ngữ cảnh từ các tài liệu truy xuất.
        model_name (str): Tên mô hình trong MODEL_ID_MAP.
    
    Returns:
        str: Câu trả lời đã được xử lý.
    """
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    prompt = f"""Bạn là trợ lý AI trả lời câu hỏi về quy chế đào tạo của trường Đại học Công nghệ, Đại học Quốc gia Hà Nội. Dựa trên ngữ cảnh dưới đây, trả lời câu hỏi một cách ngắn gọn, chính xác và đầy đủ. Nếu ngữ cảnh không cung cấp đủ thông tin, hãy trả lời rằng thông tin không có sẵn và không suy đoán.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: """
    print("\n--- PROMPT (input to model) ---\n" + prompt)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        temperature=0.3,  # Giảm để tăng độ chính xác
        top_p=0.7         # Giảm để giảm ngẫu nhiên
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- RAW OUTPUT (from model) ---\n" + answer)
    
    # Post-process câu trả lời
    if "Trả lời:" in answer:
        answer = answer.split("Trả lời:", 1)[-1].strip()
    else:
        answer = answer[len(prompt):].strip()

    answer = re.sub(r'(<think>|</think>|<\|thinking\|>|<\|/thinking\|>|Quyết định này|Bản quyền ©.*$)', '', answer, flags=re.DOTALL)
    answer = re.sub(r'BULLET \d+.*?(?=(BULLET \d+|$))', '', answer, flags=re.DOTALL)
    answer = re.sub(r'[ \t]+', ' ', answer).strip()
    answer = re.sub(r'\n\s*\n+', '\n', answer).strip()

    print("\n--- FINAL ANSWER (postprocessed) ---\n" + answer)

    # Kiểm tra câu hỏi trong câu trả lời
    questions_in_answer = re.findall(r"[^\n.?!]*\?", answer)
    if questions_in_answer:
        print("\n--- QUESTIONS FOUND IN ANSWER ---")
        for q in questions_in_answer:
            print(q.strip())
    else:
        print("\n--- NO QUESTIONS FOUND IN ANSWER ---")
    
    return answer
