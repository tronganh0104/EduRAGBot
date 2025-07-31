from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
from peft import PeftModel
import os
import torch
import re

MODEL_ID_MAP = {
    "Qwen3 4B pretrain": "/kaggle/input/qwen3-4b-legal-pretrain/transformers/default/1/qwen3-4b-legal-pretain",
    "Qwen3 1.7B": "/kaggle/input/qwen-3/transformers/1.7b/1",
    "Qwen3 4B finetune": "/kaggle/input/qwen3-4b-finetune-ver3/transformers/default/1/kaggle/working/qwen_vietnamese_qa",
}

@lru_cache(maxsize=3)
def get_model_and_tokenizer(model_name):
    model_id = MODEL_ID_MAP.get(model_name, None)
    if model_id is None:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. Các mô hình hợp lệ: {list(MODEL_ID_MAP.keys())}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer

def generate_answer(question, context="", model_name=None):
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    prompt = f"""Bạn là trợ lý AI trả lời câu hỏi về quy chế đào tạo của trường Đại học Công nghệ, Đại học Quốc gia Hà Nội. Dựa trên ngữ cảnh dưới đây, trả lời câu hỏi một cách ngắn gọn và chính xác. Nếu ngữ cảnh không cung cấp đủ thông tin, hãy trả lời rằng thông tin không có sẵn và không suy đoán. Không tự sinh thêm câu hỏi, chỉ sinh câu trả lời và chỉ sinh một câu trả lời duy nhất.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: """
    print("\n--- PROMPT (input to model) ---\n" + prompt)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        temperature=0.3,  
        top_p=0.5        
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- RAW OUTPUT (from model) ---\n" + answer)

    if "Trả lời:" in answer:
        answer = answer.split("Trả lời:", 1)[-1].strip()
    else:
        answer = answer[len(prompt):].strip()

    answer = re.sub(r'(<think>|</think>|<\|thinking\|>|<\|/thinking\|>|Quyết định này|Bản quyền ©.*$)', '', answer, flags=re.DOTALL)
    answer = re.sub(r'BULLET \d+.*?(?=(BULLET \d+|$))', '', answer, flags=re.DOTALL)
    answer = re.sub(r'[ \t]+', ' ', answer).strip()
    answer = re.sub(r'\n\s*\n+', '\n', answer).strip()

    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    if sentences and not re.search(r'[.!?]$', sentences[-1]):
        sentences = sentences[:-1]  # Loại bỏ câu cuối không hoàn chỉnh
    answer = ' '.join(sentences).strip()
    
    print("\n--- FINAL ANSWER (postprocessed) ---\n" + answer)

    questions_in_answer = re.findall(r"[^\n.?!]*\?", answer)
    if questions_in_answer:
        print("\n--- QUESTIONS FOUND IN ANSWER ---")
        for q in questions_in_answer:
            print(q.strip())
    else:
        print("\n--- NO QUESTIONS FOUND IN ANSWER ---")
    
    return answer