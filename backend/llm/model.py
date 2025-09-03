from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
from peft import PeftModel
import os
import torch
import re
from openai import OpenAI

MODEL_ID_MAP = {
    "Qwen3 4B": "/kaggle/input/qwen3-4b-legal-pretrain/transformers/default/1/qwen3-4b-legal-pretain",
    "Qwen3 4B finetune": "/kaggle/input/qwen3-4b-finetune-ver-3/transformers/default/1/kaggle/working/fine_tuned_model",
    "Qwen3 8B API": "qwen/qwen3-8b:free",
    "Qwen3 14B API": "qwen/qwen3-14b:free"
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"  # thay bằng API key của bạn
)

@lru_cache(maxsize=3)
def get_model_and_tokenizer(model_name):
    model_id = MODEL_ID_MAP.get(model_name, None)
    if model_id is None:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. Các mô hình hợp lệ: {list(MODEL_ID_MAP.keys())}")

    if model_name in ["Qwen2 8B API", "Qwen2 14B API"]:
        # Không load model local, chỉ để phân nhánh trong generate_answer
        return None, None
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Đảm bảo pad_token hợp lệ
    
    if model_name == "Qwen2 4B finetune":
        model = AutoModelForCausalLM.from_pretrained(
            "/kaggle/input/qwen2-4b-legal-pretrain/transformers/default/1/qwen2-4b-legal-pretrain",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model,
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    return model, tokenizer
    
def generate_answer(question, model_name=None): 
    query_system = DprAdaptiveQuerySystem(CONFIG) 
    retrieved_docs = query_system.query(question) 
    context = "\n".join([doc.page_content for doc in retrieved_docs]) 
    print(f"\n--- RETRIEVED CONTEXT ---\n{context}") 

    # Định nghĩa messages với prompt hiện tại
    messages = [
        {"role": "system", "content": """Bạn là một chuyên gia tư vấn về Quy chế Đào tạo của trường Đại học Công nghệ, Đại học Quốc gia Hà Nội. 
Nhiệm vụ của bạn là trả lời câu hỏi dựa CHỈ vào ngữ cảnh được cung cấp. Không lặp lại câu hỏi, ngữ cảnh, hoặc nội dung prompt trong câu trả lời. Không thêm thông tin ngoài ngữ cảnh, không suy diễn, không tạo ra quy trình hoặc chi tiết không được đề cập rõ ràng.
Hướng dẫn trả lời:
- Trả lời ngắn gọn, bắt đầu bằng câu tóm tắt trực tiếp, sau đó giải thích với trích dẫn chính xác (ví dụ: "Theo Điều X: [trích dẫn ngắn]").
- Tập trung vào quy định, điều kiện, mốc thời gian. Không đề cập thao tác cụ thể (như đăng nhập website).
- Nếu câu hỏi là Yes/No, bắt đầu bằng "Có" hoặc "Không", rồi giải thích.
- Nếu câu hỏi là danh sách, trả lời bằng bullet points.
- Nếu câu hỏi không liên quan đến quy chế đào tạo (ví dụ: lời chào, câu hỏi chung chung), trả lời: "Vui lòng hỏi về quy chế đào tạo để tôi hỗ trợ bạn."
- Nếu ngữ cảnh rỗng hoặc không đủ thông tin, trả lời: "Quy chế không quy định chi tiết về vấn đề này, bạn vui lòng tham khảo các kênh thông báo chính thức của Nhà trường."
- Giữ giọng điệu lịch sự, chuyên nghiệp, tránh lặp từ hoặc ý.

Ví dụ minh họa:
User: Ngữ cảnh: Điều 10: Sinh viên phải nộp đơn xin nghỉ học trước ngày 15/9. Câu hỏi: Khi nào nộp đơn nghỉ học?
Assistant: Sinh viên phải nộp đơn xin nghỉ học trước ngày 15/9. Theo Điều 10, thời hạn nộp đơn là trước ngày 15/9 để được xem xét.

User: Ngữ cảnh: Điều 5: Sinh viên có quyền nghỉ học tạm thời nếu có lý do chính đáng. Câu hỏi: Sinh viên có quyền nghỉ học tạm thời không?
Assistant: Có. Theo Điều 5, sinh viên có quyền nghỉ học tạm thời nếu có lý do chính đáng.

User: Ngữ cảnh: (rỗng) Câu hỏi: Cách đăng ký môn học?
Assistant: Quy chế không quy định chi tiết về vấn đề này, bạn vui lòng tham khảo các kênh thông báo chính thức của Nhà trường."""},
        {"role": "user", "content": f"Ngữ cảnh: {context}\nCâu hỏi: {question}"}
    ]

    if model_name in ["Qwen3 8B API", "Qwen3 14B API"]:
        print("\n--- MESSAGES (input to API) ---\n" + str(messages))
        completion = client.chat.completions.create(
            model=MODEL_ID_MAP[model_name],
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=512
        )
        answer = completion.choices[0].message.content.strip()
        print("\n--- RAW OUTPUT (from API) ---\n" + answer)
    else:
        # Local model
        model, tokenizer = get_model_and_tokenizer(model_name) 
        
        # Áp dụng chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("\n--- PROMPT (sau apply_chat_template) ---\n" + text) 

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device) 
        outputs = model.generate( 
            **inputs, 
            max_new_tokens=100, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id, 
            no_repeat_ngram_size=4, 
            temperature=0.2, 
            top_p=0.7 
        ) 
        print("\n--- RAW OUTPUT BEFORE DECODE ---\n" + str(outputs[0].tolist()))
        answer = tokenizer.decode(outputs[0], skip_special_tokens=False)  # Giữ special tokens
        print("\n--- DECODED OUTPUT ---\n" + answer)
        
        # Tách phần trả lời của assistant
        # Cách 1: Tìm vị trí cuối cùng của <|im_start|>assistant
        assistant_start = answer.rfind("<|im_start|>assistant")
        if assistant_start != -1:
            answer = answer[assistant_start + len("<|im_start|>assistant"):].strip()
            # Loại bỏ <|im_end|> nếu có
            answer = re.sub(r'<\|im_end\|>', '', answer).strip()
            print("\n--- EXTRACTED ASSISTANT OUTPUT (split by assistant tag) ---\n" + answer)
        else:
            # Cách 2: Loại bỏ prompt đầu vào bằng text
            if text in answer:
                answer = answer.split(text)[-1].strip()
                print("\n--- EXTRACTED ASSISTANT OUTPUT (split by prompt) ---\n" + answer)
            else:
                # Fallback: Loại bỏ system prompt và user input dựa trên nội dung
                system_prompt = messages[0]["content"]
                user_input = messages[1]["content"]
                answer = answer.replace(system_prompt, "").replace(user_input, "").strip()
                print("\n--- EXTRACTED ASSISTANT OUTPUT (fallback replace) ---\n" + answer)

        # Loại bỏ các ký hiệu ChatML còn lại
        answer = re.sub(r'<\|im_start\|>.*?\|>', '', answer, flags=re.DOTALL).strip()
        print("\n--- AFTER REMOVING CHATML TAGS ---\n" + answer)

    # Postprocess
    # 1. Loại bỏ tag thinking và các prefix không cần thiết
    answer = re.sub(r'(?i)(<think>|</think>|<\|thinking\|>|<\|/thinking\|>|Assistant:|Trả lời:|Quyết định này|Bản quyền ©.*$)', '', answer, flags=re.DOTALL).strip()
    print("\n--- AFTER REMOVING TAGS ---\n" + answer)
    
    # 2. Loại bỏ các bullet lạ, chuẩn hóa bullet thành -
    answer = re.sub(r'BULLET \d+.*?(?=(BULLET \d+|$))', '', answer, flags=re.DOTALL)
    answer = re.sub(r'^\s*[\*\-\•]\s*', '- ', answer, flags=re.MULTILINE)
    print("\n--- AFTER BULLET NORMALIZATION ---\n" + answer)
    
    # 3. Loại bỏ câu cuối nếu không hoàn chỉnh
    sentences = re.split(r'(?<=[\.\?!])\s+', answer.strip())
    if sentences and not sentences[-1].endswith(('.', '!', '?')):
        sentences.pop()
        print("\n--- REMOVED INCOMPLETE LAST SENTENCE ---\n" + str(sentences))
    
    # 4. Kết hợp lại
    answer = ' '.join(sentences).strip()
    
    # 5. Đảm bảo kết thúc bằng dấu chấm nếu không phải danh sách
    if answer and not answer.endswith(('.', '!', '?')) and not re.search(r'\n- ', answer):
        answer += '.'
    
    # 6. Kiểm tra nếu output trống hoặc chỉ chứa prompt
    if not answer or answer == messages[0]["content"].strip() or answer == messages[1]["content"].strip():
        answer = "Không thể tạo câu trả lời hợp lệ, vui lòng thử lại."
    
    print("\n--- FINAL ANSWER (postprocessed) ---\n" + answer)

    references = [{"id": str(uuid.uuid4()), "content": doc.page_content} for doc in retrieved_docs] 
    return {"answer": answer, "references": references}