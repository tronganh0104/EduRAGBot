from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

def generate_answer(question, context=""):
    prompt = f"""Bạn là trợ lý AI chuyên về quy chế đào tạo của trường Đại học Công nghệ. Trả lời câu hỏi bằng tiếng Việt, ngắn gọn, chính xác, dựa trên ngữ cảnh. Nếu không có ngữ cảnh hoặc câu hỏi không rõ, trả lời: "Cần thêm thông tin về quy chế đào tạo để trả lời chính xác." Chỉ trả về câu trả lời, không lặp lại câu hỏi.

Ngữ cảnh: {context}
Câu hỏi: {question}
Trả lời: """

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Trả lời:" in answer:
        answer = answer.split("Trả lời:")[-1].strip()
    else:
        answer = answer[len(prompt):].strip()

    answer = answer.replace(question, "").strip()
    answer = answer.replace("<think>", "").replace("</think>", "").replace("<|thinking|>", "").replace("<|/thinking|>", "").strip()
    return answer

