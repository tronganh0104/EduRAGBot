from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu')

def generate_answer(question, context=""):
    prompt = f"""Bạn là trợ lý AI trả lời câu hỏi về quy chế đào tạo của trường Đại học Công nghệ, Đại học Quốc gia Hà Nội. Dựa trên ngữ cảnh và câu hỏi dưới đây hãy trả lời câu hỏi.

    Ngữ cảnh: {context}
    Câu hỏi: {question}
    Trả lời: """
    print("\n--- PROMPT (input to model) ---\n" + prompt)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- RAW OUTPUT (from model) ---\n" + answer)
    
    if "Trả lời:" in answer:
        answer = answer.split("Trả lời:")[-1].strip()
    else:
        answer = answer[len(prompt):].strip()

    answer = answer.replace(question, "").strip()
    answer = answer.replace("<think>", "").replace("</think>", "").replace("<|thinking|>", "").replace("<|/thinking|>", "").strip()
    print("\n--- FINAL ANSWER (postprocessed) ---\n" + answer)

    # Extract and print question-like sentences from the answer
    import re
    questions_in_answer = re.findall(r"[^\n.?!]*\?", answer)
    if questions_in_answer:
        print("\n--- QUESTIONS FOUND IN ANSWER ---")
        for q in questions_in_answer:
            print(q.strip())
    else:
        print("\n--- NO QUESTIONS FOUND IN ANSWER ---")
    return answer




