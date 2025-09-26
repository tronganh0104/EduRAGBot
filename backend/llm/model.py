from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
from peft import PeftModel
import os
import torch
import re
from openai import OpenAI

CONFIG_QCDT = {
    "index_path": "/kaggle/input/database-ver-11/vs_qcdt_o/vs_qcdt_o/index.faiss",
    "docs_path": "/kaggle/input/database-ver-11/vs_qcdt_o/vs_qcdt_o/docs.pkl",
    "question_encoder_path": "/kaggle/input/qcctsv_model/pytorch/default/4/models/dpr_merge",
    "question_classifier_path": "/kaggle/input/classifier-model-ver-2/transformers/default/1/question_classifier",
    "reranker_model": "BAAI/bge-reranker-base",
    
    "strategy_config": {
        "Definition": {"k": 5, "top_n": 2},
        "Factoid":    {"k": 5, "top_n": 2},
        "Yes/No":     {"k": 7, "top_n": 3},
        "List":       {"k": 8, "top_n": 4},
        "Inference":  {"k": 12, "top_n": 4}
    },
    
    "relevance_similarity_threshold": 0.29391
}

CONFIG_QCCTSV = {
    "index_path": "/kaggle/input/database-ver-11/vs_qcctsv_o/vs_qcctsv_o/index.faiss",
    "docs_path": "/kaggle/input/database-ver-11/vs_qcctsv_o/vs_qcctsv_o/docs.pkl",
    "question_encoder_path": "/kaggle/input/qcctsv_model/pytorch/default/4/models/dpr_merge",
    "question_classifier_path": "/kaggle/input/classifier-model-ver-2/transformers/default/1/question_classifier",
    "reranker_model": "BAAI/bge-reranker-base",
    "strategy_config": {
        "Definition": {"k": 4, "top_n": 1},
        "Factoid": {"k": 4, "top_n": 2},
        "Yes/No": {"k": 5, "top_n": 3},
        "List": {"k": 7, "top_n": 3},
        "Inference": {"k": 10, "top_n": 3}
    },
    "relevance_similarity_threshold": 0.23833
}

CONFIG_QCTDKT = {
    "index_path": "/kaggle/input/database-ver-11/vs_qctdkt_o/vs_qctdkt_o/index.faiss",
    "docs_path": "/kaggle/input/database-ver-11/vs_qctdkt_o/vs_qctdkt_o/docs.pkl",
    "question_encoder_path": "/kaggle/input/qcctsv_model/pytorch/default/4/models/dpr_merge",
    "question_classifier_path": "/kaggle/input/classifier-model-ver-2/transformers/default/1/question_classifier",
    "reranker_model": "BAAI/bge-reranker-base",
    "strategy_config": {
        "Definition": {"k": 4, "top_n": 1},
        "Factoid": {"k": 4, "top_n": 2},
        "Yes/No": {"k": 5, "top_n": 3},
        "List": {"k": 7, "top_n": 3},
        "Inference": {"k": 10, "top_n": 3}
    },
    "relevance_similarity_threshold": 0.22689
}

CONFIG_QCTS = {
    "index_path": "/kaggle/input/database-ver-10/vs_tuyensinh/index.faiss",
    "docs_path": "/kaggle/input/database-ver-10/vs_tuyensinh/docs.pkl",
    "question_encoder_path": "/kaggle/input/qcctsv_model/pytorch/default/3/models/dpr_tuyensinh",
    "question_classifier_path": "/kaggle/input/classifier-model-ver-2/transformers/default/1/question_classifier",
    "reranker_model": "BAAI/bge-reranker-base",
    "strategy_config": {
        "Definition": {"k": 4, "top_n": 2},
        "Factoid":    {"k": 4, "top_n": 2},
        "Yes/No":     {"k": 5, "top_n": 2},
        "List":       {"k": 7, "top_n": 4},
        "Inference":  {"k": 10, "top_n": 4}
    },
    "relevance_similarity_threshold": 0.71
}

MAIN_CONFIG = {
    "qcdt": {
        "config": CONFIG_QCDT,
        "name": "Đào tạo"
    },
    "qcctsv": {
        "config": CONFIG_QCCTSV,
        "name": "Công tác Sinh viên"
    },
    "qctdkt": {
        "config": CONFIG_QCTDKT,
        "name": "Thi đua Khen thưởng"
    },
    "tuyensinh": {
        "config": CONFIG_QCTS,
        "name": "Tuyển sinh"
    }
}

MODEL_ID_MAP = {
    "Qwen3 4B": "/kaggle/input/qwen3-4b-legal-pretrain/transformers/default/1/qwen3-4b-legal-pretain",
    "Qwen3 4B Finetune": "/kaggle/input/qwen3-4b-finetune-ver-3/transformers/default/1/kaggle/working/fine_tuned_model",
    "Qwen3 8B API": "qwen/qwen3-8b:free",
    "Qwen3 14B API": "qwen/qwen3-14b:free",
    "GPT OSS 20B API": "openai/gpt-oss-20b:free"
}

# Khởi tạo OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)

@lru_cache(maxsize=3)
def get_model_and_tokenizer(model_name):
    model_id = MODEL_ID_MAP.get(model_name, None)
    if model_id is None:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. Các mô hình hợp lệ: {list(MODEL_ID_MAP.keys())}")

    if model_name in ["Qwen3 8B API", "Qwen3 14B API", "GPT OSS 20B API"]:
        return None, None
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    if model_name == "Qwen3 4B Finetune":
        model = AutoModelForCausalLM.from_pretrained(
            "/kaggle/input/qwen3-4b-legal-pretrain/transformers/default/1/qwen3-4b-legal-pretain",
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

def generate_answer(question, model_name="Qwen3 4B"):
    rag_system = RAGSystem(MAIN_CONFIG)
    retrieved_docs = rag_system.ask(question)
    context = "\n".join([doc['content'] for doc in retrieved_docs])
    print(f"\n--- RETRIEVED CONTEXT ---\n{context}")

    messages = [
        {"role": "system", "content": """Bạn là một chuyên gia tư vấn về các quy chế của trường Đại học Công nghệ, Đại học Quốc gia Hà Nội, bao gồm Quy chế Đào tạo, Công tác Sinh viên, và Thi đua Khen thưởng. 
Nhiệm vụ của bạn là trả lời câu hỏi dựa CHỈ vào ngữ cảnh được cung cấp. Không lặp lại câu hỏi, ngữ cảnh, hoặc nội dung prompt trong câu trả lời. Không thêm thông tin ngoài ngữ cảnh, không suy diễn, không tạo ra quy trình hoặc chi tiết không được đề cập rõ ràng.
Hướng dẫn trả lời:
- Trả lời ngắn gọn, bắt đầu bằng câu tóm tắt trực tiếp, sau đó giải thích với trích dẫn chính xác (ví dụ: "Theo Điều X: [trích dẫn ngắn]").
- Tập trung vào quy định, điều kiện, mốc thời gian. Không đề cập thao tác cụ thể (như đăng nhập website).
- Nếu câu hỏi là Yes/No, bắt đầu bằng "Có" hoặc "Không", rồi giải thích.
- Nếu câu hỏi là danh sách, trả lời bằng bullet points.
- Nếu câu hỏi không liên quan đến quy chế (ví dụ: lời chào, câu hỏi chung chung), trả lời: "Vui lòng hỏi về quy chế để tôi hỗ trợ bạn."
- Nếu ngữ cảnh rỗng hoặc không đủ thông tin, trả lời: "Quy chế không quy định chi tiết về vấn đề này, bạn vui lòng tham khảo các kênh thông báo chính thức của Nhà trường."
- Sử dụng tiếng Việt chuẩn, chính tả đúng (không lỗi như 'nghành' thay vì 'ngành'), viết hoa danh từ riêng (ví dụ: Đại học Công nghệ, Quy chế Đào tạo). Giữ giọng lịch sự, chuyên nghiệp, tránh lặp từ/ý, đảm bảo văn bản sạch sẽ, không lỗi ngữ pháp hoặc viết hoa linh tinh."""},
        {"role": "user", "content": f"Ngữ cảnh: {context}\nCâu hỏi: {question}"}
    ]

    if model_name in ["Qwen3 8B API", "Qwen3 14B API", "GPT OSS 20B API"]:
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
        model, tokenizer = get_model_and_tokenizer(model_name)
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
            temperature=0.3,
            top_p=0.7
        )
        print("\n--- RAW OUTPUT BEFORE DECODE ---\n" + str(outputs[0].tolist()))
        answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("\n--- DECODED OUTPUT ---\n" + answer)

        assistant_start = answer.rfind("<|im_start|>assistant")
        if assistant_start != -1:
            answer = answer[assistant_start + len("<|im_start|>assistant"):].strip()
            answer = re.sub(r'<\|im_end\|>', '', answer).strip()
            print("\n--- EXTRACTED ASSISTANT OUTPUT (split by assistant tag) ---\n" + answer)
        else:
            if text in answer:
                answer = answer.split(text)[-1].strip()
                print("\n--- EXTRACTED ASSISTANT OUTPUT (split by prompt) ---\n" + answer)
            else:
                system_prompt = messages[0]["content"]
                user_input = messages[1]["content"]
                answer = answer.replace(system_prompt, "").replace(user_input, "").strip()
                print("\n--- EXTRACTED ASSISTANT OUTPUT (fallback replace) ---\n" + answer)

        answer = re.sub(r'<\|im_start\|>.*?\|>', '', answer, flags=re.DOTALL).strip()
        print("\n--- AFTER REMOVING CHATML TAGS ---\n" + answer)

    answer = re.sub(r'(?i)(<think>|</think>|<\|thinking\|>|<\|/thinking\|>|Assistant:|Trả lời:|Quyết định này|Bản quyền ©.*$)', '', answer, flags=re.DOTALL).strip()
    print("\n--- AFTER REMOVING TAGS ---\n" + answer)

    answer = re.sub(r'BULLET \d+.*?(?=(BULLET \d+|$))', '', answer, flags=re.DOTALL)
    answer = re.sub(r'^\s*[\*\-\•]\s*', '- ', answer, flags=re.MULTILINE)
    print("\n--- AFTER BULLET NORMALIZATION ---\n" + answer)

    sentences = re.split(r'(?<=[\.\?!])\s+', answer.strip())
    if sentences and not sentences[-1].endswith(('.', '!', '?')):
        sentences.pop()
        print("\n--- REMOVED INCOMPLETE LAST SENTENCE ---\n" + str(sentences))

    answer = ' '.join(sentences).strip()
    if answer and not answer.endswith(('.', '!', '?')) and not re.search(r'\n- ', answer):
        answer += '.'

    if not answer or answer == messages[0]["content"].strip() or answer == messages[1]["content"].strip():
        answer = "Không thể tạo câu trả lời hợp lệ, vui lòng thử lại."

    print("\n--- FINAL ANSWER (postprocessed) ---\n" + answer)
    references = [{"id": str(uuid.uuid4()), "content": doc['content']} for doc in retrieved_docs]
    return {"answer": answer, "references": references}