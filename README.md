# EduRAGBot

## Cách chạy 

### 1. Backend (Tạm thời chưa cập nhật)
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```
### 2. Frontend
```bash
cd frontend
npm install 
npm run dev
```

### 3. Chạy Backend trên Kaggle
- Upload file run_on_kaggle.ipynb lên kaggle 
- Thêm vào input những nội dung sau
    + Dataset: https://www.kaggle.com/datasets/buitronganh/qcdt-data-ver-2
    + Query Classifier Model: https://www.kaggle.com/models/buitronganh/query-model-ver-1/Transformers/default/1
    + Qwen3 1.7B: https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/1.7b/1
    + Qwen3 4B Pretrain Legal: https://www.kaggle.com/models/trnphmhong/qwen3-4b-legal-pretrain/Transformers/default/1
    + Qwen3 4B Finetune Ver3: https://www.kaggle.com/models/buitronganh/qwen3-4b-finetune-ver3/Transformers/default/1

- Chạy các dòng trong kaggle lần lượt tới cuối trừ ô có nội dung ngrok.kill()
- Sau khi chạy ô cuối sẽ có nội dung output gồm một dòng có nội dung: Public URL: NgrokTunnel: "https://Sample.ngrok-free.app" -> "http://localhost:8000"
- Copy nội dung của NgrokTunnel và dán vào file frontend/app/config.ts
- Nếu muốn khởi động lại thì chạy ô ngrok.kill(), chạy lại ô cuối và lặp lại việc copy đường dẫn vào file config.tsts