# EduRAGBot

## Cách chạy 

### 1. Frontend
```bash
cd frontend
npm install 
npm run dev
```

### 3. Chạy Backend trên Kaggle
- Upload file run_on_kaggle.ipynb lên kaggle 
- Thêm API của OpenAI, Ngrok và Openrouter vào code
- Thêm vào input những nội dung sau
    + Dataset: https://www.kaggle.com/datasets/buitronganh/database-ver-11
    + Question Encoder Model version 3 và version 4: https://www.kaggle.com/models/buitronganh/qwen3-4b-finetune-ver-4/Transformers/default/1
    + Query Classifier Model: https://www.kaggle.com/models/buitronganh/classifier-model-ver-2/Transformers/default/1
    + Qwen3 4B Pretrain Legal: https://www.kaggle.com/models/trnphmhong/qwen3-4b-legal-pretrain/Transformers/default/1
    + Qwen3 4B Finetune Ver3: https://www.kaggle.com/models/buitronganh/qwen3-4b-finetune-ver-3/Transformers/default/1

- Chạy các dòng trong kaggle lần lượt tới cuối trừ ô có nội dung ngrok.kill()
- Sau khi chạy ô cuối sẽ có nội dung output gồm một dòng có nội dung: Public URL: NgrokTunnel: "https://Sample.ngrok-free.app" -> "http://localhost:8000"
- Copy nội dung của NgrokTunnel và dán vào file frontend/app/config.ts
- Nếu muốn khởi động lại thì chạy ô ngrok.kill(), chạy lại ô cuối và lặp lại việc copy đường dẫn vào file config.ts