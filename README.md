# EduRAGBot

## Cách chạy 

### 1. Backend
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