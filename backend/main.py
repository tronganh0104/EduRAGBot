from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm.model import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    context: str = "Bạn là một chatbot có khả năng giúp tôi trong việc hỏi đáp về quy chế đào tạo của trường Đại học Công nghệ"

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    answer = generate_answer(request.question, request.context)
    return {"answer": answer}



