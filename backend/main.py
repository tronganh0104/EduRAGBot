from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

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
    context: str = ""

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    fake_answer = f"Bạn đang hỏi {request.question}"
    return {"answer": fake_answer}



