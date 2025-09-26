from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm.model import generate_answer
import uuid
from typing import List
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# rag_system = RAGSystem(MAIN_CONFIG)

class QuestionRequest(BaseModel):
    question: str
    context: str = ""
    model: str = "Qwen3 4B"

class Reference(BaseModel):
    id: str
    content: str

class AnswerResponse(BaseModel):
    answer: str
    references: List[Reference]

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    result = generate_answer(request.question, model_name=request.model)
    return AnswerResponse(answer=result["answer"], references=result["references"])