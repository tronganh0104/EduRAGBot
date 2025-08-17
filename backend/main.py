from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm.model import generate_answer
from retrieval.query_system import QuerySystem
from retrieval.config import CONFIG
import uuid

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
    model: str = ""

class Reference(BaseModel):
    id: str
    content: str

class AnswerResponse(BaseModel):
    answer: str
    references: list[Reference]

query_system = QuerySystem(CONFIG)

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    retrieved_docs = query_system.query(request.question)

    context = request.context + "\n\nThông tin liên quan từ tài liệu:\n"
    references = []
    for i, doc in enumerate(retrieved_docs):
        context += f"Chunk {i+1}: {doc.page_content}\n\n"
        print(f"Chunk {i+1}: {doc.page_content}")
        references.append(Reference(id=str(uuid.uuid4()), content=doc.page_content))
    
    answer = generate_answer(request.question, context)
    return {"answer": answer, "references": references}