import os
from pathlib import Path

CONFIG = {
    "index_path": Path("data/process/index.faiss"),
    "docs_path": Path("data/process/docs.pkl"),
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker_model": "BAAI/bge-reranker-base",
    "default_dense_k": 10,
    "default_bm25_k": 10,
    "default_top_n": 3
}