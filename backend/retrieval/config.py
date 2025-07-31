from pathlib import Path

CONFIG = {
    "index_path": Path("/kaggle/input/processed-data/index.faiss"),
    "docs_path": Path("/kaggle/input/processed-data/docs.pkl"),
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker_model": "BAAI/bge-reranker-base",
    "retriever_k": 5,
    "reranker_top_n": 2
}