import faiss
import numpy as np
import pickle

def save_faiss_index(vectors, docs, index_path="backend/data/process/index.faiss", doc_path="backend/data/process/docs.pkl"):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)

    with open(doc_path, "wb") as f:
        pickle.dump(docs, f)