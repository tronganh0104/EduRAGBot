from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(model, docs):
    texts = [d["content"] for d in docs if d["content"].strip() != ""]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings)  # đảm bảo trả về numpy array 2D
