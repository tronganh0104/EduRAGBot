# from sentence_transformers import SentenceTransformer
# import numpy as np

# def load_embedding():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# def embed_documents(model, docs):
#     texts = [d["content"] for d in docs if d["content"].strip() != ""]
#     embeddings = model.encode(texts, convert_to_tensor=False)
#     return np.array(embeddings) 

from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding(model_path="backend\data\dpr-phobert-augmented"):
    """
    Load SentenceTransformer model từ local (đã fine-tune).
    - model_path: đường dẫn đến thư mục chứa model đã fine-tune
    """
    return SentenceTransformer(model_path)

def embed_documents(model, docs):
    """
    Encode documents thành embeddings.
    """
    texts = [d["content"] for d in docs if d["content"].strip() != ""]
    embeddings = model.encode(
        texts,
        convert_to_tensor=False,  # Nếu muốn tensor thì để True
        show_progress_bar=True
    )
    return np.array(embeddings)