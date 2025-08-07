import os
import re
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CONFIG = {
    "source_file_path": "data/Raw/Quy-chế-ĐTĐH-3626.txt", 
    "output_dir": "vector_store_data_v3", 
    "index_file": "index.faiss",
    "docs_file": "docs.pkl",
    "embedding_model_path": "models/dpr-phobert-augmented",
    "max_chunk_size": 2000
}

def extract_and_clean_from_txt(txt_path):
    print(f"Đang đọc và làm sạch file: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại {txt_path}")
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    print("--- Đọc và làm sạch cơ bản hoàn tất ---")
    return text.strip()

def semantic_chunking(text: str):
    print("Bắt đầu chia chunk theo ngữ nghĩa (Semantic Chunking)...")
    pattern = r"(Điều \d+\..*?)(?=(Điều \d+\.|$))"
    articles = re.finditer(pattern, text, re.DOTALL)
    
    chunks_with_metadata = []
    current_chapter = "NHỮNG QUY ĐỊNH CHUNG"
    chunk_id = 0

    for article_match in articles:
        article_text = article_match.group(1).strip()
        pos = article_match.start()
        chapters_before = re.findall(r'Chương ([IVXLC\d]+.*?)\n', text[:pos])
        if chapters_before:
            current_chapter = chapters_before[-1].strip()
        article_title_match = re.search(r'^(Điều \d+\..*)', article_text)
        article_title = article_title_match.group(1).replace('\n', ' ').strip() if article_title_match else "Không xác định"
        
        metadata = {
            "chunk_id": chunk_id,
            "source": os.path.basename(CONFIG["source_file_path"]),
            "chapter": current_chapter,
            "article": article_title
        }
        chunk = {"content": article_text, "metadata": metadata}
        chunks_with_metadata.append(chunk)
        chunk_id += 1
    
    return chunks_with_metadata

def create_vector_store_with_custom_model(docs: list):
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = CONFIG['embedding_model_path']
    print(f"\nĐang tải Passage Encoder đã được fine-tune từ: {model_path}")
    passage_encoder = SentenceTransformer(model_path)
    
    texts = [d["content"] for d in docs if d["content"]]
    
    embeddings = passage_encoder.encode(
        texts, 
        convert_to_tensor=True, 
        show_progress_bar=True,
        batch_size=32 
    )
    
    vectors = embeddings.detach().cpu().numpy().astype('float32')

    dim = vectors.shape[1]
    print(f"Chiều của vector (dimension): {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    
    index_path = output_dir / CONFIG["index_file"]
    docs_path = output_dir / CONFIG["docs_file"]

    print(f"Đang lưu FAISS index vào: {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Đang lưu documents và metadata vào: {docs_path}")
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
        
    print(f"\n--- Đã tạo và lưu Vector Store phiên bản mới (v3) thành công tại '{output_dir}'! ---")

if __name__ == '__main__':
    raw_text = extract_and_clean_from_txt(CONFIG["source_file_path"])
    
    if raw_text:
        documents = semantic_chunking(raw_text)
        
        print("\n--- KIỂM TRA MỘT CHUNK MẪU ---")
        if documents:
            print(f"Nội dung: {documents[0]['content'][:300]}...")
            print(f"Metadata: {documents[0]['metadata']}")
        
        create_vector_store_with_custom_model(documents)
