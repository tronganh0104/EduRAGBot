import os
import re
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- CẤU HÌNH TRUNG TÂM ---
CONFIG = {    
    "source_file_path": "EduRAGBot\data\Raw\Quy-chế-ĐTĐH-3626.txt", 
    "output_dir": "vector_store_data_v2",
    "index_file": "index.faiss",
    "docs_file": "docs.pkl",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
    return text.strip()

def semantic_chunking(text: str, max_size: int):
    """
    Chia văn bản quy chế thành các chunk dựa trên cấu trúc logic (Chương, Điều).
    """    
    pattern = r"(Điều \d+\..*?)(?=(Điều \d+\.|$))"
    articles = re.finditer(pattern, text, re.DOTALL)
    
    chunks_with_metadata = []
    current_chapter = "NHỮNG QUY ĐỊNH CHUNG"

    for article_match in articles:
        article_text = article_match.group(1).strip()
        
        pos = article_match.start()
        chapters_before = re.findall(r'Chương ([IVXLC\d]+.*?)\n', text[:pos])
        if chapters_before:
            current_chapter = chapters_before[-1].strip()

        article_title_match = re.search(r'^(Điều \d+\..*)', article_text)
        article_title = article_title_match.group(1) if article_title_match else "Không xác định"
        
        metadata = {
            "source": os.path.basename(CONFIG["source_file_path"]),
            "chapter": current_chapter.replace('\n', ' ').strip(),
            "article": article_title.replace('\n', ' ').strip()
        }
        
        chunk = {
            "content": article_text,
            "metadata": metadata
        }
        chunks_with_metadata.append(chunk)

    print(f"--- Đã tạo được {len(chunks_with_metadata)} chunks dựa trên các 'Điều' ---")
    return chunks_with_metadata

def create_and_save_vector_store(docs: list):
    """
    Sử dụng SentenceTransformer để mã hóa các chunk và lưu vào FAISS index.
    """
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Đang tải model embedding: {CONFIG['embedding_model']}")
    model = SentenceTransformer(CONFIG['embedding_model'])

    print("Bắt đầu mã hóa các documents...")
    texts = [d["content"] for d in docs if d["content"]]
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    
    vectors = np.array(embeddings, dtype='float32')

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    
    index_path = output_dir / CONFIG["index_file"]
    docs_path = output_dir / CONFIG["docs_file"]

    faiss.write_index(index, str(index_path))    
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
        
    print("--- Đã tạo và lưu Vector Store thành công! ---")

if __name__ == '__main__':    
    raw_text = extract_and_clean_from_txt(CONFIG["source_file_path"])
    if raw_text:        
        documents = semantic_chunking(raw_text, CONFIG["max_chunk_size"])        
        print("\n--- KIỂM TRA 3 CHUNK ĐẦU TIÊN ---")
        for i, doc in enumerate(documents[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Nội dung: {doc['content'][:400]}...")
            print(f"Metadata: {doc['metadata']}")
        print("\n" + "="*50)
        #save        
        create_and_save_vector_store(documents)