from data_processor import load_raw, split_chunks, clean_text, metadata, split_chunks_by_structure
from embedding import load_embedding, embed_documents
from vector_store import save_faiss_index
import numpy as np

file_path = "backend\\data\\raw\\Quy-chế-ĐTĐH-3626.txt"

raw_text = load_raw(file_path)
print(raw_text[:1000])

# 2. Làm sạch văn bản
cleaned_text = clean_text(raw_text)

chunks = split_chunks_by_structure(cleaned_text)
print(f"Chunk tạo được: {len(chunks)}")
for i, chunk in enumerate(chunks[:10]):
    print(f"\n--- Chunk {i+1} ---\n{chunk}") 

docs = metadata(chunks,
    source=file_path,
    # doc_name="Quy chế đào tạo đại học ĐHQGHN",
    # doc_type="Quyết định",
    # doc_number="3626/QĐ-ĐHQGHN",
    # issue_date="2022-10-21",
    # sign_date="2022-11-24T09:55:08+07:00",issuer="Giám đốc ĐHQGHN"
)
for i, doc in enumerate(docs[:10]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Nội dung: {doc['content']}")  # in 200 ký tự đầu
    print(f"Metadata: {doc['metadata']}")


model = load_embedding()
vectors = embed_documents(model, docs)
vectors = np.array(vectors)

save_faiss_index(vectors, docs)
print("Đã lưu FAISS index và metadata.")