from pdf_processor import extract_pdf, split_chunks, clean_text, metadata
from embedding import load_embedding, embed_documents
from vector_store import save_faiss_index
import numpy as np 

pdf_path = "data/raw/ctdt_vnu.pdf"
raw_text = extract_pdf(pdf_path)
print(raw_text[:1000])

cleaned_text = clean_text(raw_text)

chunks = split_chunks(cleaned_text)
print(f"Chunk tạo được: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{chunk}") # in ra 3 chunk đầu tiên

docs = metadata(chunks)
print("Metadata cho 3 chunk đầu")
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Nội dung: {doc['content'][:200]}...")  # in 200 ký tự đầu
    print(f"Metadata: {doc['metadata']}")

model = load_embedding()
vectors = embed_documents(model, docs)
vectors = np.array(vectors)

save_faiss_index(vectors, docs)