from data.process_pdf import extract_pdf
from data.cleaner import clean_text
from data.chunk import split_chunks
from data.metadata import metadata
from retrieval.embedding import load_embedding, embed_documents
from retrieval.vector_store import save_faiss_index
import numpy as np


pdf_path = "data\Raw\Quy-chế-ĐTĐH-3626.pdf"
raw_text = extract_pdf(pdf_path)
print(raw_text[:1000])  # in thử các kí tự sau khi đã xử lí qua file pdf


cleaned_text = clean_text(raw_text)

chunks = split_chunks(cleaned_text)
print(f"chunk tạo được: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{chunk}") # in ra 3 chunk đầu tiên

docs = metadata(chunks)
print("metadata cho 3 chunk đầu:")
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Nội dung: {doc['content'][:200]}...")  # in 200 ký tự đầu
    print(f"Metadata: {doc['metadata']}")

model = load_embedding()
vectors = embed_documents(model, docs)
vectors = np.array(vectors)

save_faiss_index(vectors, docs)
