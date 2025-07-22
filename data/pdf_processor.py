from cgitb import text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from pypdf import PdfReader

def split_chunks(text, chunk_size=900, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)  
    text = re.sub(r"[ \t]+", " ", text)  
    text = text.strip()
    return text

def metadata(chunks, source="data/raw/ctdt_vnu.pdf"):
    return [{"content": chunk, "metadata": {"source": source}} for chunk in chunks]

def extract_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text