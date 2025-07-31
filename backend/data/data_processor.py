# from cgitb import text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from pypdf import PdfReader
import os

def split_chunks(text, chunk_size=1200, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def split_chunks_by_structure(text):
    chunks = []
    # Tách Chương
    # Tạo một chunk cho phần trước chương I (nếu có)
    preamble = []
    parts = re.split(r'(?=Chương\s+[IVX]+)', text, flags=re.M)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Xác định tiêu đề Chương
        chap_match = re.match(r'Chương\s+[IVX]+(?:[^\n]*)', part)
        if not chap_match:
            # Preamble
            preamble.append(part)
            continue
        chapter = chap_match.group().strip()
        body_after_chap = part[len(chapter):]

        # Tách Điều trong mỗi Chương
        articles = re.split(r'(?=Điều\s+\d+\.)', body_after_chap, flags=re.M)
        for art in articles:
            art = art.strip()
            if not art:
                continue
            art_match = re.match(r'(Điều\s+\d+\.[^\n]*)', art)
            if art_match:
                article = art_match.group(1).strip()
                content = art[len(article):].strip()
                chunks.append({
                    'chapter': chapter,
                    'article': article,
                    'content': content
                })
            else:
                if chunks and chunks[-1]['chapter'] == chapter and chunks[-1]['article']:
                    chunks[-1]['content'] += '\n' + art
                else:
                    chunks.append({
                        'chapter': chapter,
                        'article': None,
                        'content': art
                    })
    if preamble:
        chunks.insert(0, {
            'chapter': 'Preamble',
            'article': None,
            'content': '\n\n'.join(preamble)
        })
    return chunks


def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)  
    text = re.sub(r"[ \t]+", " ", text)  
    text = text.strip()
    return text

# def metadata(chunks, source):
#     return [{"content": chunk, "metadata": {"source": source}} for chunk in chunks]

def metadata(chunks, source):
    docs = []
    for chunk in chunks:
        text = chunk["content"] if isinstance(chunk, dict) else str(chunk)
        # gom các metadata gốc
        meta = {
            "source": source,
            **{k: chunk[k] for k in ("chapter", "article", "clause") if k in chunk}
        }
        docs.append({"content": text, "metadata": meta})
    return docs


def extract_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text

def load_txt(file_txt):
    raw =  open(file_txt, 'r', encoding='utf-8').read()
    return clean_text(raw)

def load_raw(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")