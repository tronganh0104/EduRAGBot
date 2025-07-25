def metadata(chunks, source="data\Raw\ctdt_vnu.pdf"):
    return [{"content": chunk, "metadata": {"source": source}} for chunk in chunks]

# import re

# def metadata(text, source="data/Raw/ctdt_vnu.pdf"):
#     lines = text.split('\n')
    
#     current_chapter = None
#     current_article = None
#     current_title = None
#     current_content_lines = []

#     chunks_with_metadata = []

#     def save_current_article():
#         if current_article and current_content_lines:
#             content = "\n".join(current_content_lines).strip()
#             chunks_with_metadata.append({
#                 "content": content,
#                 "metadata": {
#                     "source": source,
#                     "chuong": current_chapter,
#                     "dieu": current_article,
#                     "tieu_de": current_title
#                 }
#             })

#     for i, line in enumerate(lines):
#         line = line.strip()
#         # Bỏ dòng trống
#         if not line:
#             continue

#         if re.match(r"^Chương\s+[IVXLCDM]+\s*[:\.]?", line, re.IGNORECASE):
#             save_current_article()
#             current_chapter = line
#             current_article = None
#             current_title = None
#             current_content_lines = []
#             continue

#         if re.match(r"^Điều\s+\d+(\.|:)?", line):
#             save_current_article()
#             current_article = line
#             current_title = None
#             current_content_lines = []
#             if i+1 < len(lines):
#                 next_line = lines[i+1].strip()
#                 # nếu không phải là "Điều ..." hoặc "Chương ..." thì coi là tiêu đề
#                 if not re.match(r"^(Điều|Chương)\s+", next_line):
#                     current_title = next_line
#                 else:
#                     current_title = ""
#             continue

#         current_content_lines.append(line)

#     save_current_article()

#     return chunks_with_metadata
