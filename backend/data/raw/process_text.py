import pdfplumber

file_path = r"backend\data\raw\Thong-tin-tuyen-sinh-DHCQ-nam-2025-QHI.pdf"
output_txt = "tuyen_sinh_text.txt"

all_text = []

with pdfplumber.open(file_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            all_text.append(f"=== Trang {page_num} ===\n{text}\n")

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print(f"Đã lưu toàn bộ văn bản vào file: {output_txt}")
