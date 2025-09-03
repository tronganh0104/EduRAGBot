import pdfplumber

file_path = r"backend\data\raw\Thong-tin-tuyen-sinh-DHCQ-nam-2025-QHI.pdf"

all_text = []
all_tables = [] 

with pdfplumber.open(file_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        #cắt văn bản
        text = page.extract_text()
        if text:
            all_text.append({
                "page": page_num,
                "content": text
            })
        #cắt bảng
        tables = page.extract_tables()
        for table in tables:
            all_tables.append({
                "page": page_num,
                "table": table
            })

print("Số đoạn văn bản:", len(all_text))
print("Số bảng trích xuất:", len(all_tables))
print("\n=== Đại diện văn bản ===")
for sample in all_text[:2]:
    print(f"Trang {sample['page']}:")
    print(sample['content'][:500], "...\n")

print("\n=== Đại diện bảng ===")
for sample in all_tables[:2]:
    print(f"Trang {sample['page']}:")
    for row in sample['table'][:5]:
        print(row)
    print("...\n")