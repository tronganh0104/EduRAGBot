import tabula
import pandas as pd
import json

file_path = r"backend\data\raw\Thong-tin-tuyen-sinh-DHCQ-nam-2025-QHI.pdf"
output_json = "tuyen_sinh_tables_tabula.json"

# Đọc tất cả bảng trong file PDF (mỗi bảng -> DataFrame)
tables = tabula.read_pdf(file_path, pages="all", multiple_tables=True, lattice=True)

all_tables = []
for idx, df in enumerate(tables, start=1):
    # Chuyển DataFrame thành dict (JSON-friendly)
    table_dict = {
        "table_id": idx,
        "header": list(df.columns),
        "rows": df.fillna("").values.tolist()
    }
    all_tables.append(table_dict)

# Lưu ra JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_tables, f, ensure_ascii=False, indent=2)

print(f"Đã lưu {len(all_tables)} bảng vào file: {output_json}")
