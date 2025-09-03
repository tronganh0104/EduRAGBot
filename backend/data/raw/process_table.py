import pdfplumber
import json
import os
def extract_tables_to_json(pdf_path, json_path="backend\data\raw"):
    tables_json = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_id, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue
                
                headers = table[0]
                rows = table[1:]
        
                formatted_rows = []
                for row in rows:
                    row_dict = {}
                    for col_idx, cell in enumerate(row):
                        header = headers[col_idx] if headers[col_idx] else f"col_{col_idx+1}"
                        row_dict[header.strip()] = cell.strip() if cell else None
                    formatted_rows.append(row_dict)
                
                tables_json.append({
                    "page": page_num,
                    "table_id": table_id,
                    "headers": headers,
                    "rows": formatted_rows
                })
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tables_json, f, ensure_ascii=False, indent=2)
    
    return tables_json

pdf_file = r"backend\data\raw\Thong-tin-tuyen-sinh-DHCQ-nam-2025-QHI.pdf"
json_file = "tables.json"
tables = extract_tables_to_json(pdf_file, json_file)
print(f"length text: {len(tables)}")
print(f"length table: {len(tables)}")
print("File JSON được lưu tại:", os.path.abspath(json_file))
