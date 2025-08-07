# file: scripts/2_mine_hard_negatives.py

import os
import json
import pickle
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

INITIAL_DATA_PATH = "data/initial_qa_pairs.json"
ALL_CHUNKS_PATH = "vector_store_data_v2/docs.pkl"
OUTPUT_DATA_PATH = "data/dpr_dataset_with_negatives.json"
NUM_HARD_NEGATIVES = 5
BM25_TOP_K = 20

def mine_hard_negatives():    
    print("Bắt đầu quá trình Mining Hard Negatives...")

    if not os.path.exists(INITIAL_DATA_PATH):
        print(f"LỖI: Không tìm thấy file dữ liệu gốc tại: {INITIAL_DATA_PATH}")
        return
    if not os.path.exists(ALL_CHUNKS_PATH):
        print(f"LỖI: Không tìm thấy file docs.pkl tại: {ALL_CHUNKS_PATH}")
        return

    with open(INITIAL_DATA_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)    

    with open(ALL_CHUNKS_PATH, 'rb') as f:
        all_docs = pickle.load(f)
        corpus = [doc['content'] for doc in all_docs]    
    tokenized_corpus = [word_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index đã sẵn sàng.")

    final_dataset = []
    processed_count = 0
    for i, pair in enumerate(qa_pairs):
        question = pair.get('question')        
        positive_passage = pair.get('answer')         

        if not question or not positive_passage:
            print(f"Cảnh báo: Bỏ qua cặp dữ liệu thứ {i+1} do thiếu 'question' hoặc 'answer'.")
            continue
            
        print(f"Đang xử lý câu hỏi {i+1}/{len(qa_pairs)}: '{question[:50]}...'")
        
        tokenized_query = word_tokenize(question)
        top_passages_by_bm25 = bm25.get_top_n(tokenized_query, corpus, n=BM25_TOP_K)
        
        hard_negatives = []
        for passage in top_passages_by_bm25:
            p_strip = passage.strip()
            pos_strip = positive_passage.strip()
            if p_strip != pos_strip and not p_strip.startswith(pos_strip):
                hard_negatives.append(passage)
            if len(hard_negatives) >= NUM_HARD_NEGATIVES:
                break
        
        if len(hard_negatives) > 0:            
            final_dataset.append({
                "question": question,
                "positive_passage": positive_passage,
                "hard_negatives": hard_negatives
            })
            processed_count += 1
        else:
            print(f"Cảnh báo: Không tìm thấy hard negative nào cho câu hỏi '{question[:50]}...'.")

    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    
    with open(OUTPUT_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print("\n=============================================")
    print(f"Hoàn tất Mining Hard Negatives.")
    print(f"Đã xử lý thành công {processed_count}/{len(qa_pairs)} câu hỏi.")
    print(f"Dữ liệu đã được lưu vào: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    mine_hard_negatives()