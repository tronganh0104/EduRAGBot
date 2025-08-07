import json
import os
import shutil
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from tqdm.auto import tqdm

CONFIG = {
    "model_name": "vinai/phobert-base-v2",
    "data_path": "data/dpr_dataset_augmented.json",
    "save_path": "models/dpr-phobert-augmented",
    "num_epochs": 4,
    "batch_size": 16,
    "max_seq_length": 256
}

def prepare_data(data_path):
    print(f"Đang đọc và chuẩn bị dữ liệu từ '{data_path}'...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{data_path}'.")
        return None

    train_examples = []
    for data_point in tqdm(dataset, desc="Preparing data"):
        question = data_point['question']
        positive_passage = data_point['positive_passage']
        train_examples.append(InputExample(texts=[question, positive_passage]))

    print(f"Đã chuẩn bị được {len(train_examples)} cặp (anchor, positive) để huấn luyện.")
    return train_examples

def train_model():
    train_samples = prepare_data(CONFIG["data_path"])
    if not train_samples:
        return
        
    print(f"Đang xây dựng mô hình từ nền tảng: {CONFIG['model_name']}")
    word_embedding_model = models.Transformer(CONFIG['model_name'], max_seq_length=CONFIG['max_seq_length'])
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=CONFIG['batch_size'])
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    num_epochs = CONFIG['num_epochs']
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    print("\n=============================================")
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH DPR...")
    print(f"Số epochs: {num_epochs}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Lưu model tại: {CONFIG['save_path']}")
    print("=============================================")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=CONFIG['save_path'],
        show_progress_bar=True,
        checkpoint_path=f"{CONFIG['save_path']}/checkpoints",
        checkpoint_save_steps=500,
        checkpoint_save_total_limit=3
    )
    
    print(f"\nĐã huấn luyện xong. Mô hình đã được lưu vào: '{CONFIG['save_path']}'")

if __name__ == '__main__':
    train_model()
