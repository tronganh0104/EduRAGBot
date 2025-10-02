# EduRAGBot - Chatbot Hỗ Trợ Hỏi Đáp Quy Chế Bằng RAG

## Giới thiệu

EduRAGBot là hệ thống chatbot được phát triển để hỗ trợ hỏi đáp về các quy chế, sử dụng kỹ thuật Retrieval-Augmented Generation (RAG). Dự án được xây dựng nhằm cung cấp câu trả lời chính xác, nhanh chóng và phù hợp dựa trên cơ sở dữ liệu quy chế được cung cấp, phù hợp cho các ứng dụng giáo dục hoặc tổ chức. Hệ thống bao gồm frontend (giao diện người dùng) và backend (xử lý logic và truy vấn RAG), được triển khai với các mô hình ngôn ngữ lớn (LLM) và công cụ hỗ trợ như Kaggle, Ngrok.

## Tính năng
- Hỏi đáp tự động về quy chế dựa trên cơ sở dữ liệu được cung cấp.
- Sử dụng mô hình RAG để kết hợp truy xuất thông tin và sinh câu trả lời tự nhiên.
- Giao diện người dùng thân thiện, dễ sử dụng.
- Hỗ trợ triển khai trên môi trường Kaggle cho backend và chạy cục bộ cho frontend.

## Công nghệ sử dụng
- Frontend: React, TypeScript.
- Backend: Python, các thư viện AI/ML như Transformers (Hugging Face), PyTorch.
- Mô hình AI:
    - Qwen3 4B Pretrain Legal.
    - Qwen3 4B Finetune (version 3 & 4).
- Công cụ triển khai: Kaggle (backend), Ngrok (tạo tunnel để kết nối frontend-backend).
- API: OpenAI, Openrouter.
- Cơ sở dữ liệu: Dataset quy chế trên Kaggle.

## Yêu cầu hệ thống
- Node.js (>= 18.x) và npm để chạy frontend.
- Tài khoản Kaggle với quyền truy cập vào dataset và các mô hình được liệt kê.
- API key từ OpenAI và Openrouter.
- Ngrok để tạo tunnel kết nối backend với frontend.

## Cài đặt và chạy chương trình

### 1. Chuẩn bị môi trường
- Clone repository:
```bash
git clone https://github.com/[your-username]/EduRAGBot.git
cd EduRAGBot
```
- Cài đặt Node.js và npm (nếu chưa có): Tải Node.js.
- Tạo tài khoản Ngrok và lấy auth token: Ngrok.
- Đảm bảo có tài khoản Kaggle và API key từ OpenAI, Openrouter.

### 2. Chạy Frontend
1. Di chuyển vào thư mục frontend:
```bash
cd frontend
```
2. Cài đặt các thư viện cần thiết:
```bash
npm install
```
3. Chạy frontend ở chế độ phát triển:
```bash
npm run dev
```
Mở trình duyệt và truy cập vào URL được cung cấp

### 3. Chạy Backend trên Kaggle
1. Chuẩn bị notebook
- Tải file run_on_kaggle.ipynb từ repository.
- Upload file này lên Kaggle: Kaggle Notebooks.
2. Thêm API key:
- Trong notebook, thêm API key của OpenAI, Openrouter, và auth token của Ngrok vào các biến tương ứng.
3. Thêm input dữ liệu và mô hình:
- Thêm vào input những nội dung sau
    + Dataset: https://www.kaggle.com/datasets/buitronganh/database-ver-11
    + Question Encoder Model version 3 và version 4: https://www.kaggle.com/models/buitronganh/qwen3-4b-finetune-ver-4/Transformers/default/1
    + Query Classifier Model: https://www.kaggle.com/models/buitronganh/classifier-model-ver-2/Transformers/default/1
    + Qwen3 4B Pretrain Legal: https://www.kaggle.com/models/trnphmhong/qwen3-4b-legal-pretrain/Transformers/default/1
    + Qwen3 4B Finetune Ver3: https://www.kaggle.com/models/buitronganh/qwen3-4b-finetune-ver-3/Transformers/default/1

4. Chạy notebook:
- Thực thi các cell trong notebook lần lượt từ đầu đến cuối, trừ cell chứa lệnh ngrok.kill().
- Sau khi chạy cell cuối, bạn sẽ nhận được output chứa URL công khai của Ngrok, ví dụ:
```bash
Public URL: NgrokTunnel: "https://Sample.ngrok-free.app" -> "http://localhost:8000"
```
5. Cập nhật URL vào frontend:
- Copy URL của Ngrok (ví dụ: https://Sample.ngrok-free.app) và dán vào file frontend/app/config.ts tại biến tương ứng (thường là backendUrl).
- Lưu file và đảm bảo frontend đã chạy để kết nối với backend.
6. Khởi động lại backend (nếu cần):
- Nếu cần reset kết nối, chạy cell chứa ngrok.kill() trong notebook.
- Chạy lại cell cuối để tạo URL Ngrok mới.
- Cập nhật URL mới vào file frontend/app/config.ts.

### 4. Kiểm tra và sử dụng
- Mở giao diện frontend trên trình duyệt và thử nhập các câu hỏi về quy chế.
- Ví dụ:
    - Câu hỏi: "Quy định về thời gian nộp bài tập là gì?"
    - Trả lời: [Chatbot sẽ trả lời dựa trên dữ liệu quy chế từ dataset].
- Nếu gặp lỗi, kiểm tra:
    - Kết nối Ngrok có hoạt động không.
    - API key có đúng và còn hiệu lực không.
    - Các mô hình và dataset đã được thêm đúng vào Kaggle.
### Lưu ý
- Bảo mật: Không đẩy API key hoặc auth token của Ngrok lên GitHub. Sử dụng file .env hoặc biến môi trường để lưu trữ.
- Hiệu suất: Đảm bảo tài khoản Kaggle có đủ tài nguyên để chạy các mô hình lớn.
- Khởi động lại: Nếu Ngrok hết hạn hoặc gặp lỗi kết nối, chạy lại cell tạo tunnel và cập nhật URL.
- Tùy chỉnh: Nếu cần thay đổi dataset hoặc mô hình, cập nhật link tương ứng trong notebook.

## Nhóm phát triển
- Phạm Anh Quân - Xây dựng cơ sở dữ liệu
    - Thu thập các văn bản quy chế (Quy chế Đào tạo, Tuyển sinh, Công tác Sinh viên, Khen thưởng, Kỷ luật) từ nguồn đáng tin cậy.
    - Tiền xử lý dữ liệu: loại bỏ nhiễu (ký tự đặc biệt, watermark), chuẩn hóa định dạng (UTF-8, plain text), xử lý bảng biểu và tài liệu scan.
    - Phân tách văn bản thành các chunk theo cấu trúc logic (Điều, Khoản, Mục).
    - Mã hóa chunk thành vector bằng mô hình embedding (PhoBERT hoặc Sentence-BERT).
    - Lưu trữ vector trong cơ sở dữ liệu vector (FAISS) kèm metadata (số Điều, ngày ban hành) để hỗ trợ truy xuất chính xác.
- Nguyễn Trường An - Xây dựng hệ thống truy vấn RAG
    - Thiết kế kiến trúc Multi-RAG với cơ chế Router để phân loại câu hỏi theo lĩnh vực (Đào tạo, Tuyển sinh, Công tác Sinh viên) và ý định (tra cứu, suy luận, liệt kê).
    - Phát triển cơ chế truy xuất thích ứng, tự động xác định số lượng tài liệu cần lấy dựa trên độ phức tạp của câu hỏi.
    - Xây dựng lớp kiểm duyệt với ngưỡng tương đồng để phát hiện câu hỏi ngoài phạm vi, giảm thiểu hallucination.
    - Tích hợp bước reranking sử dụng Cross-Encoder để sắp xếp tài liệu, ưu tiên các đoạn văn bản liên quan nhất.
- Bùi Trọng Anh - Xây dựng giao diện, tích hợp hệ thống và tinh chỉnh mô hình
    - Phát triển giao diện người dùng thân thiện, hỗ trợ nhập câu hỏi tiếng Việt và nhận phản hồi tức thời.
    - Tích hợp backend với các thành phần: cơ sở dữ liệu vector, pipeline Multi-RAG, và mô hình Qwen3-4B.
    - Tinh chỉnh mô hình Qwen3-4B bằng kỹ thuật LoRA trên bộ dữ liệu 5.000 cặp hỏi–đáp, đảm bảo câu trả lời ngắn gọn, tự nhiên, bám sát quy chế và hạn chế hallucination.
    - Kiểm thử hệ thống: đánh giá tốc độ phản hồi, độ chính xác, tính ổn định trong các tình huống thực tế (câu hỏi đơn giản, phức tạp, ngoài phạm vi).

## Liên hệ

Nếu có thắc mắc, liên hệ nhóm qua tronganhsl93@gmail.com.