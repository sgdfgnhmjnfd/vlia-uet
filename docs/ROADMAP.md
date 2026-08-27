# Roadmap — 5 giai đoạn

| Giai đoạn | Mục tiêu | Hành động cụ thể |
|---|---|---|
| **Giai đoạn 1** (Tuần 1) | Thiết lập Baseline nhẹ | Áp dụng cấu hình tinh gọn (Layer Skipping 16 lớp, 64 Visual Tokens) để train SmolVLA trên LIBERO spatial. Đảm bảo chạy mượt và đạt success rate ổn định. |
| **Giai đoạn 2** (Tuần 2) | Tạo dữ liệu Ý định | Dùng một VLM ngoài (Qwen2.5-VL-3B) quét qua các video LIBERO để sinh nhãn ý định văn bản (What, Why, Next). |
| **Giai đoạn 3** (Tuần 3) | Huấn luyện Text Baseline | Huấn luyện SmolVLA với cấu trúc prompt `<think> -> <intention> -> <action>` trên tập dữ liệu đã gán nhãn ở Giai đoạn 2. |
| **Giai đoạn 4** (Tuần 4-5) | Triển khai Intention Token | Viết lại module Prefix Embedding trong `modeling_smolvla.py` để chèn thêm 1 token ý định (960D). Huấn luyện module này căn chỉnh với không gian nhãn ý định của Giai đoạn 3. |
| **Giai đoạn 5** (Tuần 6) | Đánh giá & Triển khai | Rollout kiểm tra trên PyBullet, so sánh Success Rate giữa: SmolVLA gốc vs. Text-based VLIA vs. Token-based VLIA. |

## Ghi chú kỹ thuật theo từng giai đoạn

### Giai đoạn 1 — Baseline nhẹ
- Layer Skipping: giảm còn 16 lớp SmolVLM (so với 16 lớp gốc — xác nhận lại số lớp thực tế của SmolVLM2-500M-Video-Instruct trước khi cắt).
- Visual Tokens: giới hạn 64 token thị giác (so với cấu hình mặc định) để giảm tải tính toán.
- Mục tiêu: chạy ổn định trên LIBERO spatial trước khi mở rộng sang các suite khác, làm nền so sánh (baseline) cho các giai đoạn sau.

### Giai đoạn 2 — Sinh nhãn ý định
- Input: video rollout LIBERO (từ checkpoint baseline Giai đoạn 1 hoặc dataset gốc).
- Output nhãn theo 3 trường: **What** (đang làm gì), **Why** (vì sao/mục tiêu), **Next** (bước tiếp theo dự kiến).
- Model gán nhãn: Qwen2.5-VL-3B chạy offline, xuất ra file nhãn (JSON/JSONL) gắn với từng frame/segment.

### Giai đoạn 3 — Text-based Intention baseline
- Cấu trúc prompt: `<think> ... </think> <intention> ... </intention> <action> ... </action>`.
- Train SmolVLA trên tập dữ liệu đã gán nhãn Giai đoạn 2, coi ý định là chuỗi văn bản chèn vào prompt (chưa sửa kiến trúc model).

### Giai đoạn 4 — Intention Token
- Sửa `modeling_smolvla.py`: chèn 1 token ý định 960D vào **Prefix Embedding** (không gian 960D, theo kiến trúc SmolVLA: Image 16×960 + Language 48×960 + State 1×960 → Prefix 97×960).
- Sau khi chèn: Prefix Embedding mở rộng thành 98×960.
- Token ý định cần được huấn luyện để căn chỉnh (align) với không gian nhãn ý định đã học ở Giai đoạn 3 (có thể dùng contrastive loss hoặc regression loss tuỳ thiết kế).

### Giai đoạn 5 — Đánh giá
- Rollout trên PyBullet (môi trường LIBERO).
- So sánh 3 hệ thống: SmolVLA gốc (không ý định) / Text-based VLIA (Giai đoạn 3) / Token-based VLIA (Giai đoạn 4).
- Chỉ số: Success Rate, Avg Sum Reward, Avg Max Reward (theo format đã dùng ở báo cáo 10k vs 50k trước đó).
