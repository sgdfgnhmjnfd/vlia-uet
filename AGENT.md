# Ghi chú cho Agent / Cộng tác viên

- Repo tuân theo cấu trúc module chuẩn Hugging Face LeRobot (xem `policies/smolvla/`).
- Mỗi giai đoạn trong `docs/ROADMAP.md` có 1 script tương ứng trong `scripts/` — không gộp logic nhiều giai đoạn vào 1 file.
- Checkpoint/dataset lớn KHÔNG commit vào git (xem `.gitignore`) — dùng `scripts/push_to_hub.py` để lưu trên Hugging Face Hub.
- Trước khi sửa `policies/smolvla/modeling_smolvla.py` (Giai đoạn 4), đọc kỹ phần kiến trúc Prefix Embedding trong `docs/ROADMAP.md` để không phá vỡ shape (97x960 -> 98x960).
- Khi resume training, luôn chạy `scripts/tools/verify_checkpoint.py` trước để tránh lỗi path/mount (bài học từ vụ ổ đĩa đổi mount point `/media/.../d` -> `/media/.../d1`).
