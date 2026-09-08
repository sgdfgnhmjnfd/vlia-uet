# Ghi chú cho Agent / Cộng tác viên

- Repo tuân theo cấu trúc module chuẩn Hugging Face LeRobot (xem `policies/smolvla/`).
- Mỗi giai đoạn trong `docs/ROADMAP.md` có 1 script tương ứng trong `scripts/` — không gộp logic nhiều giai đoạn vào 1 file.
- Checkpoint/dataset lớn KHÔNG commit vào git (xem `.gitignore`) — dùng `scripts/push_to_hub.py` để lưu trên Hugging Face Hub.
- Before modifying `policies/smolvla/modeling_smolvla.py` (Phase 4), review the Prefix Embedding architecture in `docs/ROADMAP.md` to preserve the dynamic prefix structure (`L_prefix -> L_prefix + 1` intention token).
- Before resuming training, always run `scripts/tools/verify_checkpoint.py` to validate checkpoint paths and mounts.
