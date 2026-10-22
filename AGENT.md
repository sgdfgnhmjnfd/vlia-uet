# Ghi chú cho Agent / Cộng tác viên

- Repo tuân theo cấu trúc module chuẩn Hugging Face LeRobot (xem `policies/smolvla/`).
- Mỗi giai đoạn trong `docs/ROADMAP.md` có 1 script tương ứng trong `scripts/` — không gộp logic nhiều giai đoạn vào 1 file.
- Checkpoint/dataset lớn KHÔNG commit vào git (xem `.gitignore`) — dùng `scripts/push_to_hub.py` để lưu trên Hugging Face Hub.
- Before modifying `policies/smolvla/modeling_smolvla.py` (Phase 4), review the Prefix Embedding architecture in `docs/ROADMAP.md` to preserve the dynamic prefix structure (`L_prefix -> L_prefix + 1` intention token).
- Before resuming training, always run `scripts/tools/verify_checkpoint.py` to validate checkpoint paths and mounts.

Cấu trúc repo chuẩn (y như Hugging Face Transformers/LeRobot)
policy/                                             ← giữ nguyên policy base (như LeRobot)
  ├── scripts/                                       ← giữ nguyên script train/inference base
  └── policies/smolvla/                            ← phần bạn sẽ modify (cơ chế multi-step & intention)
      ├── configuration_smolvla.py
      ├── modeling_smolvla.py
      ├── tokenization_smolvla.py
      └── ...

docs/
  ├── ROADMAP.md                                     ← giữ nguyên — đây là "bản thiết kế" của bạn
  └── multi_step_env.md                              ← document về multi-step environment

datasets/
scripts/
  ├── push_to_hub.py
  ├── tools/verify_checkpoint.py                     ← quan trọng: trước khi train hãy chạy script này
  └── run_exp_multi_step.sh

Lưu ý quan trọng:

🧠 Phase 4 (Prefix Embedding) đã có trong modeling_smolvla.py

Các file核心 bạn cần tập trung là:
docs/ROADMAP.md — xem cấu trúc multi-step & intention
policies/smolvla/modeling_smolvla.py — xem cách intention token nối vào prefix
scripts/tools/verify_checkpoint.py — chạy trước khi resume training

💡 Bạn muốn tôi hỗ trợ thêm phần nào?

Kiểm tra lại cấu trúc Prefix Embedding
Chạy verify_checkpoint.py
Thay đổi model architecture cho multi-step?
Hay bắt đầu với script train/inference cho multi-step?
