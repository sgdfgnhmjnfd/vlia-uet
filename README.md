# vlia-uet — VLIA (Vision-Language-Intention-Action) trên SmolVLA & LIBERO

Nghiên cứu bổ sung một **module Ý định (Intention)** vào pipeline SmolVLA, hướng tới hợp tác người-robot nhận biết ý định (intention-aware human-robot collaboration). Backbone SmolVLA (chuẩn Hugging Face LeRobot), benchmark **LIBERO** (spatial / object / goal / 10), robot mục tiêu triển khai thật: **UR5**.

---

## 💻 1. Môi trường Hoạt động (Environment Setup)

Kích hoạt môi trường trước khi chạy bất kỳ script nào:

```bash
conda activate vlia_uet
# hoặc: source .venv/bin/activate  (nếu dùng venv thay vì conda)
```

Cài đặt lần đầu:

```bash
conda env create -f environment.yml
conda activate vlia_uet
pip install -r requirements.txt
```

---

## 📂 2. Cấu trúc Dự án (Repository Architecture)

```
vlia-uet/
│
├── ⚙️ config/                     # Cấu hình Tasks & Suites
│   ├── tasks/                    # phase1_baseline / phase3_text_intention / phase4_intention_token
│   └── suites/                   # libero_suites.yaml (spatial, object, goal, 10)
│
├── 🌐 envs/                       # Môi trường & tiện ích dữ liệu
│   ├── libero_env.py             # Wrapper Gym-style quanh benchmark LIBERO
│   ├── intention_labeler.py      # Gọi VLM ngoài (Qwen2.5-VL-3B) sinh nhãn What/Why/Next
│   └── scene_config.py           # Parser nạp TaskConfig từ YAML
│
├── 🧠 policies/                   # Kiến trúc Model AI
│   └── smolvla/                  # SmolVLA Policy mở rộng cho VLIA
│       ├── configuration_smolvla.py  # Dataclass cấu hình (+ use_intention_token, intention_token_dim)
│       ├── modeling_smolvla.py       # Chèn Intention Token (960D) vào Prefix Embedding
│       └── processor_smolvla.py      # Prompt template <think>-><intention>-><action>
│
├── 🦾 robots/                     # Trừu tượng hoá phần cứng
│   └── ur5_robot.py              # Driver UR5 cho Giai đoạn 5 (triển khai thật)
│
├── 📦 datasets/                   # Dataset LIBERO + nhãn ý định (tải riêng, không commit)
│
├── 🎯 scripts/                    # Entrypoints CLI theo từng giai đoạn
│   ├── train_baseline.py             # [1] Baseline nhẹ (Layer Skipping, Visual Tokens)
│   ├── generate_intention_labels.py  # [2] Sinh nhãn ý định bằng Qwen2.5-VL-3B
│   ├── train_text_intention.py       # [3] Text-based Intention baseline
│   ├── train_intention_token.py      # [4] Intention Token trong Prefix Embedding
│   ├── eval_policy.py                # [5] Đánh giá & so sánh 3 hệ thống
│   ├── push_to_hub.py                # Đẩy checkpoint/dataset lên Hugging Face Hub
│   └── 🛠️ tools/
│       ├── inspect_scheduler.py      # Debug LR scheduler khi resume checkpoint
│       └── verify_checkpoint.py      # Kiểm tra checkpoint hợp lệ trước khi eval/resume
│
├── urdf/ & meshes/                # (Dự phòng) file 3D nếu cần mô phỏng UR5 riêng ngoài LIBERO
│
├── docs/
│   └── ROADMAP.md                 # Bảng lộ trình 5 giai đoạn + ghi chú kỹ thuật
│
├── .gitignore
├── AGENT.md
├── LICENSE
├── README.md
├── environment.yml
└── requirements.txt
```

---

## 🚀 3. Hướng dẫn Toàn bộ Các Lệnh Chạy (Full Workflow CLI)

---

### 📦 GIAI ĐOẠN 1 (Tuần 1): Baseline nhẹ (`train_baseline.py`)

> Layer Skipping 16 lớp + 64 Visual Tokens, train trên LIBERO spatial. Mục tiêu: chạy mượt, success rate ổn định.

```bash
python scripts/train_baseline.py --config config/tasks/phase1_baseline.yaml
```

---

### 🏷️ GIAI ĐOẠN 2 (Tuần 2): Sinh dữ liệu Ý định (`generate_intention_labels.py`)

> Dùng Qwen2.5-VL-3B quét video rollout LIBERO, sinh nhãn **What / Why / Next**.

```bash
python scripts/generate_intention_labels.py \
    --video-dir datasets/libero_rollouts \
    --out datasets/intention_labels/phase2_labels.jsonl
```

---

### 🧠 GIAI ĐOẠN 3 (Tuần 3): Text Baseline (`train_text_intention.py`)

> Train SmolVLA với cấu trúc prompt `<think> -> <intention> -> <action>` trên nhãn Giai đoạn 2.

```bash
python scripts/train_text_intention.py --config config/tasks/phase3_text_intention.yaml
```

---

### 🔧 GIAI ĐOẠN 4 (Tuần 4-5): Intention Token (`train_intention_token.py`)

> Chèn 1 token ý định (960D) vào Prefix Embedding của `modeling_smolvla.py`, huấn luyện căn chỉnh với không gian nhãn Giai đoạn 3.

```bash
python scripts/train_intention_token.py --config config/tasks/phase4_intention_token.yaml
```

---

### 📊 GIAI ĐOẠN 5 (Tuần 6): Đánh giá & Triển khai (`eval_policy.py`)

> Rollout kiểm tra, so sánh Success Rate: **SmolVLA gốc** vs **Text-based VLIA** vs **Token-based VLIA**.

```bash
# Đánh giá 1 checkpoint trên 1 suite
python scripts/eval_policy.py \
    --policy.path outputs/phase4_intention_token/checkpoints/last/pretrained_model \
    --env.task libero_10 \
    --eval.n_episodes 10

# Đánh giá đầy đủ 4 suite, gộp bảng so sánh 3 hệ thống
python scripts/eval_policy.py \
    --suites-config config/suites/libero_suites.yaml \
    --compare baseline,text_intention,intention_token
```

#### 📋 Chỉ số đánh giá

| Chỉ số | Ý nghĩa |
| --- | --- |
| **Success Rate** | Tỷ lệ episode hoàn thành nhiệm vụ thành công (%) |
| **Avg Sum Reward** | Tổng reward trung bình mỗi episode |
| **Avg Max Reward** | Reward cao nhất trung bình mỗi episode |

---

### 🛠️ Công cụ phụ trợ (`scripts/tools/`)

```bash
# Xem scheduler_state.json / training_step.json / optimizer_param_groups.json của 1 checkpoint
python scripts/tools/inspect_scheduler.py --checkpoint-dir outputs/smolvla_libero/checkpoints/025000

# Kiểm tra checkpoint tồn tại & hợp lệ trước khi eval/resume
python scripts/tools/verify_checkpoint.py --path outputs/smolvla_libero/checkpoints/050000
```

---

## 🤝 4. Đẩy Checkpoint/Dataset lên Hugging Face Hub (`push_to_hub.py`)

```bash
python scripts/push_to_hub.py \
    --checkpoint-dir outputs/phase4_intention_token/checkpoints/last/pretrained_model \
    --repo-id username/vlia-smolvla-libero \
    --private
```

---

## 📈 Trạng thái hiện tại

- [x] Fine-tune SmolVLA baseline trên LIBERO (10k → đang tiếp tục tới 100k step)
- [ ] Giai đoạn 1: Baseline nhẹ (Layer Skipping 16 lớp, 64 Visual Tokens)
- [ ] Giai đoạn 2: Sinh nhãn ý định bằng Qwen2.5-VL-3B
- [ ] Giai đoạn 3: Text-based Intention baseline
- [ ] Giai đoạn 4: Intention Token (960D) trong Prefix Embedding
- [ ] Giai đoạn 5: Đánh giá & so sánh

Chi tiết đầy đủ: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## About

Intention-aware SmolVLA (Vision-Language-Intention-Action) cho manipulation trên benchmark LIBERO, hướng triển khai UR5.
