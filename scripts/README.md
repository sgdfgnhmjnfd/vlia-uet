# scripts/

**Tệp chính** (dùng trực tiếp):

| Tệp | Việc |
|---|---|
| `train_intention_only_LIBERO7D.py` | huấn luyện VLIA: SmolVLA + token ý định, câu lệnh để trống |
| `train_matched_oracle_LIBERO7D_FIXED.py` | huấn luyện SmolVLA gốc (có câu lệnh) để so sánh, cùng cấu hình |
| `libero_goal_ablation.py` | chạy robot trên LIBERO-10 với mọi điều kiện (câu lệnh, token đúng/sai, token từ mô-đun ý định) |
| `run_stage15_supervised.py` | mô-đun C: phân loại ý định sớm có giám sát |
| `run_stage12_e2e.py` | mô-đun F: ghép câu bằng SigLIP, hỏi lại, vòng kín |
| `run_stage14_generate.py` | mô-đun G: Qwen2.5-VL tự viết câu ý định |
| `train_app_classifier.py` | huấn luyện bộ phân loại C cuối cùng cho app |
| `test_*.py` | kiểm thử (`test_stage12.py` chạy không cần GPU hay dữ liệu) |

**Tệp hỗ trợ** (các tệp chính nhập hàm từ đây; giữ nguyên để mã chạy đúng như lúc làm thí nghiệm):
`extract_ladder_embeddings.py` (trích đặc trưng SigLIP), `hoi_utils.py` (đọc khung hình), `run_stage13_human.py` (đọc video), `token_matrix.py`, `v3_stage3_mllm.py`, `v3_common.py`, `run_ladder_v1.py`, `run_scene_cv.py`, `run_prefix_length_ablation.py`, `run_zero_shot_alignment.py`, `train_temporal_progression_intention.py` (giai đoạn đầu trên EgoIntent).
