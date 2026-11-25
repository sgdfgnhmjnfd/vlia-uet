# Stage 20 audit manifest

Thời gian: 25/11/2026 18:12–18:15 (+07), khoảng 3 phút; script chạy < 1 s. CPU, `nice -n 19`, `CUDA_VISIBLE_DEVICES=""`, 2 luồng. Không có job hay tmux nào đang chạy lúc bắt đầu (`tmux ls`: không có server). Worktree `/media/dhqg/d3/vlia-ego` sạch, HEAD `4942b23` (claude/stage19-temporal); không đổi nhánh, không commit.

## File nguồn đã đọc (chỉ đọc)
- `scripts/stage18_scene_motion.py` (import: `ego_data`, `verb`, `lemma`, `KEEP`, `load_manifest`, `locked_uids`)
- `scripts/run_scene_cv.py` (`assign_folds`)
- `results/hoi_detection_v1/manifest.jsonl` (2.503 bước phát triển)
- `results/test_protocol_full_v1/locked_protocol.json` (chỉ danh sách video test khóa, dùng để assert loại trừ)
- `results/stage18_scene_motion_ego/labels_ego.csv`, `label_rule.md`, `config.json`
- `/media/dhqg/d3/vlia_outputs/stage18/ego_oof.npz` (chỉ `y`, `outer` để kiểm khớp thứ tự/fold; không đọc lại accuracy)
- `/media/dhqg/d3/vlia_outputs/stage18/durations.json` (qua `ego_data`)
- `/media/dhqg/d3/vlia_outputs/ladder/siglip-so400m-patch14-384/sample_ids.json`, `frame_emb.npy` (mmap qua `ego_data`, chỉ để kiểm thứ tự; không tính gì trên đặc trưng)
- `/media/dhqg/d3/datasets/egointent_full/README.md`, `metadata.jsonl` (tìm trường người quay và mô tả cách cắt clip)

## Lệnh đã chạy
- `git -C /media/dhqg/d3/vlia-ego status --short`, `log --oneline -1`; `tmux ls`; `date`
- `head`/`grep` trên manifest, `metadata.jsonl`, README (tên trường, participant)
- `nice -n 19 /home/dhqg/anaconda3/envs/lerobot/bin/python /media/dhqg/d3/vlia_outputs/stage20_audit/scripts/stage20_audit.py`
  (script ngoài git; số liệu thô: `/media/dhqg/d3/vlia_outputs/stage20_audit/stage20_numbers.json`; script từ chối ghi đè file đã có)

## Output
`data_audit_report.md`, `class_fold_distribution.csv` (80 dòng fold × lớp), `scene_video_distribution.csv` (lớp × cảnh × video có ≥ 1 bước), `audit_manifest.md`, `STATE.md`.

## Không xác minh được
- Danh tính / mã người quay: không có trong dữ liệu.
- Động tác có thực sự nhìn thấy được trong clip hay không: không xem video.
- Split không lưu thành file: đã xác minh bằng code + khớp `outer` trong `ego_oof.npz`.
