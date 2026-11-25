# Stage 20 STATE

- 2026-11-25 18:15 | Stage 20 data audit **xong** (chỉ đọc, ~3 phút, CPU). Không train, không inference, không chạy lại accuracy, không sửa nhãn/split, không commit/push. Khối B (temporal difference) không chạy.
- Xác minh: 27 video, 2.503 → 1.001 bước, 10 lớp, 8 fold theo cảnh; khớp labels_ego.csv và ego_oof.npz; không trùng video/cảnh giữa train và validation.
- Phát hiện chính: vắng trong train 0/80; vắng trong validation 10/80 (fold 6 chỉ 5 lớp); align 84% workshop (train còn 8 bước ở fold 0), wipe 63% ở một video bedroom, position 54% workshop; 55/1.001 câu có ≥ 2 động từ thuộc lớp; README: clip cắt trước khi kết quả hiện rõ, nhãn = ý định tức thời.
- Chưa hoàn thành / không làm được: thống kê theo người quay (không có dữ liệu); kiểm bằng mắt động tác trong clip (ngoài phạm vi).
