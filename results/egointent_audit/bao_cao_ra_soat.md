# Rà soát dữ liệu EgoIntent (25/11/2026)

Chỉ đọc + đếm, CPU, ~30 s chạy. Không train, không sửa nhãn/fold. Dữ liệu: 1.001 bước, 10 động từ, 27 video phát triển, 13 cảnh, 8 fold (khớp `labels_ego.csv`, lớp lớn nhất 25,5%). Dự đoán lấy từ file đã lưu (Stage 18 `ego_oof.npz`, Stage 19 `eval_ego.json`); độ chính xác cân bằng tính lại khớp số đã báo (S18 S 11,2 / M p50 9,9 / S+M p50 10,4 / S+M p100 12,5; S19 9,8 / 11,9 / 11,9 / 17,8).

**Nhóm cảnh** = trường `scene` trong manifest (13 cảnh: kitchen 5 video, workshop 5, garage 3, bedroom/farm/laundry/living_room 2, còn lại 1). Mỗi video thuộc đúng 1 cảnh. Fold gom cảnh nguyên khối (fold 0 = workshop, 1 = kitchen, 2 = garage, 3 = farm, 4–7 ghép 2–3 cảnh nhỏ). Không có mã người.

## Tóm tắt

| | Số chứng cứ chính | Đánh giá | Giải thích |
|---|---|---|---|
| H-a lớp thiếu khi train | 0/70 cặp (fold, lớp) có lớp test với < 3 video train; 0 cặp train = 0; 20/70 cặp có < 5 bước test | **không** | Không fold nào thiếu hẳn lớp. Ca yếu nhất: fold 0 (workshop) giữ ra 41/49 bước *align*, train chỉ còn 8 bước align. Lệch phân bố train↔test TV trung bình 0,28 (max 0,45). |
| H-b động từ gắn với cảnh | 3/10 lớp > 50% bước ở 1 cảnh: align 84% (workshop), wipe 63% (bedroom, cũng 63% ở 1 video), position 54% (workshop) | **có** | Đúng 3 lớp này có recall ≈ 0 ở S18 S+M p50 (wipe 0, align 0, position 1%). |
| H-c nhãn mơ hồ | 5,5% câu có ≥ 2 động từ thuộc 10 lớp; 7,6% có and/then/phẩy; lỗi giữa cặp gần nghĩa 20% (S18 S+M p50), 17% (S19) | **không** | Câu WHAT ngắn, phần lớn 1 động từ. Gần nghĩa có thật (xem mẫu dưới) nhưng không chiếm phần lớn lỗi. |
| H-d cửa sổ quá ngắn | trung vị bước 1,77 s; trung vị cửa sổ p25 0,23 s, **p50 0,73 s**, p100 1,77 s; 605/1001 bước có p50 < 1 s, 300 < 0,5 s; khoảng giữa 2 khung cache 0,23 s | **có** (theo ngưỡng) | Nhưng không giải thích được may rủi: độ dài theo lớp gần như như nhau (trung vị 1,4–2,3 s), Spearman độ dài–recall ρ = 0,40 (p 0,25) S18, 0,10 (p 0,78) S19; và Stage 18b cùng cửa sổ, chia trong video, đạt 37,9. |
| H-e đặc trưng gom theo cảnh | SigLIP S: láng giềng cùng video **98,6%** (ngẫu nhiên 4,3% → 23×); cùng động từ khi chỉ tìm ở cảnh khác 13,3% (ngẫu nhiên 13,5% → **0,99×**). S+M p50: 91% / 0,88×. VideoMAE clip p50: 58% (13×) / 1,15× | **có** (rất mạnh) | Không gian đặc trưng xếp theo video/cảnh; sang cảnh khác thì láng giềng gần nhất không mang động từ. |
| H-f dự đoán bị dồn | S18 S+M p50: 2/10 lớp recall 0 (wipe, align); place + pick_up chiếm 49% dự đoán (S19: 37%) | **không** (theo ngưỡng) | Có dồn một phần về 2 lớp lớn nhất, nhưng dưới ngưỡng 60%. |

## 5 câu WHAT mỗi lớp (seed 0), đặt các nhóm gần nghĩa cạnh nhau

Nhóm place / position / align:
- **place**: “place the bolt on the lift arm” (4.4s) · “placing screwdriver on lift arm” (5.8s) · “placing red cloth into washing machine” (0.9s) · “placing scissors on the table” (1.0s) · “placing a glass on the drainboard” (4.8s)
- **position**: “positioning the green basin” (0.7s) · “positioning a thin wood slat vertically” (1.4s) · “position the nail gun” (0.3s) · “positioning wood piece” (2.1s) · “positioning polythene cover with left hand and aiming staple gun with right” (7.1s)
- **align**: “aligning the towel edges” (1.8s) · “aligning a small wood block against another block” (0.4s) · “aligning a spacer block against the wood slat” (0.7s) · “aligning the spirit level on the plywood” (5.6s) · “aligning level on table” (8.6s)

Nhóm grasp / pick_up / hold:
- **grasp**: “grasp green bowl” (0.5s) · “grasp loose wood strip” (0.4s) · “grasps a retractable knife on the rack with the left hand” (2.4s) · “grasp the rake with both hands” (0.9s) · “grasping handle of yellow bucket” (5.2s)
- **pick_up**: “pick up nail gun” (1.3s) · “pick up a pink eraser from the table with the left hand” (1.4s) · “pick up tape strip from cutting mat” (3.5s) · “picking up wine glasses” (3.7s) · “picking up a stone from the border with the right hand” (4.4s)
- **hold**: “holding a squeegee against the window frame” (0.0s) · “hold vent cover open” (6.2s) · “holding a white plate in front of the open dishwasher” (4.7s) · “hold the top dishwasher rack” (1.7s) · “holding a measuring spoon with detergent over the open drawer” (3.2s)

adjust (so với align ở trên):
- **adjust**: “adjusting vise grips on suspension component” (3.2s) · “adjusts engine component with pliers” (2.0s) · “adjust head-mounted camera while holding spirit level” (3.8s) · “adjusting the black plastic liner into the corner of the wooden bed” (0.4s) · “adjusting the black plastic sheet along the wooden frame edge” (0.5s)

Còn lại:
- **wipe**: “wiping the window glass with a cloth” (1.9s) · “wipe window glass with towel” (4.7s) · “wipe face with right arm” (1.6s) · “wipe vent grille slats with right hand” (1.9s) · “wipe right hand on trousers” (4.3s)
- **remove**: “remove socks from drying rack” (1.2s) · “remove vise grips from suspension linkage” (2.8s) · “removes food processor lid” (1.2s) · “remove debris from engine” (2.9s) · “removing debris from engine with pliers” (1.9s)
- **walk**: “walk to wheel hub” (1.4s) · “walk towards tool chest” (1.5s) · “walking towards the cat” (7.4s) · “walking through the hallway” (5.6s) · “walking into the laundry room” (6.3s)

## Recall từng lớp (%), từ dự đoán đã lưu

| lớp | S18 S|0 | S18 M|50 | S18 S+M|50 | S19 S|0 | S19 M|50 | S19 S+M|50 | S19 S+M|100 |
|---|---|---|---|---|---|---|---|
| place | 33 | 18 | 26 | 16 | 24 | 21 | 26 |
| pick_up | 25 | 14 | 26 | 14 | 22 | 25 | 24 |
| grasp | 9 | 11 | 7 | 18 | 15 | 17 | 23 |
| wipe | 10 | 4 | 0 | 1 | 3 | 3 | 4 |
| hold | 15 | 36 | 16 | 5 | 30 | 23 | 27 |
| position | 0 | 3 | 1 | 0 | 3 | 7 | 6 |
| adjust | 6 | 3 | 12 | 12 | 7 | 15 | 13 |
| remove | 2 | 2 | 4 | 14 | 10 | 0 | 6 |
| align | 2 | 6 | 0 | 4 | 0 | 0 | 12 |
| walk | 11 | 2 | 11 | 13 | 4 | 6 | 36 |

Cặp nhầm nhiều nhất S18 S+M p50: place→pick_up 58, wipe→place 52, grasp→pick_up 36. S19 S+M p50: place→pick_up 39, place→adjust 30, wipe→place 29.

## File

`fold_lop.csv`, `fold_tom_tat.csv`, `canh_lop.csv`, `video_lop.csv`, `lop_tap_trung.csv`, `mau_cau_what.csv`, `recall_lop.csv`, `phan_bo_du_doan.csv`, `cap_nham_nhieu.csv`, `do_dai_lop.csv`, `lang_gieng.csv`, `ra_soat.json`. Script: `../scripts/ra_soat_ego.py`.
Ghi chú cách tính: cửa sổ p theo đúng chỉ số khung cache `linspace(0, n−1, 8)` (khung 1 = đầu bước, 0 s); láng giềng dùng cosine, S+M ghép sau khi L2 từng khối (không có nhãn trong bước này).
