# Stage 20: Data audit EgoIntent (chỉ đọc)

Ngày 25/11/2026, CPU, không train/inference, không chạy lại accuracy. Kết quả Stage 16–19 giữ nguyên. Đây là audit mô tả, **không chứng minh quan hệ nhân quả**.

## 1. Tổng quan dữ liệu và kiểm tra khớp

- Nguồn: `results/hoi_detection_v1/manifest.jsonl` (2.503 bước phát triển, 27 video, 13 cảnh; `split` chỉ có train/val; không có video test khóa, đã assert theo `test_protocol_full_v1/locked_protocol.json`). Câu WHAT = trường `what` (= `local_intent` của EgoIntent).
- Nhãn: luật `label_rule.md`, hàm `verb()` trong `scripts/stage18_scene_motion.py` (import, không chép lại) → 1.001 bước, 10 lớp; số bước từng lớp **khớp** `labels_ego.csv`; lớp lớn nhất 25,5%.
- Thứ tự bước và fold **khớp** file OOF đã lưu của Stage 18 (`vlia_outputs/stage18/ego_oof.npz`: `y`, `outer` trùng 100%).
- Mỗi video thuộc đúng 1 cảnh (kiểm trên dữ liệu).
- **Không có trường người quay** trong manifest, `metadata.jsonl` hay README → không thống kê được theo người (xem mục 4).

## 2. Cách chia fold (xác minh từ code, không tạo lại)

Split không lưu thành file; được tạo động trong `S18.ego_data()`: `run_scene_cv.assign_folds(scenes, 8)` gọi trên **toàn bộ 2.503 bước** (trước khi lọc 10 lớp), xếp cảnh theo số bước giảm dần (hoà thì theo tên), gán tham lam vào fold đang ít bước nhất. Không có seed (tất định). Fold k = validation, 7 fold còn lại = train. Do cân bằng trên 2.503 bước nên số bước validation sau khi lọc lệch nhau (57–216).

| Fold | Cảnh validation | Train videos | Val videos | Train steps | Val steps | Trùng video | Trùng cảnh |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | workshop | 22 | 5 | 785 | 216 | 0 | 0 |
| 1 | kitchen | 22 | 5 | 893 | 108 | 0 | 0 |
| 2 | garage | 24 | 3 | 858 | 143 | 0 | 0 |
| 3 | farm | 25 | 2 | 904 | 97 | 0 | 0 |
| 4 | hallway|living_room | 24 | 3 | 874 | 127 | 0 | 0 |
| 5 | bedroom|yard | 24 | 3 | 851 | 150 | 0 | 0 |
| 6 | clothing_closet|space | 25 | 2 | 944 | 57 | 0 | 0 |
| 7 | art_studio|deck|laundry_room | 23 | 4 | 898 | 103 | 0 | 0 |

Kiểm tra: không video nào và không cảnh nào nằm ở cả train và validation trong bất kỳ fold nào; tổng bước validation = 1.001; tổng video = 27. Chú ý: 5/8 fold có validation chỉ 2–3 video.

## 3. Phân bố lớp theo fold (`class_fold_distribution.csv`)

| Class | Total steps | Total videos | Folds missing in train | Folds missing in val (fold) | Min train steps / videos | Max % lớp nằm ở 1 fold val |
|---|---:|---:|---:|---|---|---:|
| place | 255 | 27 | 0 | 0 (–) | 211 / 22 | 17% (fold 4) |
| pick_up | 185 | 26 | 0 | 0 (–) | 150 / 21 | 19% (fold 0) |
| grasp | 115 | 25 | 0 | 0 (–) | 89 / 20 | 23% (fold 0) |
| wipe | 91 | 11 | 0 | 2 (3|6) | 34 / 8 | 63% (fold 5) |
| hold | 73 | 23 | 0 | 0 (–) | 52 / 19 | 29% (fold 0) |
| position | 69 | 16 | 0 | 1 (6) | 32 / 11 | 54% (fold 0) |
| adjust | 67 | 18 | 0 | 0 (–) | 48 / 14 | 28% (fold 2) |
| remove | 50 | 14 | 0 | 1 (6) | 35 / 11 | 30% (fold 0) |
| align | 49 | 9 | 0 | 3 (1|4|6) | 8 / 5 | 84% (fold 0) |
| walk | 47 | 10 | 0 | 3 (0|5|6) | 26 / 7 | 45% (fold 2) |

- **Vắng trong train: 0/80 trường hợp.** Mọi lớp có ở train của mọi fold, tối thiểu 5 video train.
- **Vắng trong validation: 10/80 trường hợp** (wipe 2, position 1, remove 1, align 3, walk 3); fold 6 (clothing_closet + space, 57 bước, 2 video) chỉ có 5/10 lớp. Ở những fold này recall của lớp đó không đo được; độ chính xác cân bằng của Stage 18 tính trên toàn bộ dự đoán gộp 8 fold nên vẫn xác định.
- **Có mẫu nhưng train mỏng:** fold 0 (workshop) giữ ra 41/49 bước *align* → train còn **8 bước** align (5 video); *position*: train 32 bước khi giữ workshop; *wipe*: train 34 bước khi giữ bedroom/yard (fold 5).
- **Lớp chỉ có ở 1–2 video trong train:** không có trường hợp nào (tối thiểu 5 video).

Mất cân bằng từng fold:

| Fold | Lớp lớn nhất train | Tỉ lệ max/min train | Lớp có mặt ở val | Lớp lớn nhất val | Total variation train↔val |
|---|---:|---:|---:|---|---:|
| 0 | 28.9% | 28.4 | 9/10 | align 19% | 0.37 |
| 1 | 24.6% | 5.5 | 9/10 | place 32% | 0.21 |
| 2 | 26.1% | 8.6 | 10/10 | place 22% | 0.22 |
| 3 | 25.3% | 5.5 | 9/10 | place 27% | 0.24 |
| 4 | 24.1% | 4.8 | 9/10 | place 35% | 0.25 |
| 5 | 25.5% | 6.4 | 9/10 | wipe 38% | 0.36 |
| 6 | 24.0% | 4.8 | 5/10 | place 49% | 0.45 |
| 7 | 25.6% | 7.0 | 10/10 | place 24% | 0.18 |

Train: lớp lớn nhất (place) luôn 24–29%; fold 0 lệch nhất (max/min 28× vì align gần như vắng). Validation lệch nhiều hơn: TV train↔val 0,18–0,45.

## 4. Phân bố theo cảnh và video (`scene_video_distribution.csv`)

| Class | Number of scenes | Number of videos | Largest-video step share | Cảnh lớn nhất (% bước) | Số video chứa ≥ 50% bước | Observation |
|---|---:|---:|---:|---|---:|---|
| place | 13 | 27 | 9% | kitchen 14% | 8 | trải đều |
| pick_up | 13 | 26 | 10% | workshop 19% | 7 | trải đều |
| grasp | 11 | 25 | 15% | workshop 23% | 6 | trải đều |
| wipe | 7 | 11 | 63% | bedroom 63% | 1 | **1 video (bedroom) chiếm 63%**; 1 video đủ một nửa số bước |
| hold | 12 | 23 | 10% | workshop 29% | 6 | trải đều |
| position | 9 | 16 | 22% | workshop 54% | 4 | **54% ở workshop** |
| adjust | 9 | 18 | 16% | garage 28% | 4 | tương đối trải |
| remove | 8 | 14 | 20% | workshop 30% | 3 | 30% ở workshop; 3 video đủ một nửa |
| align | 6 | 9 | 39% | workshop 84% | 2 | **84% ở workshop**; 2 video đủ một nửa |
| walk | 7 | 10 | 23% | garage 45% | 3 | 45% ở garage |

- 4 lớp lớn (place, pick_up, grasp, hold) có mặt ở 12–13 cảnh và 23–27 video; không video nào > 15%.
- 3 lớp **gắn với một cảnh**: align 84% workshop, wipe 63% bedroom (và đó là **một video**), position 54% workshop. Workshop có 5 video nên align/position dựa vào một cảnh nhưng vài video; wipe dựa vào **một video**: 57 trong 91 bước wipe là các bước cùng một video, không phải mẫu độc lập.
- Người quay: **không xác minh được** (không có mã người trong dữ liệu; video gốc từ Ego4D). Không suy đoán.

## 5. Kiểm tra nhãn

Luật nhãn = động từ đầu tiên của câu WHAT (bỏ chủ ngữ/trạng từ cố định, lemma theo biến tố, ghép tiểu từ up/down/...), không gộp theo nghĩa.

Đếm trên 1.001 câu (trung bình 5,7 từ/câu):
- Động từ thuộc 10 lớp xuất hiện đầu tiên trong câu luôn trùng nhãn (100%, đúng luật).
- **5,5% (55 câu)** có ≥ 2 động từ thuộc 10 lớp; động từ thứ hai hay gặp: position 18, hold 13, adjust 10. Ví dụ: “grasp the paper cutout while holding a tape strip”, “remove socket from tool and place on lift”. 8,4% câu có and/then/while/dấu phẩy.
- 46% câu bắt đầu bằng V-ing (“positioning…”, “grasping…”); luật xử lý cùng nhãn với dạng nguyên mẫu.

Các cặp ưu tiên:

| Cặp | Số bước | Cùng câu có cả 2 | Video chung | Từ đi sau động từ (a) | Từ đi sau động từ (b) |
|---|---|---:|---:|---|---|
| place/position | 255 / 69 | 2 | 16 | palm 21, cut 16, iron 13, bucket 13, glass 13 | wood 9, tape 8, engine 6, small 6, squeegee 5 |
| pick_up/grasp | 185 / 115 | 0 | 24 | tape 14, cloth 9, iron 9, laundry 8, glass 8 | lemon 14, paint 10, and 8, water 7, branch 7 |
| hold/grasp | 73 / 115 | 3 | 22 | tape 9, palm 7, measure 5, branch 5, paper 3 | lemon 14, paint 10, and 8, water 7, branch 7 |
| adjust/align | 67 / 49 | 0 | 6 | cloth 8, plastic 7, rear 6, black 6, position 5 | wood 13, level 11, spirit 7, tape 6, small 6 |
| remove/place | 50 / 255 | 1 | 14 | debris 10, food 6, processor 6, engine 6, drying 4 | palm 21, cut 16, iron 13, bucket 13, glass 13 |

Ví dụ (seed 0, 3 câu mỗi bên):
- place/position: a: “place plastic bag in trunk” · “placing wire connector on tray” · “place a glass on the rack and pick a glass from the sink”; b: “positioning blue tape strip” · “positioning blue tape strip horizontally across painting” · “positioning a squeegee on the window glass”
- pick_up/grasp: a: “pick up timber slats” · “pick up tape strip from cutting mat” · “picking up a stone from the border with the right hand”; b: “grasp loose wood strip” · “grasping engine wiring” · “grasping a palm branch from the heap”
- hold/grasp: a: “hold cigarette while walking” · “holding the spirit level steady on the plywood” · “holding tape measure extended”; b: “grasp iron handle” · “grasp long timber piece” · “grasping the yellow extension cord with the right hand”
- adjust/align: a: “adjust fabric and pick up iron” · “adjusting blue tool” · “adjusting soil near border with right hand”; b: “aligning the towel edges” · “aligning small timber blocks” · “aligning wooden slat assembly on board”
- remove/place: a: “remove small piece of tape” · “remove black garment from drying rack” · “remove vent grille”; b: “place the nut onto the mower bolt by hand” · “place white powder in center” · “place a white powder dot on a flower petal”

Nhận xét (mô tả, không sửa nhãn):
- Các cặp hiếm khi cùng xuất hiện trong một câu, nhưng phân biệt chủ yếu bằng **cách người chú thích chọn từ**: “grasp lemon” vs “pick up tape” vs “hold tape measure” đều là nắm/cầm một vật; “place” vs “position” (“positioning a squeegee on the window glass”) cùng là đặt vật. README EgoIntent cũng ghi: *“Labels … may reflect annotator wording preferences.”*
- Từ đi kèm mỗi nhãn khác nhau theo **đồ vật**, và đồ vật gắn với cảnh: align ↔ wood, spirit level, timber (workshop); wipe ↔ một video bedroom. Nghĩa là nhãn động từ và cảnh/đồ vật trùng nhau một phần.
- **Nhãn có phản ánh hành động trong clip không:** theo README, mỗi micro-step được cắt *“immediately before that outcome became visually explicit”*, và WHAT là `local_intent` (mục tiêu tức thời). Như vậy clip được thiết kế để **không** chứa phần kết quả của hành động (ví dụ vật đã nằm yên sau “place”). Audit không xem video nên không đo được tỉ lệ bước có thấy động tác.
- Không kết luận nhãn nào sai chỉ từ việc mô hình đoán sai.

## 6. Phát hiện, theo mức bằng chứng

**A. Được dữ liệu xác nhận**
1. Split Stage 18 đúng như mô tả: 27 video, 1.001 bước, 8 fold, không trùng video/cảnh giữa train và validation.
2. Không lớp nào vắng trong train ở bất kỳ fold nào (0/80).
3. 10/80 cặp (fold, lớp) vắng trong validation; fold 6 chỉ có 5 lớp, 57 bước, 2 video.
4. align (84% workshop), wipe (63% một video bedroom), position (54% workshop) tập trung ở một cảnh; khi cảnh đó bị giữ ra, train còn 8 / 34 / 32 bước của lớp.
5. 5/8 fold có validation 2–3 video; dữ liệu chỉ 13 cảnh, 27 video → mỗi fold kiểm trên rất ít bối cảnh độc lập.
6. Theo README: clip kết thúc trước khi kết quả hành động hiện rõ; nhãn là ý định tức thời do người chú thích viết tự do.

**B. Có dấu hiệu nhưng chưa đủ bằng chứng**
1. Các cặp place/position, pick_up/grasp/hold mô tả cử động gần giống nhau; ranh giới có thể phụ thuộc cách chọn từ của người chú thích.
2. Nhãn động từ trùng một phần với đồ vật và cảnh (align ↔ dụng cụ gỗ ở workshop), nên một bộ phân loại có thể học đồ vật/cảnh thay vì cử động.
3. Một số lớp có thể phụ thuộc vào người quay hoặc phong cách của một video (wipe), nhưng không có mã người để kiểm.
4. Vì clip dừng trước kết quả, một phần động từ (place, position, align) có thể ít thấy được trong khung hình.

**C. Không thể kết luận từ audit này**
1. Scene shift là nguyên nhân **duy nhất** khiến Stage 18 ra ngang may rủi.
2. Mô hình không học được thông tin chuyển động (audit không train; Stage 18b/19 chỉ cho dấu hiệu).
3. Nhãn nào là sai, hay gộp lớp có giúp ích không (không sửa nhãn, không thử).
4. Tỉ lệ bước mà động tác nhìn thấy được trong clip (không xem video).

## 7. Trả lời: tại sao Stage 18 có thể thất bại khi gặp cảnh mới?

Dữ liệu cho thấy các điều kiện **khiến việc tổng quát sang cảnh mới khó**, nhưng không chứng minh nguyên nhân: chỉ 13 cảnh / 27 video, nhiều fold kiểm trên 2–3 video; 3/10 động từ gần như chỉ có ở một cảnh (một trong số đó chủ yếu ở một video), nên khi cảnh đó bị giữ ra, train gần như không có ví dụ đại diện; nhãn là ý định tức thời do người viết tự do, các động từ gần nghĩa (place/position, grasp/pick_up/hold) và đi kèm đồ vật đặc trưng của cảnh; clip cố ý dừng trước kết quả hành động. Ngược lại, giả thuyết “lớp vắng hẳn trong train” bị loại (0/80). Kết hợp với kết quả đã có (Stage 18b chia trong video đạt 37,9 so với 11,2 khi chia theo cảnh), cách đọc phù hợp nhất là: thông tin mô hình dùng được gắn với cảnh/video nhiều hơn với động từ, và dữ liệu không đủ đa dạng cảnh để tách hai yếu tố này.
