<div align="center">

# VLIA-UET: Tăng Cường Khả Năng Hiểu Ý Định Người Dùng Cho Robot Cộng Tác Dựa Trên Mô Hình VLA

### *Enhancing User Intention Understanding for Collaborative Robots based on Vision-Language-Action Models*

[![Status](https://img.shields.io/badge/Status-Active%20Research-2ea44f?style=flat-square)](#current-status)
[![Backbone](https://img.shields.io/badge/Backbone-SmolVLA%20%7C%20SmolVLM2-8a2be2?style=flat-square)](#smolvla-integration)
[![Framework](https://img.shields.io/badge/Framework-LeRobot%20%7C%20PyTorch-ff69b4?style=flat-square)](#environment--setup)
[![Benchmark](https://img.shields.io/badge/Benchmark-EgoIntent%20%7C%20LIBERO--10-blue?style=flat-square)](#preliminary-results)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

<br/>

**Đồ án tốt nghiệp / Undergraduate Research Thesis — Trường Đại học Công nghệ (VNU-UET)**

*Core Research Focus: Early Goal Disambiguation from Egocentric Video under Shared Sub-actions*

---

</div>

## 📌 Mục lục / Table of Contents

- [1. Giới thiệu & Đặt bài toán (Research Problem)](#1-giới-thiệu--đặt-bài-toán)
- [2. Giả thuyết nghiên cứu (Core Hypothesis)](#2-giả-thuyết-nghiên-cứu)
- [3. Câu hỏi nghiên cứu (Research Questions)](#3-câu-hỏi-nghiên-cứu)
- [4. Phương pháp đề xuất (Methodology)](#4-phương-pháp-đề-xuất)
- [5. Kết quả thực nghiệm Stage-A (Experimental Results)](#5-kết-quả-thực-nghiệm-stage-a)
- [6. Phân tích lỗi nội tác vụ (Same-Task Error Analysis)](#6-phân-tích-lỗi-nội-tác-vụ)
- [7. Quy thức dự đoán sớm (Temporal Anticipation Protocol)](#7-quy-thức-dự-đoán-sớm)
- [8. Tích hợp SmolVLA & Đánh giá Downstream (Robot Integration)](#8-tích-hợp-smolvla--đánh-giá-downstream)
- [9. Cấu trúc mã nguồn (Repository Structure)](#9-cấu-trúc-mã-nguồn)
- [10. Hướng dẫn cài đặt & Thực thi (Quickstart)](#10-hướng-dẫn-cài-đặt--thực-thi)
- [11. Kế hoạch nghiên cứu tiếp theo (Roadmap)](#11-kế-hoạch-nghiên-cứu-tiếp-theo)

---

## 1. Giới thiệu & Đặt bài toán

Trong các kịch bản tương tác người -- robot (Human-Robot Interaction -- HRI), việc chỉ dựa vào câu lệnh ngôn ngữ tĩnh không đủ để robot cộng tác (Cobot) hành động nhịp nhàng. Trong các tác vụ thao tác tay của con người, **cùng một hành vi vật lý ban đầu (sub-action -- WHAT) có thể dẫn tới nhiều đích đến hoàn toàn khác nhau (intentions/goals -- WHY)**.

```text
               ┌──► move to shelf  (G1: Cất vào giá)
pick up cup ───┼──► put into box   (G2: Đóng gói vào hộp)
 (WHAT)        └──► move to tray   (G3: Thu dọn vào khay)
```

> **Bài toán nghiên cứu trọng tâm:**  
> *Làm thế nào để phân định sớm mục tiêu thao tác của con người từ video egocentric khi các hành động ban đầu có tính nhập nhằng cao, trước khi quỹ đạo vật lý bộc lộ rõ ràng đích đến?*

Dự án định vị việc **dự đoán ý định (Intention/Goal Disambiguation)** là đóng góp khoa học chính, trong khi mô hình Vision-Language-Action (VLA) đóng vai trò giao diện tiếp nhận hạ tầng (downstream interface).

---

## 2. Giả thuyết nghiên cứu

> **Hypothesis:**  
> Khi nhiều mục tiêu cùng tương thích với một hành vi quan sát được, việc **ràng buộc suy luận mục tiêu (WHY) đi qua biểu diễn hành động con (WHAT)** trên đồ thị tính toán trực tiếp sẽ tạo ra một *semantic information bottleneck* hiệu quả, giúp lọc nhiễu nền và phân biệt các đích đến tốt hơn so với dự đoán trực tiếp hoặc dùng auxiliary loss độc lập.

```text
Video Egocentric Prefix V_{1:t}
              │
              ▼
  Frozen Visual Backbone (SmolVLM2)
              │
              ├────────────────────────► WHAT (Hành động con quan sát)
              │                                    │
              ▼                                    ▼
       Direct Path                       Guided Reasoning Path
              │                                    │
              └────────────────► WHY ◄─────────────┘
                         (Đích đến ngữ nghĩa)
                                  │
                                  ▼
                         z_int ∈ R^256 (Intention)
                                  │
                                  ▼
                     Intention Adapter (256 → 960)
                                  │
                                  ▼
                      SmolVLA Multimodal Prefix
```

---

## 3. Câu hỏi nghiên cứu

* **RQ1 (Goal Disambiguation):** Mô hình có thể phân định các mục tiêu có chung hành vi khởi đầu từ video egocentric với độ chính xác cao hơn ngẫu nhiên không?
* **RQ2 (Computational Path Inductive Bias):** Việc đưa $\widehat{\text{WHAT}}$ vào đường suy luận trực tiếp của WHY có thực sự vượt trội hơn dự đoán trực tiếp (Direct Why) và auxiliary loss độc lập không?
* **RQ3 (Strict Temporal Anticipation):** Hiệu năng dự đoán thay đổi như thế nào khi quan sát bị giới hạn nghiêm ngặt trước thời điểm rẽ nhánh vật lý ($t_{\mathrm{obs}} \le t_{\mathrm{divergence}} - \Delta$)?
* **RQ4 (Robot Relevance):** Ý định dự đoán có thể chuyển đổi thành tín hiệu điều khiển hữu ích (qua Intention Adapter của SmolVLA hoặc proxy điều phối kỹ năng robot) không?

---

## 4. Phương pháp đề xuất

Hệ thống được thiết kế theo quy trình hai pha tinh gọn:

### Pha 1: Predicted-What-Guided Reasoning kết hợp Same-Task Hard-Negatives
1. Trích xuất đặc trưng khung hình từ mô hình nền thị giác đóng băng `SmolVLM2-500M-Video-Instruct`.
2. Dự đoán vector hành động con tức thời $\hat{z}_t^{\mathrm{what}}$.
3. Kết hợp visual context và $\hat{z}_t^{\mathrm{what}}$ qua reasoner có cổng đóng mở (learned gate) để sinh $z_{t,\mathrm{base}}^{\mathrm{why}}$.
4. Nhánh WHAT được tối ưu bằng hàm mất mát tương phản kết hợp mẫu âm nội tác vụ (Same-task Hard Negative) để tách các trạng thái gần nhau.

### Pha 2: Frozen Residual Reranker với Soft-MRR
1. Sau khi Pha 1 hội tụ, **đóng băng hoàn toàn** toàn bộ mô hình cơ sở để bảo toàn không gian biểu diễn đã học.
2. Huấn luyện một mạng thặng dư nhỏ ($\Delta z_t^{\mathrm{why}}$) nhận đầu vào $[z_{t,\mathrm{base}}^{\mathrm{why}}; \hat{z}_t^{\mathrm{what}}]$ bằng hàm mục tiêu **Soft-MRR** (surrogate khả vi của MRR) để tinh chỉnh thứ hạng mà không gây xung đột gradient.

---

## 5. Kết quả thực nghiệm Stage-A

Thực nghiệm được thực hiện trên benchmark **EgoIntent** (split cố định theo `video_uid`: Train gồm 556 mẫu indoor, Validation gồm 215 mẫu outdoor). Kết quả trung bình qua 3 seed ngẫu nhiên (Mean $\pm$ Std):

| Cấu hình mô hình | Why MRR | Top-1 Accuracy | Top-3 Accuracy | Ghi chú & Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **Random Baseline** | `0.0420` | `0.0087` | `0.0259` | Xác suất ngẫu nhiên đa nhãn |
| **Direct Why** | `0.0475 ± 0.0061` | `0.0186 ± 0.0047` | `0.0372 ± 0.0081` | Baseline dự đoán trực tiếp |
| **Auxiliary What + Why** | `0.0471 ± 0.0030` | `0.0155 ± 0.0054` | `0.0403 ± 0.0054` | Head độc lập không cải thiện |
| **Predicted-What Guided** | `0.0518 ± 0.0015` | `0.0217 ± 0.0027` | `0.0434 ± 0.0027` | **+8.9%** so với Direct Why |
| **+ What Hard-Negative** | `0.0540 ± 0.0023` | `0.0279 ± 0.0047` | `0.0419 ± 0.0047` | Phân tách không gian nội tác vụ |
| **+ Frozen Residual Reranker** | **`0.0551 ± 0.0009`** | **`0.0295 ± 0.0027`** | **`0.0434 ± 0.0027`** | **+15.9%** so với Direct Why |
| *Baseline tĩnh: Mean-pool 8 frames* | `0.0589 ± 0.0016` | `0.0233 ± 0.0046` | `0.0419 ± 0.0059` | Baseline không cấu trúc |
| *Diagnostic: Oracle What (Privileged)* | `0.1055 ± 0.0067` | `0.0527 ± 0.0054` | `0.0915 ± 0.0194` | Cận trên lý thuyết (chẩn đoán) |

> [!NOTE]
> * **Về giả thuyết Computational Path:** Việc chuyển từ Auxiliary Head ($0.0471$) sang Predicted-What Guided ($0.0518$) chứng minh giá trị của nút thắt ngữ nghĩa.
> * **Tính minh bạch học thuật:** Mô hình tốt nhất ($0.0551$) xấp xỉ mức baseline tĩnh `Mean-pool 8 frames` ($0.0589$) và chưa vượt qua nó trên thước đo tổng thể, nhưng vượt trội ở khả năng phân định các ca khó nội tác vụ.
> * **Oracle What ($0.1055$):** Khẳng định tiềm năng lớn của phương pháp nếu chất lượng nhận diện hành động con tiếp tục được nâng cao.

---

## 6. Phân tích lỗi nội tác vụ

Phân tích chi tiết 626 lỗi Top-1 trên tập kiểm định cho thấy:

```text
Tổng số lỗi Top-1: 626
├── Lỗi trong cùng tác vụ (Same-Task Confusion): 510 mẫu (81.5%)  ◄── ĐIỂM NGHẼN CHÍNH
└── Lỗi khác tác vụ (Cross-Task Confusion):      116 mẫu (18.5%)
```

Con số **81.5% lỗi nhầm lẫn nội tác vụ** chứng minh rằng mô hình không gặp khó khăn trong việc hiểu ngữ cảnh chung, mà gặp bế tắc ở việc **phân định chi tiết giữa các đích đến cùng chia sẻ hành động ban đầu**. Đây chính là căn cứ thực nghiệm then chốt củng cố cho bài toán Goal Disambiguation.

---

## 7. Quy thức dự đoán sớm (Temporal Anticipation)

Để ngăn chặn hiện tượng rò rỉ dữ liệu khi quỹ đạo tay đã bộc lộ đích đến, quy thức đánh giá áp dụng quy tắc loại trừ kết cục (**Outcome Exclusion Rule**):

$$t_{\mathrm{obs}} \le t_{\mathrm{divergence}} - \Delta$$

* $t_{\mathrm{obs}}$: Thời điểm kết thúc cửa sổ quan sát tiền tố.
* $t_{\mathrm{divergence}}$: Thời điểm vật lý mà tay/vật thể bắt đầu rẽ hướng quỹ đạo rõ ràng hướng về đích đến (xấp xỉ bằng thời điểm nhấc vật thể $t_{\mathrm{lift}}$).
* $\Delta$: Biên an toàn thời gian tối thiểu ($\Delta \ge 0.5\text{s}$).

Ngoài ra, đồ án ghi nhận các đường cong hiệu năng theo mốc quan sát $10\%, 25\%, 50\%, 75\%$ như một phân tích phụ trợ.

---

## 8. Tích hợp SmolVLA & Đánh giá Downstream

### Giao diện Intention Adapter
* Biểu diễn ý định $z_t^{\mathrm{int}} \in \mathbb{R}^{256}$ được chiếu tuyến tính lên không gian ẩn $d_{\mathrm{VLM}} = 960$ của SmolVLA.
* Token ý định được chèn ngay trước token trạng thái robot:
  $$P_{\mathrm{VLIA}} = \left[ E_{\mathrm{image}} ; E_{\mathrm{language}} ; E_{\mathrm{int}} ; E_{\mathrm{state}} \right]$$

### Kết quả kiểm thử kỹ thuật (Unit & Integration Tests)
Toàn bộ các kiểm thử tính toàn vẹn hệ thống đã hoàn thành:
- [x] **Gradient flow:** Đạo hàm từ action loss truyền trơn tru về Intention Adapter ($\nabla W > 0$).
- [x] **KV-cache inference:** Token ý định được cache hợp lệ ngay từ bước forward đầu tiên.
- [x] **Flow matching consistency:** Không làm thay đổi phân phối huấn luyện action chunking của LeRobot.
- [x] **Baseline fallback:** Khôi phục nguyên trạng SmolVLA gốc khi không kích hoạt intention.

### Trạng thái Benchmark LIBERO-10
* Đã loại bỏ các run 6-D cũ bị sai lệch schema cấu hình.
* **Baseline 7-D chính thức** đạt **24% success rate** trên 100 episodes (đóng vai trò mốc so sánh có kiểm soát).
* Nhằm tránh rủi ro trễ hạn đồ án và hiện tượng dư thừa ngữ nghĩa (khi prompt ngôn ngữ của LIBERO đã chỉ định mục tiêu), nhánh robot policy được định vị là **Downstream Validation / Skill Dispatching Proxy** thay vì mục tiêu nghiên cứu độc lập.

---

## 9. Cấu trúc mã nguồn

```text
vlia-uet/
├── config/              # File cấu hình task, split dữ liệu và suite thử nghiệm
├── datasets/            # Adapter nạp dữ liệu (EgoIntent, Ego4D, ENIGMA-360)
├── models/              # Các biến thể Intention Encoders (Temporal, Mean-pool)
├── policies/            # Triển khai SmolVLA và Intention Adapter mở rộng
│   ├── intention/       # Clean stage-A encoder và module adapter
│   └── smolvla/         # Modeling, configuration và processor của SmolVLA
├── results/             # Nhật ký thực nghiệm, bảng số liệu và báo cáo chi tiết
├── scripts/             # Kịch bản huấn luyện, kiểm thử và vẽ đồ thị
│   ├── train_matched_oracle_LIBERO7D_FIXED.py  # Script huấn luyện downstream
│   └── run_what_hardneg_sweep.sh               # Script chạy quét tham số
├── thesis.tex           # Bản thảo đồ án tốt nghiệp chuẩn LaTeX (5 chương)
├── requirements.txt     # Danh sách các gói phụ thuộc Python
└── README.md            # Tài liệu tổng quan dự án
```

---

## 10. Hướng dẫn cài đặt & Thực thi

### 1. Chuẩn bị môi trường

```bash
# Tạo môi trường ảo Python 3.10+
conda create -n vlia python=3.10 -y
conda activate vlia

# Cài đặt các gói phụ thuộc cơ bản
pip install -r requirements.txt

# Cài đặt LeRobot từ mã nguồn
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot && pip install -e ".[smolvla]"
```

### 2. Chạy kiểm tra suy luận Stage-A

```bash
# Kiểm tra mô hình dự đoán ý định với What Hard-Negative
python scripts/train_egointent_what_hardneg.py --help
```

### 3. Kiểm tra tích hợp chính sách SmolVLA

```bash
# Kiểm tra import và assertion cấu hình LIBERO 7-D
python scripts/train_matched_oracle_LIBERO7D_FIXED.py --help
```

---

## 11. Kế hoạch nghiên cứu tiếp theo

1. **Khóa split đánh giá độc lập (Test Set):** Chia tập kiểm thử sạch chưa qua tinh chỉnh trên EgoIntent và báo cáo kèm khoảng tin cậy 95% (Bootstrap CI).
2. **Stress-test OOD trên 48 video tự quay:** Thực hiện suy luận không đổi trọng số (zero-shot inference) trên tập dữ liệu thao tác tự thu thập để phân tích định tính các ca thành công và thất bại.
3. **Hoàn thiện bản thảo đồ án tốt nghiệp:** Xuất bản và bảo vệ báo cáo tốt nghiệp dựa trên file [thesis.tex](thesis.tex).

---

<div align="center">

**VLIA-UET Research Team**  
*Khoa Công nghệ Thông tin — Trường Đại học Công nghệ, ĐHQGHN*

</div>