# Hướng dẫn gán nhãn S7414 (Trang, Hưng, làm độc lập)

- File của mỗi người: `label_S7414_trang.csv`, `label_S7414_hung.csv`. Thứ tự episode trong hai file khác nhau. **Không mở file của người kia.** Không có nhãn tự động trong file.
- Mỗi dòng có `clip` (mp4) và `still` (jpg). Media chỉ nằm trên máy lab: `/media/dhqg/d3/datasets/habit/label_media/S7414/` (không có trên GitHub, cần remote vào máy lab).
- Clip = `front_view` (camera robot, trái) | `human_front_view` (camera trên đầu người, phải), từ đầu episode đến hết **lần chỉ đầu tiên** (theo pedal).
- Điền `k_from_robot_left`: donut người chỉ, đếm **1–4 từ trái sang phải trong ảnh front_view (bên trái)**. Ảnh front_view là góc nhìn của robot, nên trái ảnh = trái của robot.
- Nhìn **tay/ngón tay người**, không dựa vào việc robot đi đâu. Nếu robot đã bắt đầu với tới trong clip, vẫn gán theo người và ghi `robot_moving` vào cột `note`.
- Không chắc thì ghi `1` vào `unclear` (để trống `k`). Episode nào hai người không khớp sẽ bị loại.
