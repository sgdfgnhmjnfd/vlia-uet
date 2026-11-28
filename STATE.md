# STATE Stage 23 Breakfast
- 28/11 22:28 | Phần 0.2 | CHƯA TẢI XONG: /data/breakfast/data.zip 17,95/30,21 GB, wget -c (pid 8868) đang chạy ~1,7 MB/s, dự kiến xong ~00:30 29/11. Dừng theo luật, chưa làm gì tiếp. Ghi chú: file ở /data/breakfast/ (chưa phải /data/breakfast_raw/), sẽ mv sau khi tải xong.
- 29/11 00:57 | tải xong | 30.210.005.282 byte (đủ theo content-length). mv /data/breakfast -> /data/breakfast_raw
- 29/11 01:10 | Phần 0 | dạng A (MS-TCN, data/breakfast/{features,groundTruth,splits,mapping.txt}); md5 078aa08875747e6264b892ae6e0ac7be (chưa đối chiếu zenodo); meta giải nén ra /data/breakfast_raw/meta; scikit-learn 1.7.2 + scipy 1.15.3 + threadpoolctl 3.8.0 cài thêm vào .venv bằng uv; inventory -> results/stage23_breakfast/inventory.md, shared_actions.csv, astar.json; Gate D0 PASS (|A*|=6); đọc 20 video 3,2 s -> cả bộ ~5 phút; nửa 2 đổi giữa khung liên tiếp ~2,2x nửa 1 (chỉ gợi ý nửa 2 = flow, không có nguồn -> gọi nửa 1/nửa 2). testzip đang chạy nền -> /data/breakfast_raw/testzip.txt
- 29/11 01:00 | Phần 1 | prereg.md sha256 281df06d29fb33de72c7cb48c395fc0c7542a11f8601549ea30de65f7caba0d9
- 29/11 01:03 | Phần 0 | testzip OK (None). selftest OK. Bắt đầu cache.
- 29/11 01:16 | Phần 2 cache | xong 1712 video, 1,2 GB /data/breakfast_cache. Bắt đầu partA (log logs/stage23_partA.log)
- 29/11 01:29 | partA split1 | xong 746 s (frame clf+crossfit ~4 phút, linear ~5 phút, T 228 s). Ước còn: partA 3 split ~40 phút, partB ~?; T kênh 128 giữ nguyên (<30 phút/split)
- 29/11 01:57 | partA xong (4 split). partB lỗi pour_oil split4 rỗng -> sửa bỏ qua ô rỗng (DEVIATIONS 1), chạy lại partB + table
- 29/11 01:58 | partB + table xong. HA-1..4, HB-1..3 theo luật: HA-2/3/4 PASS, HB-1/2/3 PASS, kết luận 1. bang_tom_tat.md. Bắt đầu explore (TCN)
