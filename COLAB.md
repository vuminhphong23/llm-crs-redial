# Code thực nghiệm — LLM trong hệ thống gợi ý hội thoại (ReDial)

Bản này để Phong **chạy thử được ngay trên Colab**, không phải đọc hiểu hết
mới chạy. Backbone trong code là **Qwen2.5-7B-Instruct** (Phong chốt 08/11, đã báo cáo
GVHD; cô Thu vẫn có thể đổi) — đổi backbone chỉ cần đổi 1 dòng trong
`config.yaml` rồi chạy lại tất cả.

## Chạy trong 15 phút đầu (smoke test)

Trên Colab, mở notebook `thuc-nghiem-LLM-CRS-Colab.ipynb` kèm theo, hoặc làm
tay theo các lệnh dưới (thư mục làm việc là thư mục `code/` này):

```bash
# 1. Kiểm tra GPU (Colab: Runtime > Change runtime type > GPU, T4 trở lên)
nvidia-smi

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Tải ReDial + đếm thật + chia tập (KHÔNG cần GPU, ~1 phút)
python 01_download_count.py
# -> data/data_card.json : số hội thoại / lượt nói / phim ĐẾM TỪ FILE ĐÃ TẢI.
#    Số này mới là số ghi vào bài báo. Kiểm tra rò rỉ = 0.

# 4. Tạo ví dụ đánh giá / huấn luyện (KHÔNG cần GPU)
python 02_build_examples.py

# 5. Xem prompt trước, không tốn GPU
python 03_prompting_baseline.py --dry_run --n 2

# 6. Chạy thật 20 ví dụ valid, chế độ ranking (CẦN GPU, vài phút)
python 03_prompting_baseline.py --split valid --n 20 --mode ranking

# 7. Đánh giá lại từ file dự đoán (KHÔNG cần GPU)
python 05_evaluate.py --predictions outputs/pred_valid_ranking_n20.jsonl
```

## Các file làm gì

| File | Việc | Cần GPU? |
|---|---|---|
| `config.yaml` | Mọi cấu hình ở 1 chỗ: tên model, 4-bit, LoRA, seed 42 | — |
| `common.py` | Hàm chung: đọc jsonl, catalogue phim, tạo ví dụ, metric Recall/NDCG/MRR, ghi nhật ký chạy | — |
| `01_download_count.py` | Tải zip ReDial chính thức, đếm, chia train 90/10, giữ nguyên test 1.342, kiểm tra rò rỉ | Không |
| `02_build_examples.py` | Tạo ví dụ: history + target là phim recommender gợi ý lần đầu (`suggested==1`, chưa nhắc trước đó) | Không |
| `03_prompting_baseline.py` | Prompting baseline: `--mode ranking` (chính), `--mode rag` (retriever TF-IDF), `--mode free` (sinh tự do + đo ảo giác) | Có |
| `04_lora_train.py` | QLoRA 4-bit, r=16, alpha=32, lr 2e-4, loss chỉ trên phần trả lời | Có |
| `05_evaluate.py` | Đánh giá lại file dự đoán bằng đúng 1 hàm metric chung | Không |
| `thuc-nghiem-LLM-CRS-Colab.ipynb` | Notebook chạy từng bước trên Colab | Có |

## Quy tắc không được phá (nếu phá thì số trong bài báo mất giá trị)

1. **Chia theo hội thoại, không chia theo câu nói.** Đã làm trong bước 1.
2. **Tập test chỉ chạy 1 lần cho cấu hình cuối.** Mọi chỉnh prompt / siêu tham
   số chỉ được tune trên tập valid. Đừng chạy test để chọn model rồi báo cáo
   số test đó như kết quả cuối.
3. **1 hàm metric cho mọi model.** Mọi so sánh dùng `common.evaluate_ranking`.
4. **Không ghi số bằng tay.** Mọi lần chạy ghi vào `runs/*.json` kèm lệnh,
   cấu hình, file dự đoán. Số điền vào mục 6 của bài báo phải copy từ file
   `_metrics.json`.
5. **Không bịa, không chép số bài khác.** Bảng kết quả trong bài báo vẫn để
   trống cho tới khi có file `_metrics.json` thật từ code này.

## Baseline KBRD / KGSF / UniCRS — chưa có trong gói này

Để so sánh công bằng, 3 baseline phải chạy lại trong cùng harness, cùng split,
cùng code metric. Cách làm chuẩn là dùng CRSLab
(`https://github.com/RUCAIBox/CRSLab`, lệnh mẫu
`python run_crslab.py --config config/kgsf/redial.yaml`), nhưng CRSLab là
toolkit cũ (PyTorch cũ, 18 model) nên cần 1 môi trường riêng, tách khỏi môi
trường LLM này. Gói này **không giả vờ đã có baseline** — khi Phong chạy
xong prompting + LoRA, bước tiếp theo là lập môi trường CRSLab riêng để chạy
3 baseline, rồi mới điền mục 6.

## Hạn chế đã biết của code này (ghi thẳng, đừng giấu trong bài báo)

- Retriever RAG chỉ dùng **tên phim** (TF-IDF), vì ReDial gốc không có thể
  loại / diễn viên / đạo diễn. Muốn có metadata thì phải merge thêm MovieLens
  — chưa làm trong gói này.
- `preference_state` là heuristic theo quy tắc (liked/disliked từ câu trả lời
  `initiatorQuestions`), không phải nhãn vàng.
- Đánh giá ranking dùng 99 phim âm ngẫu nhiên + 1 đáp án đúng. Đó là bài
  ranking chuẩn để so sánh, không phải trải nghiệm chatbot thật.
- LoRA train 1 epoch trên toàn bộ train sẽ tốn nhiều compute units trên Colab
  — chạy thử 200 ví dụ trước, rồi mới chạy hết.

## Sau khi chạy xong, gửi lại cho mình những file này

- `data/data_card.json` (số đếm thật)
- `outputs/*_metrics.json` (kết quả từng chế độ)
- `runs/*.json` (nhật ký chạy)
- 1 dòng Phong tự viết: "Em hiểu code đang làm gì ở bước ..." — để mình biết
  Phong đã sở hữu được phần này, không phải chỉ bấm Run.
