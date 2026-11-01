# Kết quả đo được (chế độ CPU, chỉ generator luật G1 + bộ chọn)

Môi trường phát triển: Windows, CPU 16 luồng, **không GPU** ⇒ các generator LLM (G2–G4) chỉ được kiểm thử
bằng LLM giả lập (`tests/test_agent_mock.py`), chưa đo độ chính xác. Mọi số dưới đây là của
`python -m datamate.predict --llm none`.

## 1. Train (3.000 câu, 30 shop) — có đáp án

| Chế độ | Đúng | Sai | Bỏ trống | PA | Thời gian |
|---|---:|---:|---:|---:|---:|
| Agent đầy đủ (G1 + coverage + selector) | 3.000 | 0 | 0 | **1.000** | ~4 phút (gồm nạp 30 shop từ cache) |
| Câu hỏi train **bị làm nhiễu** kiểu public (`scripts/eval_noise.py`, mức 0.7) | 2.990 | 0 | 10 | **0.997** | |

> **Đọc đúng con số:** PA = 1.000 là **trong mẫu** — luật được viết trong lúc phân tích lỗi trên chính các câu
> train này (không mã hoá đáp án hay id; mọi xử lý suy ra từ file + README + rulebook). Bản nhiễu (bỏ dấu, gõ
> Telex sai, viết tắt, phương ngữ, ngày viết tắt "Th7/25"…) ước lượng hành vi trên câu hỏi lộn xộn: **0 câu sai**,
> 0.3 % bỏ trống. Lưu ý: bộ sinh nhiễu và bộ chuẩn hoá dùng chung hiểu biết về viết tắt nên con số này lạc quan;
> câu hỏi public/private còn nhiều kiểu nhiễu và khái niệm mới hơn (xem §2).

Lịch sử phát triển (cùng 3.000 câu): 0.881 (bản đầu, 23 shop) → 0.972 → 0.998 → 1.000.

### Phân bố intent mà G1 nhận diện trên train

| intent | số câu | intent | số câu |
|---|---:|---|---:|
| revenue | 810 | price_on | 63 |
| units | 206 | stock | 43 |
| growth | 202 | purchase_total | 41 |
| order_count | 180 | repayments | 35 |
| aov | 169 | unsold_products | 34 |
| share | 143 | orders_min_amount / revenue_multi | 26 / 26 |
| cancel_rate | 140 | increasing / monthly_list | 23 / 22 |
| best_day | 126 | top_debtors / unanswerable | 21 / 21 |
| cancel_count | 100 | debt_flow / days_revenue / stock_split / cust_min_orders | 20 mỗi loại |
| top_products | 96 | products_min_orders / debtor_count / best_month / cust_min_orders_pct / delivery_trips / debt_total | 14–18 |
| top_customers | 87 | debt_overdue, expiring, gross_profit, top_staff, top_suppliers, stock_value, nearest_expiry, debt_aging, purchase_minus_sales, net_profit, out_of_stock | 1–8 |
| price_effect | 84 | | |
| avg_purchase_price | 63 | | |

## 2. Public test (1.000 câu, 14 shop) — **không có đáp án**

| | Số câu |
|---|---:|
| Trả lời giá trị | 850 |
| `unanswerable` (có lý do) | 21 |
| `abstain` | 129 |

Theo shop (trả lời / bỏ trống): dev_001 81/6 · dev_002 38/3 · dev_003 123/6 · dev_004 85/11 · dev_005 78/9 ·
dev_006 93/18 · dev_007 54/6 · dev_008 89/10 · dev_009 20/1 · dev_010 54/5 · dev_011 33/7 (điện thoại) ·
dev_012 38/12 (sửa xe) · dev_013 49/8 (mẹ & bé) · dev_014 36/27 (điện thoại + trả góp/bảo hành).

Thời gian: ~90 s cho 1.000 câu trên CPU (sau khi đã có cache), ~4 phút khi nạp dữ liệu lần đầu.
File nộp: `submissions/public_test_rule_only.jsonl`.

Bỏ trống tập trung ở các sổ chuyên biệt chưa được mô hình hoá (sổ góp, sổ nợ tự ghi, đếm theo loại việc sửa xe,
hàng chưa giao) và câu có từ ngoài phạm vi hiểu biết của bộ luật (cơ chế *coverage*). Đây là chỗ các generator
LLM (G2–G4) được kỳ vọng bổ sung khi chạy trên GPU.

**Tái lập / đa nền tảng:** `predict.sh` chạy trên Linux (WSL Ubuntu 22.04, Python 3.10, pandas 2.3, numpy 2.2) cho
**1000/1000 đáp án giống hệt** bản Windows (Python 3.15, pandas 3.0) — phần luật là tất định.

## 3. Các cơ chế chống trả lời sai (đã kiểm chứng trên train/nhiễu)

* **Coverage / OOV**: frame của G1 chỉ được tin khi mọi từ nội dung của câu hỏi được giải thích (từ vựng câu
  hỏi train đã che tên thực thể, tên thực thể của shop, từ đệm/phương ngữ). Ngược lại độ tin cậy bị chặn 0.3 < τ.
* **Thiếu bảng**: shop còn bảng chưa phân loại được ⇒ không khẳng định `unanswerable` vì thiếu dữ liệu, chỉ bỏ trống.
* **Mặt hàng không liên kết được** trong câu hỏi doanh thu/số lượng ⇒ bỏ trống (tránh trả doanh thu cả shop).
* **Hoà ở biên top-k** ⇒ bỏ trống (đề đã loại các câu hoà nên hoà = ta hiểu sai).
* **Rulebook đổi luật** ⇒ intent phụ thuộc không còn được tin (`datamate/rulebook.py`).

## 4. Tái lập

```bash
export DATAMATE_CACHE=.cache
python -m datamate.predict --data data/train --out runs/train/submission.jsonl --llm none        # in PA
python scripts/eval_noise.py data/train 1 0.7                                                       # PA trên bản nhiễu
python -m datamate.predict --data data/public_test --out runs/public/submission.jsonl --llm none
python scripts/build_artifacts.py data/train && python scripts/calibrate.py <eval dump> artifacts/reliability.json
```
