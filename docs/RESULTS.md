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

### Lịch sử nộp leaderboard

| Bản | Hệ thống | Điểm BTC |
|---|---|---:|
| 1 | G1 rule-only | 0.561 |
| 2 | G1 rule-only sau phân tích lỗi public không nhãn (`submissions/public_test_rule_only.jsonl`) | **0.912** |
| 3 | G1 + Qwen3-14B + LoRA pha A (`colab-finetune-phasea`) | **0.777** |
| 4 | G1 + Qwen3-14B-AWQ + LoRA pha A trên T4, bộ chọn mới, G4 đã sửa (ô 7b; nhánh `colab-finetune-phasea-20261104-1350`, `submissions/public_test_lora_phasea.jsonl`, sha256 `8124e92b…`) | **0.895** |
| 5 | Bản cuối: bộ chọn có hiệu chuẩn sự phản đối + kiểm tra hình thức + chốt rulebook, replay trên log của bản 4 (ô 7c, chỉ CPU; nhánh `colab-finetune-phasea-20261104-1534`, `submissions/public_test_lora_phasea_final_de716e81.jsonl`, sha256 `de716e81…`) | **0.909** |

Bản 2 cao một phần vì luật G1 được viết sau khi đọc câu hỏi và dữ liệu public (không nhãn). Đó là luật chung, nhưng
được chọn bằng cách nhìn public, nên 0.912 lạc quan so với private.

Bản 3 kém bản 2 135 điểm (trên 1.000) và khác bản 2 ở 116 câu. Mỗi câu khác nhau đổi điểm tối đa ±2, nên phần lớn thay
đổi của LLM là sai. Trong 116 câu đó: 73 câu LLM thay đáp án G1 không bị đánh dấu, 46 câu có số liệu bị đổi thành
`unanswerable`, và G4 không có ứng viên nào (SQL của adapter dư dấu đóng ```` ``` ````). Sửa trong bộ chọn
(`datamate/select/selector.py`: một phiếu cho mỗi LLM, G1 không bị đánh dấu thì không bị thay, `unanswerable` của riêng
LLM không thay đáp án số) và trong `extract_sql`. Hai bản tiếp theo: replay bộ chọn mới trên log của bản 3 (chỉ CPU), rồi
chạy lại toàn bộ với G4 đã sửa.

**Bản 4 (7b, T4, bộ chọn mới + G4 đã sửa): 0.895.** So với bản 2, bản 4 không thay đáp án nào, trả lời thêm 49 câu mà
G1 bỏ trống hoặc tự đánh dấu nghi ngờ, và bỏ trống 18 câu chỉ vì LLM trái ý G1. Vì giống bản 2 ở 933/1000 câu, điểm
của nó phải nằm trong khoảng 0.845–0.979. 

Đọc điểm 0.895 (chỉ dùng hai con số tổng, không dùng nhãn từng câu). Bản 4 kém bản 2 17 điểm, và mọi chênh lệch nằm ở
67 câu:
- 49 câu mới trả lời: đúng *a*, sai 49 − *a*, đóng góp 2*a* − 49;
- 18 câu mới bỏ trống: bản 2 đúng *c*, sai 18 − *c*, đóng góp 18 − 2*c*.

Cộng lại bằng −17, nên *a* = *c* + 7. Bản 2 đúng khoảng 98% số câu nó trả lời (0.912 với 942 câu trả lời: khoảng 927
đúng, 15 sai), nên phần lớn 18 câu bị bỏ trống có lẽ là câu G1 đúng (*c* ≈ 16–18). Từ đó:
- **Việc LLM phủ quyết G1 làm mất khoảng 14–18 điểm.** Bản 5 sửa điều này: LLM chỉ phủ quyết khi, trên shop giữ riêng, nó
  đúng nhiều hơn sai ở những câu trái ý G1. (Bản 5 lấy lại 13 điểm; xem điểm bản 5 bên dưới.)
- **Các câu LLM trả lời thay G1 đúng khoảng 47–51%** (*a* ≈ 23–25 trên 49), tức gần như tung đồng xu, nên đóng góp
  khoảng 0. Trên shop giữ riêng của train, các câu này gần như luôn đúng, nhưng chỉ có 5–6 câu như vậy và đều giống
  train. Hiệu chuẩn của LLM ở những câu G1 không chắc đang quá lạc quan. Đây là điểm cần sửa tiếp theo, và quan trọng cho
  private, nơi G1 sẽ không chắc ở nhiều câu hơn.

Phân tích này giả định file được nộp là file 7b của nhánh `colab-finetune-phasea-20261104-1350` (sha256 `8124e92b…`).
Nếu là file `_replay` (sha256 `28e7e73c…`; 47 câu mới trả lời, 18 câu mới bỏ trống), kết luận gần như không đổi.

**Bản 5 (7c, bản cuối, chỉ CPU trên log của bản 4).** Thêm ba chốt chặn, đều là luật chung và đều được kiểm trên
train:
- **Sự phản đối của LLM được hiệu chuẩn** (`__dissent__`). Độ tin cậy của LLM khi nó trái ý một đáp án G1 không bị
  đánh dấu được đo trên shop giữ riêng. Theo quy tắc Chow, LLM chỉ đẩy G1 về bỏ trống khi, ở những câu đó, LLM đúng
  nhiều hơn sai. Mặc định 0,5 nghĩa là không phủ quyết. (Ở bản 4, 18 câu bị bỏ trống chỉ vì LLM trái ý.)
- **Kiểm tra hình thức câu hỏi cho mọi ứng viên LLM**, so với câu hỏi gốc, không so với câu LLM đã viết lại. G4 trước
  đây không được kiểm tra gì.
- **Dòng rulebook mà engine chưa biết.** Các dòng của train và public được ghi vào `artifacts/known_rulebook_rows.json`.
  Ở private, câu hỏi nhắc tới một dòng mới thì G1, G2, G3 bị đánh dấu, và chỉ SQL kèm chính dòng đó (làm evidence) được
  trả lời. Trên train và public, chốt này không đổi gì.

**Điểm bản 5: 0.909.** Bản 5 giống bản 4 ở 986/1000 câu. 14 câu khác nhau gồm:
- 13 câu bản 4 bỏ trống vì LLM trái ý G1; bản 5 trả lại đáp án G1;
- 1 câu LLM trả lời thay G1 ở bản 4; bản 5 bỏ trống câu đó.

Mỗi câu chỉ đổi điểm được ±1, nên trước khi nộp, điểm bản 5 phải nằm trong khoảng 0.881–0.909. 0.909 là mức cao nhất,
và chỉ đạt được theo một cách: **cả 13 đáp án G1 được trả lại đều đúng, và câu LLM bị bỏ là câu sai.** Từ đó:
- **Hiệu chuẩn sự phản đối lấy lại đúng 13 điểm** (+1 điểm từ câu LLM sai bị bỏ). Ở bản 4, việc LLM phủ quyết G1 mất
  8 + 2*c*₅ điểm, với *c*₅ là số đáp án G1 đúng trong 5 câu mà bản 5 vẫn bỏ trống.
- **So với bản 2, bản 5 kém 3 điểm.** Bản 5 = bản 2 + 48 câu LLM trả lời thay G1 − 5 câu G1 bị bỏ trống, nên
  (2*a*₄₈ − 48) − (2*c*₅ − 5) = −3, tức *a*₄₈ = *c*₅ + 20. 48 câu LLM trả lời thay G1 đúng 20–25 câu (42–52%). Nếu G1
  đúng cả 5 câu còn bỏ trống, như với 13 câu kia, thì LLM đúng 25/48: 48 câu này được +2 điểm, còn 5 câu bỏ trống mất 5.

Hai chỗ còn kém bản 2 là các câu LLM trả lời thay G1 (vẫn gần như tung đồng xu) và 5 câu LLM còn phủ quyết. Ở private,
G1 sẽ bỏ trống hoặc tự đánh dấu nhiều câu hơn, nên độ chính xác của LLM ở những câu này quyết định phần lớn điểm có LLM.
Bước tiếp theo: hiệu chuẩn riêng độ tin cậy của LLM khi G1 bỏ trống hoặc tự đánh dấu (khởi điểm 0,5, như `__dissent__`).

**Bản 6 (7c lần hai, chỉ CPU trên log của bản 4).** Đọc log của 53 câu bản 5 khác bản 2, không dùng nhãn, kiểm bằng dữ
liệu public và nhãn train. Có bốn lỗi chung. Mọi thay đổi chỉ khiến hệ thống bỏ trống hoặc trả lại đáp án G1; không thay
đổi nào cho LLM thêm câu trả lời.
- **Lỗi bộ chọn: phiếu đồng ý bị tính là phiếu chống.** Ở 3/3 câu đã đọc trong 5 câu bản 5 còn bỏ trống:
  - G1 cho ra một giá trị (độ tin cậy 0,7, vì intent chưa gặp trên train);
  - G3 ra đúng giá trị đó, nhưng bị đánh dấu (0,3);
  - tổng logit bằng 0, tức p = 0,5, nên câu bị bỏ trống.

  Sửa (`selector.py`): họ mạnh nhất trong nhóm cho logit của chính nó, các họ đồng ý khác chỉ được cộng thêm, không được
  trừ đi.
- **Kết quả 0 từ SQL.** Trên train, chỉ 4/2.462 đáp án số bằng 0, và cả 4 đều hỏi về hàng không bán được hoặc giá đổi mà
  lượng bán không đổi. Ở public, LLM trả lời 0 cho 5 câu. Với 3 câu đếm xe SH Mode / Air Blade / Wave Alpha, sổ sửa xe có
  13–21 lượt mỗi loại trong tháng được hỏi. Với câu "xe vô sửa từ 2 lần trở lên", sổ có 22–29 xe như vậy. Bộ kiểm tra
  (`form_guard.py`) đánh dấu kết quả 0 của mọi route LLM. Không đánh dấu khi câu hỏi hỏi về thứ không có ("ế", "không ai
  mua") hoặc hỏi % thay đổi. Bộ máy luật đã có kiểm tra này cho frame của nó.
- **Hình thức câu hỏi** (`consistency.check_form`):
  - hỏi "mấy giờ" mà trả lời bằng một ngày;
  - hỏi "mấy lượt / đơn / phiếu" mà intent là tiền và câu hỏi không nhắc tiền.
- **Thứ sổ sách không ghi** ("ngon nhất", "chất lượng tốt nhất", "khó tính nhất"): không cột nào đo vị ngon, chất lượng
  hay tính khí, nên câu trả lời của LLM chỉ là đoán (thường là mặt hàng bán chạy). Chỉ đánh dấu câu trả lời giá trị của
  LLM; `unanswerable` giữ nguyên.
- **`__solo__`** (`validate.py`): độ tin cậy của mỗi generator LLM ở những câu không có đáp án G1 sạch, prior 0,5. Với G2
  và G3 chỉ có vài câu như vậy trên shop giữ riêng, nên giá trị gần 0,5–0,7. Với SQL, mọi câu trả lời trên shop giữ riêng
  đều được tính, theo cách nhìn leave-intent-out. Ở các câu G1 không chắc, phiếu của LLM bị giới hạn ở mức này.

Áp lên 48 câu LLM trả lời thay G1 ở bản 5, các bộ kiểm tra đánh dấu 9 câu: 5 câu ra 0, 1 câu hỏi giờ, 3 câu hỏi chất
lượng. Cộng với 5 câu G1 được trả lại, bản 6 khác bản 5 ở khoảng 14 câu, nên điểm nằm trong khoảng 0.895–0.923. Nếu G1
đúng ở 5 câu kia (như ở 13 câu lần trước) và 9 câu LLM bị chặn đều sai, điểm là 0.923, cao hơn bản 2 (0.912). Ảnh hưởng
của `__solo__` chỉ biết được khi 7c chạy lại trên log noisy_lora.

**Đổi cách trình bày dữ liệu** (`scripts/perturb_split.py`, 6 shop của fold 0, 553 câu, chỉ G1). Không giá trị nào bị
đổi; chỉ cách sắp xếp file khác đi:

| Biến thể | PA | Đáp án giống bản gốc |
|---|---:|---:|
| bản gốc | 1.0000 | — |
| đảo thứ tự cột | 1.0000 | 553/553 |
| đảo thứ tự sheet | 1.0000 | 553/553 |
| header viết hoa | 1.0000 | 553/553 |
| CSV dùng `;` | 1.0000 | 553/553 |

Phần ingest không phụ thuộc thứ tự cột, thứ tự sheet, chữ hoa/thường của header hay ký tự phân tách của CSV. Phép thử
này **không** đo tên cột mới hay sổ sách của loại shop mới; đó là phần leave-intent-out và chốt rulebook nhắm tới.

| | Bản nộp 1 (điểm BTC **0.561**) | Bản 2 (hiện tại) |
|---|---:|---:|
| Trả lời giá trị | 850 | 920 |
| `unanswerable` (có lý do) | 21 | 22 |
| `abstain` | 129 | 58 |

Theo shop ở bản 2 (trả lời / bỏ trống): dev_001 86/1 · dev_002 37/4 · dev_003 126/3 · dev_004 89/7 · dev_005 80/7 ·
dev_006 103/8 · dev_007 58/2 · dev_008 93/6 · dev_009 21/0 · dev_010 55/4 · dev_011 38/2 (điện thoại) ·
dev_012 44/6 (sửa xe) · dev_013 50/7 (mẹ & bé) · dev_014 62/1 (điện thoại + sổ nợ/sổ góp/bảo hành).

Thời gian: ~30–90 s cho 1.000 câu trên CPU (sau khi đã có cache), vài phút khi nạp dữ liệu lần đầu.
File nộp: `submissions/public_test_rule_only.jsonl`.

### 2.1 Phân tích lỗi bản 1 (không có nhãn) và các sửa đổi

PA = (đúng − sai)/N nên 0.561 với 871 câu trả lời ⇒ khoảng 716 đúng / 155 sai: **độ chính xác khi trả lời ~82 %**,
thấp hơn nhiều so với train. Đọc lại 1.000 câu public theo intent (`scripts/review_run.py`) và chẩn đoán ingest
không cần nhãn (`scripts/check_consistency.py`) cho thấy các nhóm lỗi *tổng quát* sau (không sửa theo từng câu):

| Nhóm lỗi | Ví dụ | Sửa |
|---|---|---|
| Trạng thái đơn đọc sai | "Khách trả" (261 đơn) không bị coi là trả hàng; ghi chú "kiểm tra lại" bị coi là trả hàng; "khách mang trả" | regex trạng thái có ranh giới từ, loại "kiểm tra" |
| Rơi về intent `revenue` với độ tin cậy 0.985 | "nhóm X thì mặt hàng nào thu về nhiều tiền nhất", "mấy % tổng doanh thu", "th7, th8, th9 lần lượt", "mấy bữa … từ 10 củ", "… đúng vậy không?" | định tuyến đúng intent + **bộ kiểm tra hình thức câu ↔ kiểu đáp án** (`nlu/consistency.py`) |
| Thời gian | "tính luôn hôm nay thì 7 ngày gần nhất" → chỉ hôm nay; "từ đầu tháng/mùng 1 đến hết hôm nay"; "T3/2026 tới T5/2026" thiếu tháng giữa; "tháng 7 năm rồi"; "cùng số ngày đầu tháng trước"; "tới hôm nay, đơn đặt th3" | `nlu/timeparse.py` |
| Từ địa phương / gõ sai | "o đờ" (order), "xi ô đi" (COD), "củ" (triệu), "bữa" (ngày), "tháng ba", "so dới", "nhân diên", "caonhất" (dính chữ) | `textnorm.py` |
| Tên món vùng miền | "mực rán" = "mực chiên", "lạc" = "đậu phộng", "cá tràu" = "cá lóc", "cát xây trát" = "cát xây tô", "cốc sạc" = "củ sạc" | `nlu/linker.py::dialect` |
| Shop không có cột nhóm hàng mà câu nêu "nhóm X" | trả số của cả shop | bỏ trống (bộ kiểm tra hình thức) |
| Sổ chuyên ngành chưa mô hình hoá | sổ góp, sổ nợ tự ghi (dev_014), đếm theo loại việc sửa xe, "khách sộp", "đắt khách", "đứt hàng", "cận date", số lượng nhập của một mặt hàng | `domains/books.py`, `domains/repair.py`, `nlu/rules.py` |
| Đáp án rỗng / bằng 0 | xếp hạng trả `None`; tổng/đếm = 0 (train không có đáp án 0 cho các intent này) | bỏ trống |

So với bản 1: 778 câu giữ nguyên, 141 câu đổi đáp án, 76 câu từ bỏ trống sang trả lời, 5 câu từ trả lời sang bỏ trống
(còn lại do định tuyến lại sang intent đúng). Mọi thay đổi được kiểm lại trên train (PA giữ **1.000**, bộ kiểm tra
hình thức chặn **0** câu train) và trên train làm nhiễu (**0.997**, 0 câu sai).

**Tái lập / đa nền tảng:** `predict.sh` chạy trên Linux (WSL Ubuntu 22.04, Python 3.10, pandas 2.3, numpy 2.2) cho
**1000/1000 đáp án giống hệt** bản Windows (Python 3.15, pandas 3.0) — phần luật là tất định.

## 3. Các cơ chế chống trả lời sai (đã kiểm chứng trên train/nhiễu)

* **Coverage / OOV**: frame của G1 chỉ được tin khi mọi từ nội dung của câu hỏi được giải thích (từ vựng câu
  hỏi train đã che tên thực thể, tên thực thể của shop, từ đệm/phương ngữ). Ngược lại độ tin cậy bị chặn 0.3 < τ.
* **Thiếu bảng**: shop còn bảng chưa phân loại được ⇒ không khẳng định `unanswerable` vì thiếu dữ liệu, chỉ bỏ trống.
* **Mặt hàng không liên kết được** trong câu hỏi doanh thu/số lượng ⇒ bỏ trống (tránh trả doanh thu cả shop).
* **Hoà ở biên top-k** ⇒ bỏ trống (đề đã loại các câu hoà nên hoà = ta hiểu sai).
* **Rulebook đổi luật** ⇒ intent phụ thuộc không còn được tin (`datamate/rulebook.py`).
* **Hình thức câu ↔ kiểu đáp án** (`datamate/nlu/consistency.py`): "… nào/ai" mà trả số, "%" mà trả tiền, "lần lượt"
  mà trả một số, "đúng không?" mà trả số, "nhóm X" không liên kết được, đáp án rỗng hoặc tổng/đếm = 0 ⇒ bỏ trống.
  Kiểm bằng `scripts/check_guard.py`: chặn 0/3.000 câu train, chặn các câu sai tự tin ở public bản 1.

## 4. Tái lập

```bash
export DATAMATE_CACHE=.cache
python -m datamate.predict --data data/train --out runs/train/submission.jsonl --llm none        # in PA
python scripts/eval_noise.py data/train 1 0.7                                                       # PA trên bản nhiễu
python -m datamate.predict --data data/public_test --out runs/public/submission.jsonl --llm none
python scripts/build_artifacts.py data/train && python scripts/calibrate.py <eval dump> artifacts/reliability.json
```

## 5. Thành phần XiYan-SQL — đo trên train, không cần LLM (2026-11-03)

Các số dưới đây là **trong mẫu** (thuật ngữ của Schema Filter, luật G1 và bộ biên dịch frame→SQL được viết khi đọc
train). Chúng kiểm chứng các thành phần làm đúng việc; chúng **không** đo generalization và **không** đo phần LLM.

**Silver SQL** (`scripts/build_sql_silver.py`): frame G1 đúng đáp án → SQL ở 3 định dạng → thực thi trên view → giữ
nếu kết quả bằng đáp án vàng. 1.689 câu biên dịch được, **1.671** câu có SQL đã kiểm chứng (98,9 %).

| intent | frame đúng | biên dịch | kiểm chứng |
|---|---:|---:|---:|
| revenue | 810 | 669 | 669 |
| order_count | 180 | 180 | 180 |
| aov | 169 | 169 | 169 |
| share | 143 | 84 | 84 |
| cancel_rate | 140 | 139 | 139 |
| best_day | 126 | 112 | 112 |
| cancel_count | 100 | 100 | 100 |
| top_products | 96 | 87 | 87 |
| top_customers | 87 | 86 | 69 |
| purchase_total | 41 | 41 | 40 |
| monthly_list | 22 | 22 | 22 |

**Schema Filter** (`scripts/eval_schema_filter.py`, bộ chọn cột heuristic, 1.671 câu):

| schema | recall cột trung bình | giữ đủ mọi cột của SQL đúng | số cột trung bình |
|---|---:|---:|---:|
| S_rtrv (chỉ truy hồi) | 0,568 | 5,4 % | 11,2 |
| S1 (vòng 1 + PFKey + cột nghiệp vụ) | 0,960 | **91,5 %** | 11,1 |
| S2 (vòng 2) | 0,998 | **99,6 %** | 16,9 |
| full | 1,000 | 100 % | 42,3 |

Giống xu hướng trong paper: S1 chính xác hơn (ít cột), S2 recall cao hơn. Ba thay đổi quan trọng nhất:
- bổ sung cặp tên ↔ khoá của thực thể (`product_name` ↔ `product_key`);
- thêm đường lexicon (đồng nghĩa → cột);
- luôn giữ cột điều kiện nghiệp vụ và cột ngày.

**SQL checks** (`scripts/eval_sql_checks.py`): **0 / 5.013** SQL đúng bị báo động nhầm. Tỉ lệ phát hiện near-miss làm đổi
đáp án:

| mutation | phát hiện |
|---|---:|
| bỏ `is_valid` | 999 / 1.074 (93 %) |
| lệch một tháng | 1.644 / 1.645 (100 %) |
| chặn trên `<` bỏ ngày cuối | 1.381 / 1.394 (99 %) |
| JOIN nhân bản dòng | 759 / 759 (100 %) |
| `total_due` thay doanh thu | 330 / 348 (95 %) |
| sai `LIMIT` | 94 / 94 (100 %) |

**Dữ liệu SFT** (`training/build_sql_sft.py --n-aug 1`): 20.379 bản ghi.

| task | số bản ghi |
|---|---:|
| `nl2sql_base`, `nl2sql_cte`, `nl2sql_alias` | 3.342 mỗi loại |
| `question_inference` | 1.671 |
| `evidence_inference` | 1.671 |
| `self_refine` | 5.348 |
| `select` | 1.663 |

**Validation theo business** (`scripts/validate.py`, 5 fold, log G1): PA 0,9997, Brier 0,0011, ECE 0,022 ở τ = 0,6.
Kết quả lạc quan vì luật G1 viết trên toàn bộ train; quy trình này dành cho generator LLM sau khi có log GPU.

**Hồi quy:** train vẫn PA 1,000. File public chế độ CPU **giống hệt từng byte** bản trước, vì G4 chỉ chạy khi có LLM.
