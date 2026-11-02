# DataMate-XiYan

*Zalo AI Challenge 2026 — DataMate track*

DataMate-XiYan trả lời câu hỏi tiếng Việt đời thường trực tiếp từ dữ liệu kinh doanh **CSV, Excel và SQLite**. Hệ thống kết hợp **bộ phân tích luật (rule-based semantic parser)**, các **LLM generators lấy cảm hứng từ XiYan-SQL**, một bộ thực thi xác định và cơ chế **abstention** nhằm tối ưu **Penalized Accuracy (PA)**.

> **Trạng thái hiện tại:** Chế độ **CPU / G1 rule-only** đã có kết quả đánh giá. Pipeline SQL theo **XiYan-SQL** (Schema Filter theo từng câu hỏi + PFKeyIdentifier, sinh SQL trên nhiều schema × nhiều định dạng + self-refine, candidate reorganization + selection model, dữ liệu multi-task/multi-format) đã được cài đặt; các phần **không cần LLM** đã đo trên train (§4.2). Phần dùng **Qwen3-14B** mới kiểm thử bằng mock LLM, **chưa có benchmark thật trên GPU**. Không nên hiểu điểm PA công bố dưới đây là kết quả của mô hình đã fine-tune.

## Nội dung

1. [Bài toán và mục tiêu](#1-bài-toán-và-mục-tiêu)
2. [Quick start — chạy trong vài bước](#2-quick-start--chạy-trong-vài-bước)
3. [Kiến trúc và quy trình xử lý](#3-kiến-trúc-và-quy-trình-xử-lý)
4. [Kết quả thực nghiệm](#4-kết-quả-thực-nghiệm)
5. [Cấu trúc repository](#5-cấu-trúc-repository)
6. [Huấn luyện và sử dụng LLM](#6-huấn-luyện-và-sử-dụng-llm)
7. [Đánh giá và kiểm soát độ tin cậy](#7-đánh-giá-và-kiểm-soát-độ-tin-cậy)
8. [Docker và submission](#8-docker-và-submission)
9. [Tạo `submission.jsonl` để nộp leaderboard](#9-tạo-submissionjsonl-để-nộp-leaderboard)
10. [Giới hạn và hướng phát triển](#10-giới-hạn-và-hướng-phát-triển)

## 1. Bài toán và mục tiêu

### DataMate giải quyết vấn đề gì?

Một cửa hàng có thể lưu bán hàng, nhập hàng, tồn kho, công nợ và khách hàng trong nhiều file có cấu trúc khác nhau. Người dùng không biết SQL vẫn muốn hỏi bằng ngôn ngữ tự nhiên, chẳng hạn:

- “Tuần rồi bán chạy nhất mặt hàng gì?”
- “Doanh thu từ đầu tháng đến hôm nay là bao nhiêu?”
- “Khách nào còn nợ nhiều nhất?”
- “Tỷ lệ đơn hủy tháng trước là bao nhiêu?”

Agent cần **hiểu câu hỏi, đọc đúng bảng, tuân thủ luật nghiệp vụ và tính kết quả từ dữ liệu**, không tự phỏng đoán con số.

**Đầu vào của mỗi câu hỏi** gồm `business_id`, nội dung `question` và `as_of_date` (ngày dùng để diễn giải “hôm nay”, “tuần trước”, “tháng này”...). Mỗi business có dữ liệu thô cùng `README.md`; mỗi split có `rulebook.md` và `calendar.csv`.

```json
{"id":"q_06de33fa21cc","business_id":"biz_118","question":"sp bán dc nhiều nhất tuần rồi là gì","as_of_date":"2026-03-04"}
```

**Đầu ra** là một bản ghi JSONL có kiểu đáp án chính xác theo đề bài:

```json
{"id":"q_06de33fa21cc","type":"string","answer":"Xi măng Hà Tiên PCB40 bao 50kg"}
```

Các kiểu hỗ trợ: `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable` và `abstain`.

**`unanswerable` khác `abstain`:**

- `unanswerable`: dữ liệu **không đủ để trả lời**; đây là một đáp án có thể được chấm đúng hoặc sai.
- `abstain`: agent **chủ động không trả lời** vì độ tin cậy chưa đủ; nhận 0 điểm.

### Thước đo: Penalized Accuracy

```text
PA = (số câu đúng − số câu sai) / tổng số câu
Đúng: +1     Sai: −1     Abstain: 0
```

Vì trả lời sai bị trừ điểm, mục tiêu **không phải trả lời càng nhiều câu càng tốt** mà là chọn những câu có khả năng đúng đủ cao. Ngưỡng lý thuyết để trả lời, nếu xác suất được hiệu chỉnh chuẩn, là `P(correct) > 0.5`. Repo hiện đặt ngưỡng mặc định `tau = 0.6` trong `datamate/agent.py`.

## 2. Quick start — chạy trong vài bước

### Yêu cầu

- Python **3.10+**.
- **Chỉ dùng G1 / CPU:** không cần tải LLM hay có GPU.
- **Dùng G2–G4 / GPU:** cần trọng số mô hình ở đường dẫn local và môi trường `vllm` hoặc `transformers` tương thích.

### Bước 1 — Clone và cài dependencies

```bash
git clone https://github.com/AIVIETNAM-AIO-khaisiubeo/datamate-xiyan-.git
cd datamate-xiyan-
python -m pip install -r requirements.txt
```

### Bước 2 — Đặt dữ liệu cuộc thi

Hai split **đã giải nén** được đặt như sau:

```text
data/
├── train/
│   ├── data/
│   │   ├── <business_id>/
│   │   │   ├── README.md
│   │   │   └── *.csv | *.xlsx | *.sqlite
│   │   └── ...
│   ├── questions.jsonl
│   ├── answers.jsonl
│   ├── rulebook.md
│   └── calendar.csv
└── public_test/
    ├── data/
    ├── questions.jsonl
    ├── rulebook.md
    └── calendar.csv
```

> Repo này đã kèm sẵn hai split trong `data/train` và `data/public_test` (clone xong là chạy được). Nếu dùng bản repo không có dữ liệu, giải nén dữ liệu của ban tổ chức theo đúng cấu trúc trên. `public_test` không có `answers.jsonl`. Kiểm tra lại quy định chia sẻ dữ liệu của cuộc thi trước khi để repo ở chế độ public.

### Bước 3 — Kiểm thử nhanh trên CPU

```bash
# Chạy 10 câu đầu để kiểm tra luồng xử lý và định dạng đầu ra
python -m datamate.predict --data data/train --out runs/smoke/submission.jsonl --llm none --limit 10

# Chạy toàn bộ train; CLI tự in PA khi có answers.jsonl
python -m datamate.predict --data data/train --out runs/train/submission.jsonl --llm none

# Tạo file nộp public test bằng G1 rule-only
python -m datamate.predict --data data/public_test --out runs/public/submission.jsonl --llm none
```

**File cần nộp:** `runs/public/submission.jsonl`. Log cho từng câu nằm trong thư mục `logs/` cạnh file đầu ra (mặc định).

**Lưu ý:** lệnh train ở trên là **chạy inference/evaluation trên train**, không cập nhật trọng số mô hình. Quy trình fine-tuning thật nằm ở [Mục 6](#6-huấn-luyện-và-sử-dụng-llm).

### Bước 4 — Chạy unit tests

```bash
python -m pip install pytest
python -m pytest -q tests/
```

Test mock-LLM ở `tests/test_agent_mock.py` tự bỏ qua khi không có dữ liệu train. Test này kiểm tra luồng tích hợp, **không đo chất lượng Qwen3-14B**.

## 3. Kiến trúc và quy trình xử lý

Hệ thống **không dùng LLM để tự tính tiền hay đếm số đơn**. LLM chủ yếu hỗ trợ hiểu yêu cầu và sinh truy vấn; phép tính được giao cho business engine hoặc DuckDB.

### 3.1. Ingestion & Schema Linking

**Code:** `datamate/ingest/`, `datamate/values.py`.

Đọc dữ liệu CSV/XLSX/SQLite không đồng nhất, xử lý header, sheet, dòng tổng, tên cột, đơn vị và các dạng ghi tiền/ngày. Thông tin từ README của từng shop, từ đồng nghĩa cột và mẫu giá trị được dùng để ánh xạ về **mô hình dữ liệu chuẩn `Biz`**.

Các bảng nghiệp vụ chuẩn bao gồm đơn hàng, dòng bán, danh mục hàng, nhập hàng, thanh toán, tồn kho và dữ liệu liên quan khi có sẵn. Phần `datamate/domains/` bổ sung xử lý cho sổ sửa xe, điện thoại/bảo hành và sổ nợ/trả góp.

### 3.2. Vietnamese NLU & Temporal Reasoning

**Code:** `datamate/textnorm.py`, `datamate/nlu/`.

- Chuẩn hóa tiếng Việt không dấu, Telex, teencode, viết tắt và biến thể vùng miền.
- Liên kết tên mặt hàng, nhóm hàng, khách hàng và thuộc tính trong câu hỏi.
- Diễn giải mốc thời gian dựa trên `as_of_date`; dùng lịch âm cho những khoảng Tết.
- Chuyển yêu cầu thành **Frame DSL**: intent, khoảng thời gian, bộ lọc, tiêu chí xếp hạng và kiểu trả lời.

### 3.3. Các generator

| Generator | Chức năng | Cách tạo kết quả |
|---|---|---|
| **G1 — Rule** | Phân tích trực tiếp câu hỏi bằng luật | Frame DSL → deterministic executor |
| **G2 — JSON Frame** | LLM trích intent/time/filter | JSON → câu hỏi chuẩn → G1 |
| **G3 — Rewrite** | LLM viết lại câu hỏi nhiễu | Câu viết lại → G1 |
| **G4 — XiYan SQL** (`G4_base`, `G4_cte`, `G4_alias`) | Text-to-SQL trên các view chuẩn hoá | Schema Filter → sinh SQL trên S1/S2/full × 3 định dạng → thực thi → self-refine → selection (§3.6) |
| **G_select** | Selection model chọn một SQL trong các ứng viên đã tái tổ chức | Một phiếu bầu thêm cho selector, không bỏ qua ngưỡng abstain |

**Code:** `datamate/nlu/rules.py`, `datamate/llm/generators.py`, `datamate/llm/sql_pipeline.py`, `datamate/select/selection_model.py`.  


### 3.4. Business Engine

**Code:** `datamate/engine/executor.py`, `datamate/engine/answer.py`, `datamate/rulebook.py`.

Các phép tính tuân theo quy tắc nghiệp vụ của split: loại đơn hủy/trả hàng theo luật, lượng bán quy về đơn vị cơ bản, doanh thu, công nợ, ranking, so sánh kỳ, VAT/phí giao hàng và quy tắc làm tròn. `rulebook.py` phát hiện những luật có vẻ khác cách hiểu mặc định để giảm độ tin cậy của intent bị ảnh hưởng.

### 3.5. Candidate Selection & Abstention

**Code:** `datamate/select/selector.py`, `datamate/nlu/consistency.py`, `datamate/nlu/coverage.py`.

Mỗi generator có thể đưa ra một đáp án ứng viên. Selector **gom nhóm theo giá trị đáp án đã thực thi**, kết hợp độ tin cậy theo generator/intent và trừ điểm khi các ứng viên mâu thuẫn. Các tín hiệu rủi ro gồm từ chưa giải thích được, câu hỏi và kiểu đáp án mâu thuẫn, tie ở ranh giới top-k hoặc luật nghiệp vụ thay đổi. Hệ thống trả lời khi điểm tin cậy qua ngưỡng, ngược lại chọn `abstain`. 

Với ứng viên SQL, selector còn **đọc nội dung SQL** (`datamate/select/sql_checks.py`): thiếu `is_valid`, cột ngày / khoảng thời gian sai, chặn trên dạng `<` bỏ mất ngày cuối, JOIN nhân bản dòng khi cộng cột cấp đơn, dùng tiền đã trả (`payments`, `total_due`) thay cho doanh thu, xếp hạng mặt hàng không lọc `kind = 'item'`, `LIMIT` khác top-k của câu hỏi, kết quả nhiều dòng khi cần một giá trị. Mỗi cờ hạ độ tin cậy của ứng viên. **Nhiều SQL của cùng một model không phải các phiếu độc lập:** selector gom theo *model family*, mỗi family chỉ có một phiếu trong mỗi nhóm đáp án, dù model đó sinh ở bao nhiêu schema hay định dạng.

### 3.6. Các function build từ XiYan-SQL 

| Thành phần trong paper | Trong repo | File |
|---|---|---|
| **M-Schema** (`【DB_ID】`, `# Table`, `(cột:KIỂU, mô tả, Primary Key, Examples)`, `【Foreign keys】`) | Cùng định dạng và API (`add_table`, `add_field`, `add_foreign_key`, `to_mschema(selected_columns)`) như `XGenerationLab/M-Schema`. Khoá **có cấu trúc**: PK (kể cả khoá ghép `sales_lines(order_id, line_no)`), FK, cột ngày, cột điều kiện nghiệp vụ. Mọi khai báo khoá đều **kiểm chứng trên dữ liệu** (PK duy nhất, không null; FK ≥ 95 % giá trị tồn tại), sai thì bỏ và ghi `issues` | `datamate/llm/mschema.py` |
| **Multi-path retrieval**: keyword `K = M_s(Q,E)`; `Score(k,c) = cos(V_{Q‖E}, V_tab(c)) · cos(V_k, V_c)`; value retrieval bằng edit distance + LSH | Keyword từ LLM (nếu có) ∪ keyword luật; cùng công thức Eq. 1 trên **TF-IDF n-gram ký tự** + bảng thuật ngữ nghiệp vụ tiếng Việt; value retrieval bằng chỉ mục n-gram (thay LSH) + edit distance, giữ cột chứa giá trị; thêm một đường **lexicon** (ánh xạ đồng nghĩa → cột, tức "synonym mappings" trong evidence của paper) | `datamate/llm/schema_filter.py` |
| **Algorithm 1** — chọn cột lặp `ps = 2`, `PFKeyIdentifier`, `S_i = ∪S_{k<i} ∪ S_slct ∪ P_i`, `S_rtrv ← S_rtrv \ (S_slct \ P_i)` | Như paper; `M_s` = Qwen3-14B (prompt chọn cột, gọi theo lô), không có LLM thì dùng bộ chọn heuristic. `PFKeyIdentifier` bổ sung đủ khoá ghép và khoá trên đường nối (BFS qua FK đã kiểm chứng). **Luôn giữ cột điều kiện nghiệp vụ** (`is_valid`, `kind`, `status`, `cancelled`) và cột ngày của mọi bảng dùng tới. Schema đầy đủ giữ làm **baseline** và **dự phòng** khi schema lọc không hợp lệ | `datamate/llm/schema_filter.py` |
| **Evidence E** (giải thích schema, đồng nghĩa, luật nghiệp vụ) | Ngày hôm nay + các dòng rulebook liên quan (truy hồi TF-IDF) + quy ước trong README của shop | `datamate/llm/evidence.py` |
| **Algorithm 2** — `ps` schema × `pm` generator, thực thi, self-refine 1 lần khi lỗi / kết quả bất thường | S1, S2, full × 3 định dạng (`base`, `cte` = biến thể cấu trúc, `alias` = biến thể phong cách, như Fig. 3) với few-shot SQL theo định dạng; self-refine dùng template của XiYan; loại SQL trùng (de-formalize), không chạy lại; mỗi ứng viên lưu SQL, schema, generator, model family, kết quả/lỗi, cờ kiểm tra, thời gian, số token; DuckDB chỉ-đọc, có timeout | `datamate/llm/sql_pipeline.py` |
| **Algorithm 3** — de-formalize, gom nhóm theo kết quả, sắp xếp liên nhóm/nội nhóm, đồng thuận ≥ ⌈\|L\|/2⌉ thì đưa hết, không thì mỗi nhóm một SQL ngắn nhất, `M_c(Q, S_un, E, L')` | Như paper; `M_c` dùng template chọn ứng viên của XiYan; lựa chọn của nó là **một phiếu** trong selector có hiệu chuẩn, không thay ngưỡng abstain (khác paper vì metric phạt câu sai) | `datamate/select/selection_model.py`, `datamate/select/selector.py` |
| **Multi-task fine-tuning** (Text-to-SQL, question inference, evidence inference, self-refine) + **multi-format** + dữ liệu chọn listwise với near-miss | Sinh từ SQL đã kiểm chứng trên train: 7 loại bản ghi, negative "near-miss" bằng **controlled modification** (`sql_mutations.py`); xuất cả định dạng `alibaba/XiYan-SQLTraining` | `training/build_sql_sft.py`, `scripts/build_sql_silver.py` |

**Điều chỉnh cho cuộc thi**:

- **≤ 15B tham số cho *mọi* model, offline:** không có model embedding riêng (Qwen3-14B đã chiếm 14,8B), nên similarity dùng TF-IDF n-gram ký tự. `M_s`, generator và `M_c` cùng là một Qwen3-14B. Paper dùng GPT-4o, 4 model Qwen2.5-Coder-32B fine-tune và một selection model 7B, tức nhiều model riêng biệt.
- **Đa dạng generator:** hiện là *một* model với 3 định dạng SQL qua prompt. Đây chưa phải các generator fine-tune riêng như paper, và chúng **không** được tính là phiếu độc lập. Muốn đi xa hơn thì train **một** model đa nhiệm, đa định dạng (định dạng chọn bằng chỉ dẫn trong prompt) để vẫn 14,8B. Adapter LoRA riêng cho từng định dạng phải được cộng vào tổng tham số (`scripts/count_params.py`).
- **Metric PA phạt câu sai:** paper luôn trả về một SQL; ở đây lựa chọn cuối là bỏ phiếu có hiệu chuẩn rồi mới qua ngưỡng τ, vì vậy có thể `abstain`.
- **G4 chạy cho câu G1/G2/G3 chưa chắc** (`DATAMATE_G4_SCOPE=unresolved`, mặc định) để vừa ngân sách 6 giờ; đặt `all` để G4 chạy mọi câu.

## 4. Kết quả thực nghiệm

Các số dưới đây lấy từ báo cáo hiện có tại [`docs/RESULTS.md`](docs/RESULTS.md). Đây là kết quả **G1 + coverage + selector trên CPU**, không phải kết quả LLM.

| Split / phiên bản | Đáp án giá trị | `unanswerable` | `abstain` | PA |
|---|---:|---:|---:|---:|
| Public test, submission 1 (1.000 câu) | 850 | 21 | 129 | **0.561** (điểm BTC) |
| Public test, phiên bản 2 | 920 | 22 | 58 | **0,912 (điểm BTC)** |


Chi tiết breakdown theo intent, thay đổi giữa hai bản và phân tích lỗi nằm trong [báo cáo kết quả](docs/RESULTS.md). File kết quả CPU được lưu tại [`submissions/public_test_rule_only.jsonl`](submissions/public_test_rule_only.jsonl).

## 5. Cấu trúc repository

```text
datamate-xiyan-/
├── datamate/                   # Runtime inference
│   ├── agent.py                # Orchestrator: G1–G4, selector, logs
│   ├── predict.py              # CLI tạo submission.jsonl
│   ├── ingest/                 # CSV/Excel/SQLite → canonical Biz
│   ├── nlu/                    # Rule parser, Frame, time, linking, guards
│   ├── domains/                # Sửa xe, điện thoại, sổ nợ/trả góp
│   ├── engine/                 # Business execution & answer formatting
│   ├── llm/                    # Backends, prompts (template XiYan), G2/G3, M-Schema + khoá
│   │   ├── mschema.py          # View chuẩn hoá, MSchema (PK/FK/khoá ghép đã kiểm chứng)
│   │   ├── schema_filter.py    # Multi-path retrieval, Algorithm 1, PFKeyIdentifier
│   │   ├── evidence.py         # Evidence E: dòng rulebook + README liên quan
│   │   ├── sql_pipeline.py     # Algorithm 2: S1/S2/full × định dạng, thực thi, self-refine, provenance
│   │   ├── frame2sql.py        # Frame → SQL (silver SQL, 3 định dạng)
│   │   └── sql_mutations.py    # Near-miss negatives (controlled modification)
│   ├── select/                 # Voting, reliability, abstention
│   │   ├── sql_checks.py       # Kiểm tra nội dung SQL (is_valid, ngày, fan-out, doanh thu…)
│   │   └── selection_model.py  # Algorithm 3: reorganization + selection model
│   ├── rulebook.py             # Split-specific business rules
│   ├── scorer.py               # Local Penalized Accuracy scorer
│   ├── textnorm.py             # Chuẩn hóa tiếng Việt
│   └── augment.py              # Tạo nhiễu câu hỏi train
├── training/
│   ├── build_sft_data.py       # Sinh dữ liệu rewrite/frame/answerability
│   ├── build_sql_sft.py        # Dữ liệu multi-task/multi-format cho SQL generator + selection model
│   ├── sft_lora.py             # LoRA supervised fine-tuning
│   └── grpo_abstain.py         # RL với reward +1 / −1 / 0
├── scripts/
│   ├── build_artifacts.py      # Tạo train-derived vocabulary/few-shot bank
│   ├── eval_rules.py           # Đánh giá G1 theo intent
│   ├── calibrate.py            # Hiệu chuẩn reliability theo intent
│   ├── build_sql_silver.py     # SQL đã kiểm chứng bằng thực thi (train)
│   ├── eval_schema_filter.py   # Recall cột của S_rtrv / S1 / S2 / full
│   ├── eval_sql_checks.py      # Báo động nhầm / tỉ lệ phát hiện của SQL checks
│   ├── validate.py             # Business-level K-fold: PA, ECE, Brier, quét τ, log lỗi
│   ├── check_submission.py     # Kiểm tra định dạng submission.jsonl trước khi nộp
│   ├── count_params.py         # Tổng tham số mọi model dùng khi suy luận (≤ 15B)
│   └── ...                     # Công cụ kiểm tra và debug
├── artifacts/                  # Vocab, example bank, reliability
├── tests/                      # Unit tests và mock-LLM integration test
├── docs/RESULTS.md             # Số liệu và phân tích thực nghiệm
├── submissions/                # File submission CPU hiện có
├── data/                       # train/ và public_test/ (đã kèm trong repo)
├── requirements.txt
├── requirements-train.txt
├── Dockerfile
└── predict.sh                  # Entry point của Docker
```


## 6. Huấn luyện và sử dụng LLM

### 6.1. Backbone và vai trò

| Model | Tham số theo model card | Vai trò dự kiến |
|---|---:|---|
| `Qwen/Qwen3-14B` | khoảng **14.8B** | LLM dùng chung cho G2, G3, G4 |

Model không bắt buộc ở chế độ CPU. Code hỗ trợ `none`, `vllm` và `hf` qua `datamate/llm/backend.py`.

**Giới hạn cuộc thi:** tổng tham số của **mọi model dùng tại inference** phải không vượt **15B** (không chỉ tham số đang được load vào GPU). Kiểm tra tổng sau khi bổ sung adapter hoặc model phụ; khai báo weights/license trong bản nộp.

### 6.2. Tạo artifacts từ train

```bash
python scripts/build_artifacts.py data/train
python scripts/eval_rules.py data/train
python scripts/calibrate.py eval_rules_out.jsonl artifacts/reliability.json
```

Những bước này tạo hoặc cập nhật vocabulary, few-shot example bank và reliability theo intent. `eval_rules.py` ghi `eval_rules_out.jsonl` ở thư mục làm việc hiện hành.

### 6.3. Tạo dữ liệu SFT

```bash
python training/build_sft_data.py \
  --split data/train \
  --out outputs/sft.jsonl \
  --n-aug 3
```

Mỗi ví dụ huấn luyện được tạo từ frame G1 **đã khớp đáp án train**; mục tiêu SFT bao gồm **rewrite**, **JSON frame** và **answerability**. Dữ liệu train có thể được augment bằng các biến thể câu hỏi tiếng Việt.

### 6.3b. Dữ liệu multi-task / multi-format cho SQL generator và selection model (XiYan §III-C/D)

```bash
python scripts/build_sql_silver.py data/train                         # artifacts/sql_silver.jsonl
python training/build_sql_sft.py --split data/train --out outputs/sql_sft --n-aug 2
```

Từ 1.671 câu train có SQL đã kiểm chứng (với `--n-aug 1`: 20.379 bản ghi), schema trong prompt là schema mà Schema Filter cho ra lúc suy luận:

| Task | Input → Output | Paper |
|---|---|---|
| `nl2sql_base` / `nl2sql_cte` / `nl2sql_alias` | (Q, E, S) → SQL theo từng định dạng | Fig. 2(a), Fig. 3 |
| `question_inference` | (SQL, E, S) → Q | Fig. 2(b) |
| `evidence_inference` | (Q, SQL, các luật ứng viên) → luật SQL đã dùng | Fig. 2(c) |
| `self_refine` | (Q, E, S, SQL near-miss + kết quả thực thi) → SQL đúng | Fig. 2(d) |
| `select` | (Q, E, S, ứng viên đã de-formalize, thứ tự ngẫu nhiên) → `{"choice": i}` | §III-D |

Hai file đầu ra: `sql_sft.jsonl` cho `training/sft_lora.py`, và `xiyan_sql_sft.json` theo định dạng `conversations` của `alibaba/XiYan-SQLTraining`. **Công thức trong giới hạn 15B:** train *một* model trên mọi task và định dạng, định dạng được chọn bằng chỉ dẫn trong prompt. Sau khi merge, model vẫn có 14,8B tham số. Nếu muốn dùng nhiều adapter LoRA cùng lúc, kiểm tra tổng:

```bash
python scripts/count_params.py /models/llm outputs/lora_cte outputs/lora_alias
```

### 6.4. SFT LoRA (GPU)

```bash
python -m pip install -r requirements-train.txt

python training/sft_lora.py \
  --model /models/llm \
  --data outputs/sft.jsonl \
  --out outputs/lora_sft
```

**Quan trọng:** script `training/sft_lora.py` hiện load backbone ở **BF16** và áp dụng **LoRA**; **chưa cài QLoRA 4-bit/NF4**. Vì vậy không nên mặc định script hiện tại chạy được trên GPU Colab 16–24 GB. Cần kiểm tra VRAM và khả năng tương thích phiên bản `transformers/trl/peft` khi train thực tế.

Để gộp adapter vào backbone:

```bash
python training/sft_lora.py \
  --merge \
  --model /models/llm \
  --adapter outputs/lora_sft \
  --out outputs/merged
```

### 6.5. GRPO với reward theo PA (thử nghiệm)

```bash
python training/grpo_abstain.py \
  --model outputs/merged \
  --split data/train \
  --out outputs/grpo
```

Thiết kế reward có điểm **+1 khi đúng, −1 khi sai, 0 khi abstain**, cộng phần thưởng/penalty định dạng. Đây là **code huấn luyện thử nghiệm**, chưa có xác nhận train thành công hay cải thiện PA.

### 6.6. Inference với LLM

Khi đã có **model local** và GPU tương thích:

```bash
# vLLM
python -m datamate.predict \
  --data data/public_test \
  --out runs/public_llm/submission.jsonl \
  --llm vllm \
  --model /models/llm

# Transformers fallback
python -m datamate.predict \
  --data data/public_test \
  --out runs/public_hf/submission.jsonl \
  --llm hf \
  --model /models/llm
```

Nếu model không load được, backend có cơ chế **fallback sang G1 rule-only**. Kiểm tra log trước khi diễn giải kết quả là của LLM. Mã inference **không tự nạp LoRA adapter riêng biệt** từ thư mục `outputs/lora_sft`; dùng model đã merge hoặc tích hợp adapter vào backend trước khi đánh giá.

## 7. Đánh giá và kiểm soát độ tin cậy

### Đánh giá G1 theo intent

```bash
python scripts/eval_rules.py data/train
```

Chương trình ghi các trường như câu hỏi, intent, frame, đáp án dự đoán, reference và score để phân tích từng loại lỗi. Có thể dùng `scripts/review_run.py`, `scripts/check_consistency.py` và `scripts/check_guard.py` để chẩn đoán thêm.

### Validation theo business và hiệu chuẩn confidence

```bash
python scripts/validate.py runs/train/logs data/train/answers.jsonl --folds 5
# sau một lần chạy GPU trên train: hiệu chuẩn lại mọi generator (G1–G4, G_select) theo business-level fold
python scripts/validate.py runs/train_llm/logs data/train/answers.jsonl --folds 5 --write-reliability artifacts/reliability.json
```

Mỗi shop chỉ nằm trong một fold. Độ tin cậy của từng generator × intent được fit trên K−1 fold, rồi selector chạy lại trên fold còn lại. Script in PA, coverage, precision khi trả lời, **Brier**, **ECE**, reliability diagram và kết quả quét τ; mọi câu trả lời sai được ghi vào `runs/validate/errors.jsonl`. Log của mỗi câu (`logs/<id>.json`) có: ứng viên theo từng generator, `schema_filter` (keyword, cột đã truy hồi, giá trị khớp, S1/S2), `evidence`, `sql_candidates` (SQL, schema, định dạng, model family, kết quả hoặc lỗi, cờ kiểm tra, thời gian, token, các nguồn trùng) và `selection` (mode, số ứng viên đưa ra, SQL được chọn).

### Các lớp kiểm tra trước khi trả lời

- **Business consistency:** đơn hợp lệ, đơn vị, thời gian, điều kiện lọc và luật của split.
- **Semantic coverage:** câu hỏi có khái niệm chưa được parser giải thích hay không.
- **Answer-type consistency:** ví dụ câu hỏi “ai/mặt hàng nào” nhưng trả về số; hỏi “bao nhiêu %” nhưng trả tiền.
- **Selection confidence:** ứng viên đồng thuận hay mâu thuẫn; có tie ở biên ranking; có dấu hiệu giá trị không hợp lý.
- **Selective prediction:** chọn `abstain` nếu độ tin cậy dưới ngưỡng.

> `datamate/scorer.py` là **scorer nội bộ**. Trước khi so sánh sát với leaderboard, cần kiểm toán sự phù hợp với quy tắc chấm chính thức, đặc biệt **exact numeric matching, string normalization và table column matching**. Điểm chính thức do ban tổ chức xác nhận.

**Nguyên tắc dữ liệu:** chỉ dùng nhãn của **train** để xây artifacts, học mô hình và hiệu chỉnh. Không dùng `public_test` hay `private_test` cho training hoặc thu thập đáp án tham chiếu. Để ước lượng generalization, nên giữ riêng một validation split theo **business** trước khi tối ưu tiếp.

## 8. Docker và submission

Theo thiết kế cuộc thi, image phải chạy **offline**, đọc test split từ `/data` (read-only) và ghi kết quả vào `/result`.

```bash
docker build -t datamate-xiyan:final .

docker run --rm --gpus all --network none \
  -v /absolute/path/to/test_split:/data:ro \
  -v "$(pwd)/result:/result" \
  datamate-xiyan:final
```

- `predict.sh` là entry point, chạy `datamate.predict`.
- `/result/submission.jsonl` là file nộp chính.
- `/result/logs/<question_id>.json` chứa thông tin ứng viên, đáp án chọn và thời gian.
- `DATAMATE_LLM=none` cho phép chạy không dùng LLM; `DATAMATE_TIME_BUDGET` điều chỉnh ngân sách thời gian.
- `Dockerfile` tải Qwen3-14B **lúc build image** để inference có thể chạy offline. Model weights **không được commit trong repo**.

**Trước khi nộp Docker cuối cùng:** kiểm tra base image theo guideline ban tổ chức (CUDA/Blackwell), model/license và tổng tham số, logs, giới hạn **6 giờ**, tính tái lập và chạy thử với `--network none`. Việc code có Dockerfile **không đồng nghĩa** image đã được kiểm chứng thành công trên 2 × RTX 5090.

## 9. Tạo `submission.jsonl` để nộp leaderboard

Lệnh viết cho bash (Linux, macOS, Git Bash); với PowerShell, thay `export X=y` bằng `$env:X = "y"`.

**Bước 1 — Cài đặt và kiểm tra môi trường**

```bash
git clone https://github.com/AIVIETNAM-AIO-khaisiubeo/datamate-xiyan-.git
cd datamate-xiyan-
python -m pip install -r requirements.txt pytest
export DATAMATE_CACHE=.cache            # cache dữ liệu đã đọc (nhanh hơn từ lần chạy thứ 2)
python -m pytest -q tests/              # mọi test phải pass
```

**Bước 2 — Kiểm tra trên train trước (có đáp án)**

```bash
python -m datamate.predict --data data/train --out runs/train/submission.jsonl --llm none
python scripts/check_submission.py runs/train/submission.jsonl data/train/questions.jsonl data/train/answers.jsonl
```

Dòng `[datamate] PA=…` và `local PA` phải bằng **1.0000** ở chế độ CPU. Nếu thấp hơn, môi trường hoặc code đã thay đổi; xem `runs/train/logs/` và `python scripts/validate.py runs/train/logs data/train/answers.jsonl`.

**Bước 3 — Sinh file cho public test**

Chọn một trong ba cách:

```bash
# (a) CPU, rule-only: nhanh (~1–5 phút), tất định — cách đã tạo bản nộp hiện tại
python -m datamate.predict --data data/public_test --out runs/public/submission.jsonl --llm none

# (b) GPU, đầy đủ G1–G4 + selection model (cần model local, ví dụ Qwen3-14B tại /models/llm)
python -m datamate.predict --data data/public_test --out runs/public_llm/submission.jsonl --llm vllm --model /models/llm
# tuỳ chọn: chạy G4 cho mọi câu thay vì chỉ câu chưa chắc
export DATAMATE_G4_SCOPE=all

# (c) Docker, đúng môi trường chấm (đọc /data, ghi /result)
docker build -t datamate-xiyan:final .
docker run --rm --gpus all --network none -v "$(pwd)/data/public_test:/data:ro" -v "$(pwd)/result:/result" datamate-xiyan:final
# file nộp: result/submission.jsonl
```

**Bước 4 — Kiểm tra file trước khi tải lên**

```bash
python scripts/check_submission.py runs/public/submission.jsonl data/public_test/questions.jsonl
```

Script kiểm tra: mỗi dòng là một JSON; đủ đúng 1.000 `id` của `questions.jsonl`, không trùng, không thừa; `type` hợp lệ; `answer` đúng kiểu (`number`/`percent` là số, `date` dạng `YYYY-MM-DD` hoặc `YYYY-MM`, `table` có `columns`/`rows`…). Phải in ra `OK: format valid`. Có thể so với bản đã nộp để biết bao nhiêu câu thay đổi:

```bash
python -c "import json;a=[json.loads(l) for l in open('submissions/public_test_rule_only.jsonl',encoding='utf-8')];b=[json.loads(l) for l in open('runs/public/submission.jsonl',encoding='utf-8')];print(sum(x!=y for x,y in zip(a,b)),'answers changed')"
```

**Bước 5 — Nộp**

- Tải `runs/public/submission.jsonl` (hoặc `result/submission.jsonl` nếu dùng Docker) lên trang nộp bài của Zalo AI Challenge, đúng track **SLM (≤ 15B)**. Chế độ CPU không dùng model nào; chế độ GPU dùng Qwen3-14B (14,8B tham số).
- Giữ lại thư mục `logs/` cạnh file nộp: log từng câu theo yêu cầu thể lệ.
- Lưu bản đã nộp vào `submissions/` và ghi điểm leaderboard vào `docs/RESULTS.md` để so sánh các lần nộp.

**Lưu ý**
- Chỉ dùng nhãn của **train** để fit hay hiệu chuẩn (`artifacts/`). Không dùng public/private test để huấn luyện.
- Với chế độ GPU, chạy `python scripts/count_params.py /models/llm [adapter...]` để chắc tổng tham số ≤ 15B. Kiểm tra log để chắc LLM đã thực sự được nạp: nếu không nạp được model, backend tự chuyển về rule-only.
- `DATAMATE_TAU` (mặc định 0.6) là ngưỡng abstain. Tăng τ thì trả lời ít câu hơn nhưng chắc hơn. Chỉ chỉnh τ dựa trên `scripts/validate.py` trên train, không dựa trên điểm public.

## 10. Giới hạn và hướng phát triển

1. **Generalization:** train đạt 1.000 nhưng đó là kết quả trong mẫu; public bản 1 đạt 0.561. Các cách diễn đạt mới, bảng có schema lạ và business chưa gặp vẫn là thách thức.
2. **LLM chưa được benchmark:** G2/G3 và pipeline XiYan của G4 (keyword/chọn cột bằng LLM, sinh SQL, self-refine, selection model) mới chạy với mock LLM. Cần chạy inference thật với Qwen3-14B, rồi ablation trên validation theo business: `G1` → `+G2/G3` → `+G4 full schema` → `+S1/S2` → `+3 định dạng` → `+selection model`. Chỉ giữ thành phần nào làm PA tăng: thêm thành phần không tự động làm hệ thống chính xác hơn.
3. **Đa dạng generator:** 3 định dạng SQL hiện là prompt của một model, nên chưa tương đương các generator fine-tune riêng của paper. Bước tiếp: train một model đa nhiệm, đa định dạng bằng `training/build_sql_sft.py` (vẫn 14,8B), rồi mới tính đến LoRA riêng cho từng định dạng nếu ngân sách tham số cho phép.
4. **Training GPU còn thử nghiệm:** script SFT hiện là LoRA BF16, không phải QLoRA; GRPO chưa có số liệu xác nhận.
5. **Calibration:** reliability của G4 và G_select hiện là giá trị mặc định (0,55–0,60). Cần hiệu chuẩn bằng `scripts/validate.py --write-reliability` trên log GPU của train. Hiệu chuẩn của G1 lấy từ train và có thể quá lạc quan.
6. **Schema Filter không LLM** dùng TF-IDF + bảng thuật ngữ viết khi đọc train, nên với shop có cột lạ, recall có thể thấp hơn §4.2. Schema đầy đủ luôn được giữ làm baseline.
7. **Rulebook drift và độ an toàn SQL:** thay đổi luật trong split khác có thể khiến engine không còn đúng. G4 chỉ chạy trên các view đã chuẩn hoá, DuckDB chỉ-đọc, có timeout cho mỗi truy vấn.

### Nền tảng nghiên cứu

Thiết kế của repo **lấy cảm hứng** từ:

- **XiYan-SQL: A Novel Multi-Generator Framework for Text-to-SQL** — schema filtering, multi-generator và lựa chọn ứng viên theo kết quả thực thi.
- **When Silence is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?** — học và ra quyết định từ chối trả lời để hạn chế lỗi.


---

**Documentation:** [Detailed results](docs/RESULTS.md) · [Training code](training/) · [Tests](tests/) · [Docker entry point](predict.sh)
