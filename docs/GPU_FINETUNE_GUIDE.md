# Chạy đầy đủ G1–G4 với Qwen3-14B và fine-tune — hướng dẫn từng bước

Tài liệu này dành cho người đã clone repo, có `data/train` và `data/public_test`, và muốn:

1. tạo `submission.jsonl` cho `public_test` bằng **toàn bộ hệ thống** (G1 + G2 + G3 + G4 + selection model, dùng
   Qwen3-14B), không chỉ bộ luật G1;
2. fine-tune Qwen3-14B cho các nhiệm vụ của agent rồi dùng model đó để nộp bài;
3. vẫn tuân thủ luật Zalo AI Challenge 2026.

> Dùng **Google Colab Pro**: có notebook làm sẵn mọi bước của Phần F, kèm lưu adapter về GitHub. Xem
> [`COLAB_GUIDE.md`](COLAB_GUIDE.md).

Mục lục:
- [Phần A — Khi bật LLM, hệ thống làm gì](#phần-a--khi-bật-llm-hệ-thống-làm-gì)
- [Phần B — Kiểm tra tuân thủ luật](#phần-b--kiểm-tra-tuân-thủ-luật)
- [Phần C — Phần cứng](#phần-c--phần-cứng)
- [Phần D — Tạo submission bằng toàn bộ hệ thống (chưa fine-tune)](#phần-d--tạo-submission-bằng-toàn-bộ-hệ-thống-chưa-fine-tune)
- [Phần E — Fine-tune những gì và tại sao](#phần-e--fine-tune-những-gì-và-tại-sao)
- [Phần F — Fine-tune từng bước](#phần-f--fine-tune-từng-bước)
- [Phần G — Đóng gói Docker và nộp](#phần-g--đóng-gói-docker-và-nộp)
- [Phần H — Sự cố thường gặp](#phần-h--sự-cố-thường-gặp)

---

## Phần A — Khi bật LLM, hệ thống làm gì

Lệnh `python -m datamate.predict ... --llm vllm --model <thư mục model>` chạy cho từng shop:

| Bước | Thành phần | Chạy cho câu nào | Model |
|---|---|---|---|
| 1 | **G1** — bộ luật → Frame → bộ thực thi | mọi câu | không |
| 2 | **G3** — viết lại câu hỏi lộn xộn thành câu chuẩn → G1 | mọi câu | Qwen3-14B |
| 3 | **G2** — câu hỏi → JSON frame → câu chuẩn → G1 | mọi câu | Qwen3-14B |
| 4 | **G4** — Schema Filter (keyword + chọn cột bằng LLM) → SQL trên S1, S2, full × 3 định dạng → thực thi → self-refine → kiểm tra SQL | câu mà G1–G3 **chưa chắc** (mặc định), hoặc mọi câu (`--g4-scope all`) | Qwen3-14B |
| 5 | **G_select** — selection model chọn 1 SQL trong các ứng viên đã tái tổ chức | câu G4 có nhiều kết quả khác nhau | Qwen3-14B |
| 6 | **Selector** — gom nhóm theo đáp án, bỏ phiếu có hiệu chuẩn, mỗi model family 1 phiếu, ngưỡng τ | mọi câu | không |

Mọi bước LLM dùng **cùng một model Qwen3-14B** (14,8B tham số), nên tổng tham số vẫn là 14,8B và bài thuộc
track **SLM (≤ 15B)**. Phép tính (cộng tiền, đếm đơn) luôn do bộ thực thi hoặc DuckDB làm; LLM không tự tính số.

Nếu model không nạp được (sai đường dẫn, thiếu GPU), backend tự **lùi về rule-only**. Khi đó stderr in
`-> rule-only mode` và log không có ứng viên `G2_frame` / `G3_rewrite` / `G4_*`. Luôn kiểm tra điều này trước
khi tin rằng kết quả là của LLM.

## Phần B — Kiểm tra tuân thủ luật

| Luật | Cách repo đáp ứng | Bạn cần làm |
|---|---|---|
| Tổng tham số mọi model ≤ 15B | Chỉ một model Qwen3-14B (14,8B). Model đã fine-tune và merge vẫn 14,8B; bản lượng tử hoá (FP8/AWQ) của cùng model cũng vẫn tính 14,8B | `python scripts/count_params.py <thư mục model> [adapter...]` |
| Không dùng API bên thứ 3 khi suy luận | Không có lời gọi mạng; Docker chạy với `--network none`, `HF_HUB_OFFLINE=1` | Chạy thử Docker với `--network none` |
| Model mã nguồn mở, có license | Qwen3-14B, Apache-2.0 | Khai báo model, license và nguồn trong bản nộp; đối chiếu ngày phát hành với quy định về model được phép (nếu có) |
| Không huấn luyện trên public/private test | Mọi script fine-tune và hiệu chuẩn chỉ đọc `data/train` (có `answers.jsonl`) | Không đưa `public_test` vào bất kỳ lệnh `training/*` hay `--write-reliability` nào |
| Không hard-code theo `id` / `business_id` | Không có logic theo id | Không sửa đáp án thủ công trong `submission.jsonl` |
| Chạy xong trong 6 giờ trên 2 × RTX 5090 | Ngân sách thời gian `DATAMATE_TIME_BUDGET` (mặc định 5,5 h): khi gần hết giờ, các bước LLM tự dừng và câu còn lại chỉ dùng G1 | Đo thời gian chạy Docker trên public; private có thể nhiều câu hơn |
| Tất định, có log | Nhiệt độ 0, seed cố định; log mỗi câu ở `logs/<id>.json` | Giữ thư mục `logs/` |

## Phần C — Phần cứng

Qwen3-14B ở BF16 nặng khoảng 30 GB, chưa tính KV cache. Không chạy được trên CPU.

**Suy luận (vLLM)**

| GPU | Cách chạy | Tham số |
|---|---|---|
| 2 × RTX 5090 32 GB (môi trường chấm) | BF16, chia 2 GPU | `--tp 2` (mặc định) |
| 1 × A100 / H100 80 GB | BF16 | `--tp 1` |
| 1 × 40–48 GB (A100 40GB, A6000, L40S) | FP8 | `--tp 1 --quant fp8` |
| 1 × 24–32 GB (RTX 4090 / 5090, L4) | FP8, giảm context | `--tp 1 --quant fp8 --max-len 12288 --gpu-util 0.92` |
| Colab T4 16 GB | Không khuyến nghị (không có FP8; 14B quá nặng) | — |

FP8 là lượng tử hoá của **cùng** model nên vẫn 14,8B tham số, nhưng điểm có thể lệch nhẹ so với BF16. Bản nộp
cuối nên chạy đúng như cách sẽ chạy trong Docker (2 × 5090, BF16, `--tp 2`).

**Fine-tune**

| GPU | Cách | Tham số |
|---|---|---|
| 1 × A100 / H100 80 GB | LoRA BF16 | mặc định |
| 1 × 24–48 GB (RTX 4090 / 5090, A6000, L40S) | QLoRA 4-bit | `--qlora` |

Có thể thuê GPU theo giờ (RunPod, Vast.ai, Lambda, Colab Pro+ A100). Cần ổ đĩa trống khoảng **120 GB**: model
gốc khoảng 30 GB, bản merge khoảng 30 GB, cùng checkpoint và cache.

## Phần D — Tạo submission bằng toàn bộ hệ thống (chưa fine-tune)

### D1. Cài đặt (Linux + driver NVIDIA; RTX 5090 cần CUDA ≥ 12.8)

```bash
git clone https://github.com/AIVIETNAM-AIO-khaisiubeo/datamate-xiyan-.git
cd datamate-xiyan-
python -m venv .venv && source .venv/bin/activate
pip install "vllm==0.12.0"                   # kéo theo PyTorch 2.9.0 có CUDA; cùng phiên bản với Dockerfile
pip install -r requirements.txt pytest "huggingface_hub[cli]"
export DATAMATE_CACHE=.cache
python -m pytest -q tests/                  # mọi test pass
```

### D2. Tải model (một lần, cần mạng, khoảng 30 GB)

```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir /models/qwen3-14b --exclude "*.pth" "original/*"
python scripts/count_params.py /models/qwen3-14b        # khoảng 14.8B, dưới giới hạn 15B
```

### D3. Chạy thử nhỏ để chắc LLM thực sự chạy

```bash
python -m datamate.predict --data data/train --only biz_101 --limit 20 \
  --llm vllm --model /models/qwen3-14b --tp 2 --g4-scope all --out runs/smoke_llm/submission.jsonl
python -c "import json,glob;c=[x['gen'] for f in glob.glob('runs/smoke_llm/logs/*.json') for x in json.load(open(f,encoding='utf-8'))['candidates']];import collections;print(collections.Counter(c))"
```

Kết quả phải có `G1_rule`, `G3_rewrite`, `G2_frame` và `G4_base` / `G4_cte` / `G4_alias`. Mở một file log để
xem `schema_filter`, `sql_candidates` và `selection`.

### D4. Hiệu chuẩn độ tin cậy của G2–G4 trên train (nên làm, chỉ dùng nhãn train)

Selector cần biết mỗi generator đúng bao nhiêu. Hiện G2/G3/G4/G_select đang dùng giá trị mặc định (0,55–0,80),
chưa đo. Chạy toàn bộ train với LLM rồi fit lại:

```bash
python -m datamate.predict --data data/train --llm vllm --model /models/qwen3-14b --tp 2 \
  --out runs/train_llm/submission.jsonl
python scripts/validate.py runs/train_llm/logs data/train/answers.jsonl --folds 5 \
  --write-reliability artifacts/reliability.json
```

- Khi split có `answers.jsonl` (train), agent **tự loại ví dụ few-shot của chính shop đang hỏi**
  (leave-one-business-out). Nhờ vậy câu train không "thấy" đáp án của chính nó trong prompt.
- `--write-reliability` chỉ ghi các generator LLM. Độ tin cậy của G1 giữ nguyên, trừ khi thêm `--include-g1`.
- `validate.py` in PA, coverage, Brier, ECE và quét τ. Chỉ chọn τ (`DATAMATE_TAU`) theo bảng này, **không** theo điểm
  public.

### D5. (Nên làm) Ablation: giữ thành phần nào?

Chạy trên một fold train (khoảng 6 shop) với từng cấu hình, rồi so PA:

```bash
for G in "" "g3" "g2,g3" "g2,g3,g4"; do
  python -m datamate.predict --data data/train --fold 0 --llm vllm --model /models/qwen3-14b \
    --generators "$G" --out "runs/abl_${G:-g1}/submission.jsonl"
done
```

`--generators ""` nghĩa là chỉ G1. Có thể thử thêm `--g4-scope all` hoặc `--tau 0.7`. Nếu thêm một thành phần
không làm PA tăng thì bỏ nó: nhiều thành phần hơn không tự động tốt hơn. Kết quả trên train vẫn có phần lạc quan,
vì G1 được viết trên train.

### D6. Sinh file nộp cho public test

```bash
python -m datamate.predict --data data/public_test --llm vllm --model /models/qwen3-14b --tp 2 \
  --out runs/public_llm/submission.jsonl          # thêm --generators / --g4-scope / --tau theo kết quả D5
python scripts/check_submission.py runs/public_llm/submission.jsonl data/public_test/questions.jsonl
```

Phải in `OK: format valid`. So với bản rule-only để xem LLM đã đổi bao nhiêu câu:

```bash
python -c "import json;a=[json.loads(l) for l in open('submissions/public_test_rule_only.jsonl',encoding='utf-8')];b=[json.loads(l) for l in open('runs/public_llm/submission.jsonl',encoding='utf-8')];print('changed',sum(x!=y for x,y in zip(a,b)),'| abstain',sum(x['type']=='abstain' for x in b))"
```

Tải `runs/public_llm/submission.jsonl` lên leaderboard, track **SLM**. Lưu bản này vào
`submissions/public_test_llm.jsonl`, ghi điểm vào `docs/RESULTS.md`.

## Phần E — Fine-tune những gì và tại sao

### Fine-tune là gì trong project này

Fine-tune là tiếp tục huấn luyện Qwen3-14B trên các cặp **(prompt → đáp án đúng)** lấy từ **train**, để model làm
tốt *đúng những việc agent giao cho nó*. Với LoRA, chỉ một phần nhỏ trọng số phụ (adapter) được học; sau đó adapter
được **merge** vào model gốc. Kết quả là một model 14,8B mới, dùng thay Qwen3-14B gốc ở mọi bước LLM.

**G1 (bộ luật) không fine-tune được** vì nó là code. Thứ được fine-tune là **Qwen3-14B**, phục vụ G2, G3, G4 và
selection model.

### Tại sao cần

Qwen3-14B gốc không biết:
- Frame DSL của repo: các intent, trường JSON, cách ghi mốc thời gian;
- các view chuẩn hoá và luật nghiệp vụ, ví dụ `is_valid`, đơn trả hàng bị loại cả đơn, tuần bắt đầu thứ Hai;
- teencode và phương ngữ của câu hỏi cuộc thi;
- định dạng đầu ra mà agent đọc được.

Few-shot trong prompt chỉ giúp được một phần. Paper XiYan-SQL cho thấy generator **fine-tune đa nhiệm** vượt cả
GPT-4o dùng few-shot. Với metric PA, lợi ích nằm ở chỗ: ứng viên LLM **đúng thường xuyên hơn** thì độ tin cậy đã
hiệu chuẩn cao hơn, nhiều câu vượt τ hơn, nên bớt `abstain` mà không thêm câu sai. Lợi ích chủ yếu ở những câu G1
chưa hiểu: cách nói mới của public/private test.

### Fine-tune những nhiệm vụ nào

Tất cả gộp vào **một** lần huấn luyện, **một** model (để giữ 14,8B):

| Nhiệm vụ | Dùng ở | Dữ liệu sinh bởi | Mục tiêu học |
|---|---|---|---|
| `rewrite` | G3 | `training/build_sft_data.py` | câu lộn xộn → câu chuẩn mà G1 hiểu |
| `frame` | G2 | `training/build_sft_data.py` | câu hỏi → JSON frame đúng (kể cả `unanswerable`) |
| `nl2sql_base` / `nl2sql_cte` / `nl2sql_alias` | G4 | `training/build_sql_sft.py` | câu hỏi + schema + evidence → SQL đúng, theo 3 định dạng |
| `question_inference` | (phụ trợ) | `training/build_sql_sft.py` | SQL → câu hỏi (gắn kết ngôn ngữ ↔ SQL, như paper) |
| `evidence_inference` | (phụ trợ) | `training/build_sql_sft.py` | chọn luật nghiệp vụ mà SQL dùng |
| `self_refine` | G4 refine | `training/build_sql_sft.py` | SQL sai (near-miss) + kết quả → SQL đúng |
| `select` | G_select | `training/build_sql_sft.py` | chọn SQL đúng trong các ứng viên |

Mọi đáp án đích đều **đã kiểm chứng**: frame tái tạo đúng đáp án train, và SQL chạy ra đúng đáp án train. Prompt
trong dữ liệu giống hệt prompt lúc suy luận: cùng vocab của shop, rulebook, few-shot (loại chính shop đó) và schema
mà Schema Filter cho ra.

### Không làm

- Không fine-tune trên public/private, kể cả câu hỏi không có nhãn.
- Không train và merge **nhiều** model riêng cho từng định dạng: mỗi model 14,8B, cộng lại vượt 15B. Định dạng SQL
  được chọn bằng chỉ dẫn trong prompt của cùng một model.
- GRPO (`training/grpo_abstain.py`) là bước **tuỳ chọn** sau SFT, chưa kiểm chứng. Chỉ làm khi SFT đã có lợi.

## Phần F — Fine-tune từng bước

Quy trình hai pha giúp **đo được** fine-tune có lợi hay không, mà không chạm vào public:

- **Pha A (đo):** train trên 24 shop, giữ riêng 6 shop của fold 0. So model gốc với model fine-tune trên 6 shop đó.
- **Pha B (bản cuối):** nếu pha A có lợi, train lại trên **cả 30 shop** với cùng siêu tham số, rồi dùng để nộp.

### F0. Cài môi trường huấn luyện

```bash
pip install "vllm==0.12.0"                  # kéo theo torch 2.9.0
pip install -r requirements-train.txt       # đã ghim: transformers 4.57.6, trl 0.24.0, peft 0.17.1, bitsandbytes
export DATAMATE_CACHE=.cache
```

### F1. Artifact từ train (vài phút, CPU)

```bash
python scripts/build_artifacts.py data/train        # few-shot bank (có shop id), vocab
python scripts/build_sql_silver.py data/train       # SQL đã kiểm chứng (1.671 câu)
```

### F2. Pha A — dữ liệu không có fold 0

```bash
python training/build_sft_data.py --split data/train --out outputs/sft_f0.jsonl --n-aug 2 --holdout-fold 0
python training/build_sql_sft.py  --split data/train --out outputs/sql_sft_f0 --n-aug 1 --holdout-fold 0
```

Script in danh sách 6 shop bị giữ lại: `biz_101, biz_106, biz_111, biz_116, biz_121, biz_126`. Đây cũng là các shop mà
`--fold 0` chọn ở bước F4. Few-shot trong prompt huấn luyện không bao giờ lấy từ chính shop đó hay từ shop bị giữ lại.
Nếu bank chưa có shop id, `build_sft_data.py` sẽ dừng và báo chạy lại `build_artifacts.py`.

Kích thước đo được (fold 0, `--n-aug 1` cho cả hai lệnh): 9.788 ví dụ G2/G3 và 16.886 ví dụ SQL.

| Nhiệm vụ | Độ dài prompt + đáp án, trung vị / p95 (ký tự) |
|---|---|
| `frame` | 11,5k / 14,6k (dài nhất: có danh sách intent, rulebook, vocab, few-shot) |
| `nl2sql_*`, `select` | 3,5k / 5,3k |
| `rewrite`, `self_refine` | 2,8–3,4k / 4–6,6k |

Với tiếng Việt có dấu, mỗi token khoảng 3 ký tự, nên `--max-len 8192` đủ cho gần như mọi ví dụ.

### F3. Pha A — huấn luyện và merge

```bash
# 1 phút: chạy thử toàn bộ đường train (model Qwen3 tí hon, tokenizer thật); phải in "SMOKE TEST OK"
python training/smoke_test.py --base /models/qwen3-14b --data outputs/sft_f0.jsonl outputs/sql_sft_f0/sql_sft.jsonl
# GPU 24–48 GB: --qlora ; GPU 80 GB: bỏ --qlora
python training/sft_lora.py --model /models/qwen3-14b \
  --data outputs/sft_f0.jsonl outputs/sql_sft_f0/sql_sft.jsonl \
  --out outputs/lora_f0 --qlora --epochs 1 --max-per-task 6000 --max-len 8192
python training/sft_lora.py --merge --model /models/qwen3-14b --adapter outputs/lora_f0 --out outputs/merged_f0
python scripts/count_params.py outputs/merged_f0     # phải bằng số của model gốc (~14.8B)
```

Gợi ý siêu tham số ban đầu: 1 epoch, `lr 1e-4`, rank 32, tối đa 6.000 ví dụ mỗi nhiệm vụ.

Ước lượng thô, **chưa đo thật**: khoảng 30–40k ví dụ × khoảng 1,5–2k token, tức 50–80M token mỗi epoch. Trên một
H100 với LoRA BF16 có thể mất cỡ 4–8 giờ. QLoRA trên một GPU 24–32 GB chậm hơn 2–3 lần. Muốn thử nhanh trước, giảm
`--max-per-task` (ví dụ 1500).

Theo dõi `loss` trong log: loss giảm đều là tốt; nếu loss tăng hoặc dao động mạnh, giảm `--lr`.

### F4. Pha A — so sánh trên 6 shop chưa từng thấy

```bash
python -m datamate.predict --data data/train --fold 0 --llm vllm --model /models/qwen3-14b --out runs/f0_base/submission.jsonl
python -m datamate.predict --data data/train --fold 0 --llm vllm --model outputs/merged_f0 --out runs/f0_ft/submission.jsonl
python -m datamate.predict --data data/train --fold 0 --llm none --out runs/f0_rule/submission.jsonl
```

Mỗi lệnh in `PA=...`. So sánh thêm bằng `validate.py`:

```bash
python scripts/validate.py runs/f0_ft/logs data/train/answers.jsonl --folds 3
```

**Quyết định:**
- Model fine-tune có PA cao hơn model gốc **và** cao hơn rule-only trên fold 0 → làm tiếp F5.
- Không cao hơn → không dùng fine-tune. Thử chỉnh: ít epoch hơn, `lr` nhỏ hơn, bỏ bớt nhiệm vụ (ví dụ chỉ `rewrite`
  và `frame`), rồi lặp lại F2–F4.

Sau đó hiệu chuẩn độ tin cậy của các generator LLM. Đây là số đo trung thực, vì model chưa từng thấy 6 shop này:

```bash
python scripts/validate.py runs/f0_ft/logs data/train/answers.jsonl --folds 3 --write-reliability artifacts/reliability.json
```

### F5. Pha B — model cuối trên cả 30 shop

```bash
python training/build_sft_data.py --split data/train --out outputs/sft_all.jsonl --n-aug 2
python training/build_sql_sft.py  --split data/train --out outputs/sql_sft_all --n-aug 1
python training/sft_lora.py --model /models/qwen3-14b --data outputs/sft_all.jsonl outputs/sql_sft_all/sql_sft.jsonl \
  --out outputs/lora_all --qlora --epochs 1 --max-per-task 6000 --max-len 8192      # cùng siêu tham số như pha A
python training/sft_lora.py --merge --model /models/qwen3-14b --adapter outputs/lora_all --out outputs/merged
python scripts/count_params.py outputs/merged
```

Giữ `artifacts/reliability.json` đo ở pha A. Model pha B đã thấy cả train nên không đo lại trên train được; số của
pha A hơi thận trọng, như vậy an toàn hơn.

### F6. (Tuỳ chọn) GRPO với phần thưởng PA

```bash
python training/grpo_abstain.py --model outputs/merged --split data/train --out outputs/grpo
```

Thưởng +1 khi đúng, −1 khi sai, 0 khi abstain (theo bài "When Silence is Golden"). Đây là code thử nghiệm. Muốn
dùng thì đánh giá giống F4: train GRPO trên model pha A, đo trên fold 0, chỉ giữ khi PA tăng.

### F7. Sinh file nộp bằng model đã fine-tune

```bash
python -m datamate.predict --data data/public_test --llm vllm --model outputs/merged --tp 2 \
  --out runs/public_ft/submission.jsonl
python scripts/check_submission.py runs/public_ft/submission.jsonl data/public_test/questions.jsonl
```

### Mẹo để tối ưu toàn project

1. **Đo trước, thêm sau:** mỗi thay đổi (fine-tune, `--g4-scope all`, τ, thêm hay bớt generator) đều đo trên fold
   giữ riêng bằng `--fold` và `validate.py`. Không chọn theo điểm public, vì như thế là quá khớp với public.
2. **τ:** `validate.py` in bảng quét τ. Chọn τ cho PA cao nhất trên fold giữ riêng; thường nằm trong 0,55–0,7.
3. **Đọc log lỗi:** `runs/validate/errors.jsonl` liệt kê câu sai, generator, intent và p. Câu sai lặp lại theo một
   kiểu thường là do thiếu dữ liệu huấn luyện kiểu đó, hoặc một luật sai trong G1.
4. **Thời gian:** đo thời gian chạy public trên đúng phần cứng chấm. Private có thể nhiều câu hơn; nếu sát 6 giờ thì
   giữ `--g4-scope unresolved` và giảm `g4_max_per_biz` trong `datamate/agent.py`.

## Phần G — Đóng gói Docker và nộp

```bash
# Model gốc (tải lúc build):
docker build -t datamate-xiyan:final .
# Model gốc + adapter LoRA đã commit trong models/adapter (không cần model merge 30 GB; chạy git lfs pull trước):
docker build -f Dockerfile.lora -t datamate-xiyan:lora .
# Model đã fine-tune và merge (outputs/merged phải có sẵn; cần Docker BuildKit):
docker build -f Dockerfile.finetuned -t datamate-xiyan:ft .

# Chạy đúng như môi trường chấm: offline, dữ liệu chỉ-đọc ở /data, kết quả ở /result
mkdir -p result
time docker run --rm --gpus all --network none \
  -v "$(pwd)/data/public_test:/data:ro" -v "$(pwd)/result:/result" datamate-xiyan:ft
python scripts/check_submission.py result/submission.jsonl data/public_test/questions.jsonl
```

Biến môi trường dùng được khi `docker run -e ...`:

| Biến | Ý nghĩa |
|---|---|
| `DATAMATE_LLM` | `vllm` (mặc định) hoặc `none` |
| `DATAMATE_TP` | số GPU |
| `DATAMATE_LORA` | thư mục adapter LoRA (đặt sẵn trong `Dockerfile.lora`) |
| `DATAMATE_QUANT` | lượng tử hoá, ví dụ `fp8` |
| `DATAMATE_MAX_LEN` | `max_model_len` của vLLM |
| `DATAMATE_GPU_UTIL` | `gpu_memory_utilization` của vLLM |
| `DATAMATE_GENS` | generator LLM được bật, ví dụ `g2,g3,g4` |
| `DATAMATE_G4_SCOPE` | `unresolved` hoặc `all` |
| `DATAMATE_TAU` | ngưỡng abstain |
| `DATAMATE_TIME_BUDGET` | ngân sách thời gian, tính bằng giây |

Nộp `result/submission.jsonl` lên leaderboard và nộp image theo hướng dẫn của ban tổ chức. Giữ `result/logs/`.

## Phần H — Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| stderr in `-> rule-only mode` | Sai `--model`, thiếu GPU hoặc vLLM lỗi; xem dòng `[llm] failed to start …` |
| `CUDA out of memory` khi suy luận | Giảm `--max-len` (12288 hoặc 8192), thêm `--quant fp8`, hoặc tăng `--tp` |
| `CUDA out of memory` khi fine-tune | Thêm `--qlora`, giảm `--max-len` (6144), giữ `--batch 1` |
| `Could not find a version that satisfies the requirement vllm==0.10.1.1` / `Requires-Python <3.13` | Python 3.13 (Colab hiện nay) không có vLLM 0.10.x. Repo đã chuyển sang `vllm==0.12.0` (Python 3.10–3.13, torch 2.9.0, Dockerfile `vllm/vllm-openai:v0.12.0`): `git pull` rồi cài lại |
| `SFTConfig.__init__() got an unexpected keyword argument 'warmup_ratio'`, hoặc cảnh báo `torch_dtype is deprecated` | Môi trường đang có transformers 5.x / TRL 1.x. Cài lại bản đã ghim: `pip install -r requirements-train.txt` (transformers 4.57.6, trl 0.24.0, peft 0.17.1, cài **sau** `vllm==0.12.0`), restart session, chạy `training/smoke_test.py` |
| Log không có `G4_*` | G4 chỉ chạy cho câu chưa chắc; dùng `--g4-scope all` để thử, hoặc kiểm tra `--generators` |
| PA train với LLM thấp hơn rule-only | Bình thường khi chưa hiệu chuẩn: chạy D4 (`--write-reliability`), hoặc tắt generator kém (D5) |
| Docker build lỗi ở `COPY outputs/merged` | Chưa merge (F5), hoặc Docker không có BuildKit; bật bằng `DOCKER_BUILDKIT=1` |
