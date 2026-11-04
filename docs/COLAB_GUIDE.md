# Fine-tune trên Google Colab Pro và lưu kết quả về GitHub

Notebook: [`notebooks/colab_finetune.ipynb`](../notebooks/colab_finetune.ipynb). Mở thẳng trên Colab:
`https://colab.research.google.com/github/AIVIETNAM-AIO-khaisiubeo/datamate-xiyan-/blob/gpu-finetune-guide/notebooks/colab_finetune.ipynb`

Notebook làm đúng quy trình hai pha của [`GPU_FINETUNE_GUIDE.md`](GPU_FINETUNE_GUIDE.md), Phần F, nhưng đã chỉnh cho
Colab: chọn QLoRA hay LoRA theo GPU, lưu checkpoint lên Drive, train tiếp khi mất kết nối, và cuối cùng đẩy kết quả
lên một **nhánh mới** của repo.

## 1. Cái gì được lưu ở đâu, và tại sao

| Thứ | Kích thước | Lưu ở | Lý do |
|---|---|---|---|
| Code, tài liệu, `artifacts/reliability.json`, `submissions/*.jsonl` | nhỏ | **GitHub** (git thường) | — |
| **LoRA adapter** đã fine-tune (`models/adapter`, bf16, rank 32) | khoảng **0,26 GB** | **GitHub qua Git LFS** | GitHub chặn file > 100 MB; LFS cho phép, nhưng gói miễn phí chỉ có 1 GB lưu trữ và 1 GB băng thông mỗi tháng |
| `models/adapter/TRAINING.md` (dữ liệu, siêu tham số, PA trên fold giữ riêng) | nhỏ | GitHub | để biết adapter được tạo thế nào |
| Qwen3-14B gốc | khoảng 30 GB | **không lưu**, tải từ Hugging Face khi cần (notebook, `docker build`) | đã công khai sẵn; GitHub không chứa nổi |
| Model merge (base + adapter) | khoảng 30 GB | (tuỳ chọn) **Hugging Face Hub, private** — `scripts/publish_hf.py` | quá lớn cho GitHub; chỉ cần nếu muốn dùng `Dockerfile.finetuned` |
| Checkpoint, dữ liệu fine-tune, log đánh giá | vài GB | **Google Drive** (`MyDrive/datamate`) | sống sót khi Colab ngắt kết nối; tạo lại được bất cứ lúc nào |

**Không cần model merge 30 GB:** hệ thống chạy được với *model gốc + adapter* (`--lora models/adapter`, hoặc
`Dockerfile.lora`). Tổng tham số là 14,77B + 0,13B ≈ 14,9B, vẫn ≤ 15B. Kiểm bằng
`python scripts/count_params.py /models/qwen3-14b models/adapter`.

## 2. Chuẩn bị (một lần)

1. **GPU:** Runtime → Change runtime type → **A100** (tốt nhất) hoặc **L4** để **train**, bật **High-RAM** nếu có.
   **T4** chỉ dùng để **suy luận** với adapter đã train (xem mục 6c): T4 không có bf16 và chỉ có 16 GB.
2. **Token GitHub** để notebook đẩy kết quả:
   - GitHub → Settings → Developer settings → *Fine-grained tokens* → *Generate new token*.
   - *Repository access*: chỉ chọn `datamate-xiyan-`. *Permissions*: **Contents: Read and write**. Đặt hạn dùng ngắn.
   - Trong Colab, mở biểu tượng 🔑 *Secrets* → *Add new secret*, tên `GITHUB_TOKEN`, dán token, bật **Notebook access**.
   - Không dán token vào ô code hay vào file nào.
3. (Tuỳ chọn) Thêm secret `HF_TOKEN` (Hugging Face, quyền *write*) nếu muốn publish model merge lên Hub.
4. **Dung lượng:** Drive cần khoảng 5 GB trống cho checkpoint và dữ liệu. Đĩa của máy Colab (khoảng 100–200 GB) chứa
   model gốc.

## 3. Chạy

Chạy các ô **theo thứ tự từ trên xuống**:

| Ô | Việc | Ghi chú |
|---|---|---|
| 1 | Cấu hình | `PHASE="A"` lần đầu; `MAX_PER_TASK=1500` để thử nhanh |
| 2 | Kiểm tra GPU, mount Drive, **cài git-lfs**, clone hoặc cập nhật repo | notebook tự chọn QLoRA (GPU < 70 GB) và FP8 cho vLLM (GPU < 45 GB) |
| 3 | Cài vLLM + thư viện, tải Qwen3-14B | khoảng 10–20 phút |
| 4 | Tạo dữ liệu từ **train** (pha A giữ riêng fold 0 = 6 shop) | khoảng 20–40 phút CPU; lưu trên Drive |
| 5 | Smoke test 1 phút, rồi train + xuất adapter bf16 | lâu nhất: khoảng 5 giờ trên A100 với 1500 ví dụ/nhiệm vụ (đã đo: 822 bước) |
| 6 | **Restart session**, rồi đánh giá rule / base / lora trên **bản nhiễu** của 6 shop giữ riêng, hiệu chuẩn | in bảng PA cả hệ thống và PA LLM-only; khoảng 1–2 giờ |
| 7 | Tạo `submission.jsonl` cho public test bằng model fine-tune, rồi so với bản rule-only | tự tạo lại khi reliability đổi; kiểm tra định dạng tự động |
| 8 | Đẩy adapter, reliability, file nộp lên nhánh `colab-finetune-phasea` | kiểm tra adapter đi qua LFS trước khi đẩy, kiểm tra nhánh đã có trên GitHub sau khi đẩy; in `XONG: <link>` |
| 9 | (Tuỳ chọn) Merge và publish lên Hugging Face Hub | cần `HF_TOKEN`, khoảng 60 GB đĩa |

**Khi bị ngắt kết nối** (thường gặp với job dài): máy ảo mới không còn thư viện và model, nhưng Drive vẫn giữ checkpoint
và dữ liệu. Mở lại notebook và chạy lại từ ô 1. Các bước đã xong (dữ liệu, model đã tải) sẽ được bỏ qua, còn ô 5 tự
tiếp tục train từ checkpoint mới nhất (`--resume`, lưu mỗi 200 bước). Để bớt mất kết nối, giữ tab Colab mở; Colab Pro
cho phiên dài hơn nhưng không vô hạn.

**Compute units:** A100 tiêu nhiều unit mỗi giờ hơn L4. Xem số unit còn lại ở góc phải trên. Nếu ít unit, giảm
`MAX_PER_TASK` (ví dụ 800) cho pha A.

**Lỗi `No matching distribution found for vllm==0.10.1.1`** (pip báo `Requires-Python <3.13`): Colab đã lên Python
3.13 mà vLLM 0.10.x chưa hỗ trợ. Notebook mới dùng `vllm==0.12.0` (Python 3.10–3.13). Chạy lại ô clone (`git pull`) rồi ô
cài thư viện; dòng kiểm tra phải in `vllm 0.12.0`.

**Lỗi `unexpected keyword argument 'warmup_ratio'`** (hoặc cảnh báo `torch_dtype is deprecated`): phiên Colab đang có
transformers 5.x. Kéo code mới (ô clone chạy `git pull`), chạy lại ô cài thư viện để có bản đã ghim (transformers
4.57.6, trl 0.24.0, peft 0.17.1), **Runtime → Restart session**, chạy lại ô 1, ô GPU, ô Drive. Bỏ qua tải model nếu đã
tải, rồi chạy ô smoke test trước khi train.

**Lỗi khi đẩy: `git: 'lfs' is not a git command`, rồi `GH001: Large files detected ... exceeds GitHub's file size
limit of 100.00 MB`**: máy Colab không có sẵn git-lfs. Adapter 245 MB bị commit như file thường, và GitHub từ chối cả
lần đẩy, nên trên GitHub không thấy gì. Notebook mới cài git-lfs ở ô clone, và ô 8 dừng lại kèm thông báo nếu adapter
chưa đi qua LFS. Xem mục 6 để làm lại mà không phải train lại.

## 4. Đọc kết quả và quyết định

**Vì sao không dùng PA trên câu train gốc:** luật G1 được viết khi đọc train, nên trên 553 câu gốc của fold giữ riêng
cả ba cấu hình đều cho `PA = 1.0000`. Con số đó không cho biết fine-tune có giúp gì không. Ô 6 in một bảng khác:

| Cột | Nghĩa | Dùng để |
|---|---|---|
| `PA bản nhiễu` | PA cả hệ thống trên cùng các câu hỏi đó, thêm lỗi gõ, viết tắt, bỏ dấu (đáp án không đổi) | kiểm tra LLM không làm hỏng câu G1 đã đúng; rule-only ở đây khoảng 0,985 |
| `LLM-only PA`, `coverage`, `precision` | chấm riêng G2, G3, G4, G_select khi bỏ G1 | **so `base` với `lora`** |

Bảng `per generator` còn cho biết, với từng generator, độ chính xác trên các câu mà G1 không trả lời được (cột
`G1 silent`). Đó là chỗ LLM có thể thêm điểm trên public.

- `lora >= rule` trên bản nhiễu **và** `LLM-only: lora > base`: fine-tune có lợi. Nộp file ở ô 7. Có thể làm **pha B**:
  đặt `PHASE="B"` (và `MAX_PER_TASK` lớn hơn nếu đủ unit), chạy lại notebook để có adapter học trên cả 30 shop. Pha B
  bỏ qua bước đánh giá và giữ `reliability_phaseA.json`.
- `LLM-only: lora > base` sai: **đừng dùng adapter**. Thử `LR = 5e-5`, đổi `MAX_PER_TASK`, xoá thư mục `lora_phaseA`
  trên Drive rồi train lại.
- `lora >= rule` sai: LLM đang làm hỏng một vài câu. Xem `eval_phaseA/validate_lora/errors.jsonl` trên Drive.

Bản nhiễu vẫn làm từ train, nên nó chỉ là ước lượng. **Điểm thật duy nhất trên public là leaderboard.** Ô 7 có in phần
so với bản rule-only (PA public 0,561): bao nhiêu câu mới được trả lời, bao nhiêu câu đổi đáp án, do generator nào. Mỗi
câu như vậy được +1 nếu đúng và −1 nếu sai. Không sửa luật hay prompt theo từng câu public: như thế là quá khớp với
public, và private sẽ khác.

## 5. Sau khi notebook đẩy lên GitHub

1. Trên GitHub, mở nhánh `colab-finetune-phasea` (hoặc `…phaseb`). Xem `models/adapter/TRAINING.md` và PA, rồi tạo
   pull request vào `main` nếu muốn giữ.
2. Trên máy của bạn:

   ```bash
   git fetch origin && git checkout colab-finetune-phasea
   git lfs install && git lfs pull          # tải weights thật, không phải file con trỏ LFS
   ```

3. Dùng adapter:

   ```bash
   python -m datamate.predict --data data/public_test --llm vllm --model /models/qwen3-14b \
     --lora models/adapter --out runs/public_lora/submission.jsonl
   docker build -f Dockerfile.lora -t datamate-xiyan:lora .        # image nộp: base tải lúc build + adapter
   ```

## 6. Đã train xong nhưng GitHub không thấy gì: làm lại thế nào (không cần train lại)

Adapter, dữ liệu và log đánh giá đều nằm trên Drive (`MyDrive/datamate`), nên không phải train lại.

1. Mở lại notebook từ link ở đầu trang, để có bản mới nhất. Bản cũ đã lưu trong Drive của bạn vẫn là bản cũ.
2. Chạy ô 1, các ô của mục 2 (ô clone giờ cài git-lfs, rồi đặt repo về đúng `origin/gpu-finetune-guide`; commit hỏng
   của lần trước bị bỏ), và mục 3. Ô tải model tự bỏ qua nếu model còn trên máy.
3. **Bỏ qua mục 4 và 5**: dữ liệu và adapter đã có trên Drive.
4. Chạy mục 6. Bản nhiễu được tạo mới, và rule / base / lora được chạy lại với G4 trên mọi câu. Các lần chạy cũ trên câu
   gốc vẫn được giữ; ô tuỳ chọn cuối mục 6 chấm lại chúng mà không tốn GPU. `reliability_phaseA.json` cũ (thiếu
   `G1_rule`) bị xoá và tạo lại.
5. Chạy mục 7. File nộp cũ được tạo với reliability thiếu G1, nên nó được chuyển sang `public_phaseA_prev` và file mới
   được tạo lại.
6. Chạy mục 8. Ô phải in `đã kiểm tra: ... adapter đi qua Git LFS`, rồi `Uploading LFS objects: 100% (1/1), ...` (khoảng 250 MB), và
   cuối cùng là `XONG: https://github.com/.../tree/colab-finetune-phasea`. Nếu không thấy dòng `XONG`, ô đã dừng
   ở một bước kèm thông báo lỗi.

## 6b. Sau lần nộp 0,777 (bản LoRA thấp hơn bản rule-only)

Nguyên nhân nằm ở cách kết hợp, không phải ở adapter. So file LoRA với file rule-only (116/1000 câu khác nhau):

- **73 câu** LLM thay một đáp án G1 không bị đánh dấu nghi ngờ. G2 và G3 là cùng một LLM nhưng được tính như hai phiếu.
- **46 câu** đang có số liệu bị đổi thành `unanswerable`. Độ tin cậy của khoá `unanswerable` gần như chỉ là giá trị mặc
  định 0,9, vì train có rất ít câu loại này.
- **G4 không có ứng viên nào** với adapter. Adapter viết `SELECT ...` rồi đóng ```` ``` ````, và dấu đó bị giữ lại trong
  SQL, nên mọi câu SQL đều lỗi cú pháp.

Code mới sửa cả ba (xem `datamate/select/selector.py`). Không cần train lại. Thứ tự chạy:

1. Ô 1, mục 2, mục 3. Bỏ qua mục 4 và 5.
2. **Chỉ bước 1, không GPU:** ô validate của mục 6 (tạo lại `reliability_phaseA.json` từ log cũ), rồi ô **7a**. Ô 7a
   áp bộ chọn mới lên lần chạy public cũ và ghi `public_phaseA_replay/submission.jsonl`. Dòng `--legacy` phải in
   `same_as_logged_final 1000`.
3. **Bước 1 + 2 (GPU):** ô chạy của mục 6 (tự chạy lại `noisy_lora` vì G4 giờ chạy được, khoảng 1 giờ), ô validate,
   bảng kết quả, rồi ô **7b** (khoảng 15 phút).
4. Mục 8 đẩy cả hai file lên GitHub.

Nộp mỗi file **một lần** và ghi điểm vào `docs/RESULTS.md`: file 7a đo riêng tác dụng của bộ chọn mới, file 7b đo thêm
G4. Trên public, phần lệch so với bản rule-only giờ chỉ còn ở những câu G1 bỏ trống hoặc tự đánh dấu nghi ngờ.

## 6c. Chỉ có T4 (hoặc không có GPU)

Ô GPU ở mục 2 tự chọn `PROFILE`:

| | A100 / L4 (`bf16`) | T4 (`t4`) | không GPU (`cpu`) |
|---|---|---|---|
| Train (mục 4–5) | có | **bỏ qua**: 14B trên T4 mất hàng chục giờ, và không có bf16 | bỏ qua |
| Model suy luận | `Qwen/Qwen3-14B` (FP8 nếu < 45 GB) | **`Qwen/Qwen3-14B-AWQ`**: cùng model, lượng tử 4-bit (khoảng 10 GB), chạy fp16 | — |
| Adapter | `lora_phaseA/adapter_bf16` | cùng adapter đó, áp lên bản AWQ | — |
| Mục 6, chạy | rule, base, lora; G4 trên mọi câu | rule; base chỉ khi có log A100 hoặc `T4_RUN_BASE = True`; lora chạy lại; G4 trên `T4_G4_MAX` câu mỗi shop | chỉ rule |
| Mục 6, validate; 7a replay | có | có | **có** (chỉ CPU) |
| 7b, public | khoảng 15 phút | khoảng 1,5–3 giờ (ước lượng) | không |

Lưu ý khi chạy trên T4:
- **Luật ≤ 15B:** bản AWQ vẫn là 14,8B tham số. Chỉ cách lưu trữ thay đổi; `scripts/count_params.py` đếm lại các trọng
  số 4-bit đã đóng gói.
- **Adapter được train trên bản bf16** rồi áp lên bản AWQ, nên kết quả có thể lệch một chút so với A100. So `base`
  (A100) với `lora` (T4) ở mục 6 chỉ là gần đúng.
- Mọi lệnh LLM chạy với `--require-llm`. Nếu vLLM không khởi động được trên T4, ô **dừng kèm lỗi**, thay vì lặng lẽ
  chạy rule-only như trước.
- Ô **"Kiểm tra nhanh LLM"** (cuối mục 3) khởi động vLLM và trả lời 3 câu, mất 5–10 phút. Chạy nó trước mọi lần chạy
  dài. Khi lỗi, mọi ô chạy LLM in **các dòng lỗi gốc** trong log (vLLM in nguyên nhân ở giữa log, không ở cuối).
- Trên T4, vLLM dùng `--max-num-seqs 16 --max-batched-tokens 2048 --gpu-util 0.94`, để còn đủ KV cache cho một câu
  8192 token. Nếu vẫn báo thiếu bộ nhớ (`out of memory`, `KV cache is needed, which is larger than the available`):
  ở ô GPU, đổi `--max-len 8192` thành `--max-len 6144`, rồi chạy lại ô GPU và ô kiểm tra nhanh.
- Nếu dòng lỗi nói về *attention backend*: đặt `VLLM_ATTN = "TRITON_ATTN"` ở ô 1, rồi chạy lại từ ô GPU.
- Thứ tự rẻ nhất: ô validate + **7a** (CPU, vài phút) cho file nộp bước 1. Sau đó mới đến `noisy_lora` và 7b trên T4.

**Lần chạy T4 đầu tiên báo `Engine core initialization failed`.** Có hai nguyên nhân thấy được trong output:
1. Ô đếm tham số in `0.000B`: thư mục Qwen3-14B-AWQ không có file trọng số, có lẽ do lần tải trước bị ngắt. Ô cũ chỉ kiểm
   tra `config.json`, nên không tải tiếp. Ô mới luôn gọi `snapshot_download` (tải tiếp phần còn thiếu) và kiểm tra đủ
   các file trọng số.
2. Notebook không thấy adapter, cũng không thấy log của các lần chạy trước. Drive đang mount không phải Drive của lần
   chạy A100: hoặc ô mount chưa chạy, hoặc đây là tài khoản Google khác. Ô mount giờ kiểm tra Drive đã thật sự được
   nối. Ô sau ô clone liệt kê những gì đang thiếu, và tự lấy adapter từ nhánh `colab-finetune-phasea` trên GitHub
   (khoảng 0,26 GB băng thông Git LFS). Log của lần nộp 0,777 (cho ô 7a) chỉ có trên Drive cũ: ô đó chỉ ra cách lấy.

## 7. Lưu ý về luật và chi phí

- **Chỉ dữ liệu train** được dùng để fine-tune và hiệu chuẩn; notebook không đọc nhãn public (public không có nhãn),
  và không đưa câu hỏi public vào dữ liệu train.
- **≤ 15B tham số:** base + adapter khoảng 14,9B. Notebook in ra con số này sau khi train.
- **Repo đang public:** trước khi đẩy adapter (học từ dữ liệu cuộc thi) và dữ liệu, đọc lại quy định về chia sẻ dữ
  liệu, weights của cuộc thi. Nếu không chắc, chuyển repo sang private, hoặc chỉ lưu adapter trên Hugging Face
  private.
- **Git LFS:** mỗi lần đẩy một adapter tốn khoảng 0,26 GB trong hạn mức 1 GB lưu trữ / 1 GB băng thông miễn phí mỗi
  tháng. Mỗi lần ai đó `git lfs pull` cũng tính vào băng thông. Khi hết hạn mức, xoá các nhánh adapter cũ, mua thêm
  gói LFS, hoặc chuyển adapter sang Hugging Face Hub (`scripts/publish_hf.py models/adapter <user>/<repo>`).
