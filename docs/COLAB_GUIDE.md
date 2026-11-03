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

1. **GPU:** Runtime → Change runtime type → **A100** (tốt nhất) hoặc **L4**, bật **High-RAM** nếu có. **T4 không
   dùng được**: không có bf16 và chỉ có 16 GB.
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
| 2 | Kiểm tra GPU, mount Drive, clone repo | notebook tự chọn QLoRA (GPU < 70 GB) và FP8 cho vLLM (GPU < 45 GB) |
| 3 | Cài vLLM + thư viện, tải Qwen3-14B | khoảng 10–20 phút |
| 4 | Tạo dữ liệu từ **train** (pha A giữ riêng fold 0 = 6 shop) | khoảng 20–40 phút CPU; lưu trên Drive |
| 5 | Smoke test 1 phút, rồi train + xuất adapter bf16 | lâu nhất: khoảng 3–5 giờ trên A100 với 1500 ví dụ/nhiệm vụ (ước lượng, chưa đo) |
| 6 | **Restart session**, rồi đánh giá 3 cấu hình trên 6 shop giữ riêng, hiệu chuẩn | in `PA` của rule / base / lora |
| 7 | Tạo `submission.jsonl` cho public test bằng model fine-tune | kiểm tra định dạng tự động |
| 8 | Đẩy adapter, reliability, file nộp lên nhánh `colab-finetune-phasea` | token đọc từ Secrets, không in ra |
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

## 4. Đọc kết quả và quyết định

Ô 6 in ra ba dòng, ví dụ `rule PA = …`, `base PA = …`, `lora PA = …`, tính trên 553 câu của 6 shop mà model chưa từng
thấy.

- `lora` **cao hơn cả** `base` và `rule`: fine-tune có lợi. Có thể nộp file ở ô 7, rồi làm **pha B**: đặt `PHASE="B"`
  (và `MAX_PER_TASK` lớn hơn nếu đủ unit), chạy lại notebook để có adapter học trên cả 30 shop. Pha B bỏ qua bước
  đánh giá và giữ `reliability_phaseA.json`.
- Không cao hơn: **đừng dùng adapter**. Thử `LR = 5e-5`, đổi `MAX_PER_TASK`, xoá thư mục `lora_phaseA` trên Drive rồi
  train lại.

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

## 6. Lưu ý về luật và chi phí

- **Chỉ dữ liệu train** được dùng để fine-tune và hiệu chuẩn; notebook không đọc nhãn public (public không có nhãn),
  và không đưa câu hỏi public vào dữ liệu train.
- **≤ 15B tham số:** base + adapter khoảng 14,9B. Notebook in ra con số này sau khi train.
- **Repo đang public:** trước khi đẩy adapter (học từ dữ liệu cuộc thi) và dữ liệu, đọc lại quy định về chia sẻ dữ
  liệu, weights của cuộc thi. Nếu không chắc, chuyển repo sang private, hoặc chỉ lưu adapter trên Hugging Face
  private.
- **Git LFS:** mỗi lần đẩy một adapter tốn khoảng 0,26 GB trong hạn mức 1 GB lưu trữ / 1 GB băng thông miễn phí mỗi
  tháng. Mỗi lần ai đó `git lfs pull` cũng tính vào băng thông. Khi hết hạn mức, xoá các nhánh adapter cũ, mua thêm
  gói LFS, hoặc chuyển adapter sang Hugging Face Hub (`scripts/publish_hf.py models/adapter <user>/<repo>`).
