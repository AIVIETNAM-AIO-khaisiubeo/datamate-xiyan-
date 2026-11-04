"""Generate notebooks/colab_finetune.ipynb (run: python scripts/make_colab_notebook.py)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# DataMate-XiYan — fine-tune Qwen3-14B trên Google Colab và lưu kết quả về GitHub

Notebook chạy toàn bộ quy trình trong `docs/GPU_FINETUNE_GUIDE.md` trên Colab Pro:

1. **Cài đặt**: kiểm tra GPU, nối Google Drive, clone repo, cài thư viện, tải Qwen3-14B.
2. **Dữ liệu**: tạo dữ liệu fine-tune từ **train** (không bao giờ dùng public test).
3. **Huấn luyện**: QLoRA hoặc LoRA. Checkpoint lưu trên Drive, mất kết nối thì chạy lại để train tiếp.
4. **Đánh giá** (pha A): so rule-only, model gốc và model fine-tune trên bản nhiễu của 6 shop giữ riêng (PA cả hệ
   thống, và PA của riêng các generator LLM), rồi hiệu chuẩn độ tin cậy.
5. **Tạo `submission.jsonl`** cho public test bằng model đã fine-tune, rồi so với bản rule-only (không cần nhãn).
6. **Lưu về GitHub**: adapter (Git LFS, khoảng 0,26 GB), reliability, file nộp, ghi chú huấn luyện, đẩy lên một nhánh mới.

**Cần chuẩn bị**
- Runtime → Change runtime type → **A100 GPU** (tốt nhất) hoặc **L4**. **T4 không dùng được** (không có bf16, chỉ 16 GB).
- Bật **High-RAM** nếu có.
- Colab Secrets (biểu tượng chìa khoá bên trái), bật *Notebook access*:
  - `GITHUB_TOKEN`: fine-grained token, chỉ repo này, quyền *Contents: Read and write*. Cần ở bước 6.
  - `HF_TOKEN` (tuỳ chọn): token Hugging Face, chỉ cần nếu publish model merge.
  - **Không dán token vào ô code.**

**Thời gian và compute units** (ước lượng, chưa đo): tải model 5–15 phút; tạo dữ liệu 20–40 phút (CPU); train pha A
với `MAX_PER_TASK=1500` khoảng 5 giờ trên A100 (đã đo: 822 bước), lâu hơn khoảng 3 lần trên L4; đánh giá trên fold 0
khoảng 1–2 giờ. Theo dõi compute units ở góc phải trên.

Khi bị ngắt kết nối, máy ảo mới không còn thư viện và model, còn Drive vẫn giữ checkpoint và dữ liệu. Chạy lại
từ ô **1** theo thứ tự: các bước đã xong sẽ được bỏ qua, và việc train tự tiếp tục từ checkpoint cuối.
""")

md("## 1. Cấu hình")
code("""
# ---- chỉnh ở đây ----------------------------------------------------------------
REPO_URL = "https://github.com/AIVIETNAM-AIO-khaisiubeo/datamate-xiyan-.git"
BRANCH = "gpu-finetune-guide"      # nhánh có code fine-tune
PHASE = "A"                        # "A": giữ riêng fold 0 để đo; "B": train trên cả 30 shop (bản cuối)
HOLDOUT_FOLD = 0
MAX_PER_TASK = 1500                # A: thử nhanh (1500); B: 6000 nếu đủ thời gian
EPOCHS = 1
LR = 1e-4
OUT_BRANCH = f"colab-finetune-phase{PHASE.lower()}"   # nhánh GitHub nhận kết quả (không đẩy thẳng lên main)
# ---------------------------------------------------------------------------------
import os
WORK = "/content/drive/MyDrive/datamate"            # nằm trên Drive: còn nguyên khi mất kết nối
REPO = "/content/datamate"
BASE = "/content/models/qwen3-14b"
DATA = f"{WORK}/data_phase{PHASE}"
RUN = f"{WORK}/lora_phase{PHASE}"
EVAL = f"{WORK}/eval_phase{PHASE}"
os.environ["DATAMATE_CACHE"] = "/content/cache"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["USE_TF"] = "0"                # transformers không nạp TensorFlow của Colab (tránh log "cuda drivers")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
print(PHASE, DATA, RUN, "| Python", sys.version.split()[0])
""")

md("## 2. GPU, Google Drive, repo")
code("""
import subprocess
info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout.strip()
print(info or "KHÔNG CÓ GPU: Runtime -> Change runtime type -> A100 / L4")
name, mem = (info.split(",")[0].strip(), int(info.split(",")[1])) if info else ("", 0)
assert info and "T4" not in name and "V100" not in name, "Cần A100 hoặc L4 (T4/V100 không có bf16)"
QLORA = mem < 70000                # < 70 GB: QLoRA 4-bit; A100 80GB / H100: LoRA BF16
VLLM_FLAGS = "--tp 1 --max-len 8192 --gpu-util 0.90" + (" --quant fp8" if mem < 45000 else "")
print("QLoRA:", QLORA, "| vLLM flags:", VLLM_FLAGS)
""")
code("""
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(WORK, exist_ok=True)
""")
code("""
# Colab KHÔNG có sẵn git-lfs. Thiếu nó, adapter 0,26 GB bị commit như file thường và GitHub từ chối (giới hạn 100 MB).
!command -v git-lfs >/dev/null || (apt-get -qq update && apt-get -qq install -y git-lfs >/dev/null)
!git lfs version
if not os.path.isdir(REPO):
    !git clone --branch {BRANCH} {REPO_URL} {REPO}
else:
    # lấy đúng bản mới nhất của BRANCH, kể cả khi lần trước ô 8 đã tạo commit cục bộ trên nhánh khác
    # (-f: bỏ thay đổi cục bộ trên file của repo; mọi kết quả nằm trên Drive)
    !cd {REPO} && git fetch -q origin && git checkout -q -f -B {BRANCH} origin/{BRANCH}
!cd {REPO} && git lfs install --local && git log --oneline -1
""")

md("## 3. Cài thư viện và tải Qwen3-14B")
code("""
# vLLM 0.12.0 (suy luận; hỗ trợ Python 3.10-3.13 như Colab hiện nay) kéo theo PyTorch 2.9.0; sau đó là thư viện
# huấn luyện ĐÃ GHIM (transformers 4.57.6, trl 0.24.0, peft 0.17.1). Không dùng transformers 5.x / trl 1.x.
# Mất khoảng 5-10 phút. Pip có thể in cảnh báo xung đột với gói sẵn có của Colab (TensorFlow, ...): bỏ qua được.
!pip install -q "vllm==0.12.0"
!cd {REPO} && pip install -q -r requirements.txt -r requirements-train.txt
!python -c "import torch, transformers, trl, peft, vllm; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| vllm', vllm.__version__)"
""")
md("""
Dòng cuối phải in `cuda True | transformers 4.57.6 | trl 0.24.0 | peft 0.17.1 | vllm 0.12.0`.
- `cuda False`: runtime không có GPU → Runtime → Change runtime type → A100 / L4.
- Nếu phiên này đã từng cài phiên bản khác (ví dụ transformers 5.x), vào **Runtime → Restart session**, rồi chạy lại ô 1,
  ô GPU, ô mount Drive và tiếp tục từ ô tải model. Thư viện đã cài vẫn còn sau khi restart session.
""")
code("""
from huggingface_hub import snapshot_download
if not os.path.exists(f"{BASE}/config.json"):
    snapshot_download("Qwen/Qwen3-14B", local_dir=BASE, ignore_patterns=["*.pth", "original/*"])
!cd {REPO} && python scripts/count_params.py {BASE}
""")

md("## 4. Tạo dữ liệu fine-tune (chỉ từ train)")
code("""
HOLD = f"--holdout-fold {HOLDOUT_FOLD}" if PHASE == "A" else ""
os.makedirs(DATA, exist_ok=True)
!cd {REPO} && python scripts/build_artifacts.py data/train | tail -1
!cd {REPO} && python scripts/build_sql_silver.py data/train | tail -1
if not os.path.exists(f"{DATA}/sft.jsonl"):
    !cd {REPO} && python training/build_sft_data.py --split data/train --out {DATA}/sft.jsonl --n-aug 2 {HOLD} | tail -2
if not os.path.exists(f"{DATA}/sql_sft/sql_sft.jsonl"):
    !cd {REPO} && python training/build_sql_sft.py --split data/train --out {DATA}/sql_sft --n-aug 1 {HOLD} | tail -2
!ls -la {DATA} {DATA}/sql_sft
""")

md("""
## 5. Huấn luyện

Chạy lâu. Nếu bị ngắt kết nối, chạy lại các ô 1 → 4 (bước đã xong được bỏ qua) rồi chạy ô này: `--resume` tiếp tục
từ checkpoint mới nhất trong `RUN` trên Drive. Khi xong, adapter bf16 nằm ở `RUN/adapter_bf16`.
""")
code("""
# Kiểm tra nhanh (~1 phút) toàn bộ đường train bằng một model Qwen3 tí hon: lỗi phiên bản thư viện lộ ra ở đây,
# trước khi nạp 30 GB trọng số. Phải in "SMOKE TEST OK".
!cd {REPO} && python training/smoke_test.py --base {BASE} --data {DATA}/sft.jsonl {DATA}/sql_sft/sql_sft.jsonl
""")
code("""
QL = "--qlora" if QLORA else ""
!cd {REPO} && python training/sft_lora.py --model {BASE} --data {DATA}/sft.jsonl {DATA}/sql_sft/sql_sft.jsonl \\
    --out {RUN} {QL} --epochs {EPOCHS} --lr {LR} --max-per-task {MAX_PER_TASK} --max-len 8192 --resume
if os.path.exists(f"{RUN}/adapter_bf16/adapter_model.safetensors"):
    !cd {REPO} && python scripts/count_params.py {BASE} {RUN}/adapter_bf16
else:
    print("Chưa có adapter: quá trình train ở trên chưa xong hoặc bị lỗi. Đọc thông báo lỗi phía trên.")
""")

md("""
## 6. Đánh giá trên 6 shop giữ riêng (pha A)

**Trước tiên: Runtime → Restart session** để giải phóng GPU sau khi train. Sau đó chạy lại ô 1 và các ô của mục 2
(GPU, Drive, repo), rồi chạy tiếp từ đây. Thư viện và model vẫn còn trong phiên.

**Vì sao không so PA trên câu hỏi train gốc?** Luật của G1 được viết trong lúc đọc train, nên trên câu hỏi train G1
trả lời đúng gần hết. Lần chạy trước cho `rule = base = lora = 1.0000`, và con số đó không phân biệt được ba cấu hình.
Mục này đo hai thứ:

1. **PA trên bản nhiễu** của 6 shop giữ riêng (`scripts/make_noisy_split.py`). Vẫn là các câu hỏi train gốc, chỉ thêm
   bỏ dấu, gõ Telex sai, viết tắt; dữ liệu và đáp án giữ nguyên. Con số này kiểm tra rằng thêm LLM **không làm hỏng**
   các câu G1 đã đúng. G1 vẫn mạnh ở đây (khoảng 0,98), nên chênh lệch giữa các cấu hình sẽ nhỏ.
2. **LLM-only**: chấm riêng G2, G3, G4, G_select (bỏ G1 ra). Đây là **chỉ số chính để so `base` với `lora`**, tức xem
   fine-tune có làm các generator LLM tốt hơn không.

G4 (XiYan SQL) được chạy trên **mọi** câu (`DATAMATE_G4_SCOPE=all`). Lần trước nó chỉ chạy khi G1 không trả lời được,
nên không có số đo nào cho G4. Thời gian (ước lượng, chưa đo): khoảng 1–2 giờ cho cả hai cấu hình LLM trên A100.
""")
md("""
**Cập nhật sau lần nộp 0,777** (bản LoRA thấp hơn bản rule-only). Code mới sửa ba lỗi:
1. Bộ chọn tính G2 và G3 (cùng một LLM) là hai phiếu, nên chúng thay 73 đáp án G1 không bị đánh dấu nghi ngờ. Giờ mọi
   route của cùng một LLM (G2, G3, G4, G_select) là **một** phiếu, và LLM **không được thay** một đáp án G1 không bị
   đánh dấu: kết quả là đáp án G1, hoặc bỏ trống nếu LLM phản đối mạnh.
2. `unanswerable` của riêng LLM không còn thay đáp án số của G1. Độ tin cậy của các khoá `unanswerable` co về 0,5
   thay vì 0,9 (train chỉ có 21 câu loại này).
3. G4 không đọc được SQL của adapter (adapter viết `SELECT ...` rồi đóng ```` ``` ````, và dấu này bị giữ lại trong SQL),
   nên lần trước adapter không có ứng viên G4 nào. Đáp án SQL giờ cần ít nhất 2 lần thực thi cho cùng kết quả.

Ô dưới tự chạy lại `noisy_lora`, vì G4 giờ chạy được. `noisy_rule` và `noisy_base` được giữ: log cũ vẫn chấm lại được.
**Chỉ muốn thử bước 1, không dùng GPU:** bỏ qua ô dưới, chạy ô validate (ô tiếp theo), rồi ô 7a.
""")
code("""
import re, json, shutil
os.makedirs(EVAL, exist_ok=True)
NOISY = f"/content/noisy_fold{HOLDOUT_FOLD}"   # đĩa máy Colab (Drive không tạo được symlink); seed cố định nên tạo lại y hệt
if not os.path.exists(f"{NOISY}/questions.jsonl"):
    !cd {REPO} && python scripts/make_noisy_split.py data/train {NOISY} --fold {HOLDOUT_FOLD}
LLM_EVAL = "DATAMATE_G4_SCOPE=all DATAMATE_G4_MAX=1000"
def run(tag, extra, env="", version=""):
    d = f"{EVAL}/{tag}"
    cmd = f"cd {REPO} && {env} python -m datamate.predict --data {NOISY} {extra} --out {d}/submission.jsonl"
    sig = cmd + (f"  # {version}" if version else "")
    done = f"{d}/cmd.txt"                      # chạy lại khi lệnh (hoặc version) đổi; bỏ qua nếu đã chạy xong đúng như vậy
    if not (os.path.exists(done) and open(done).read() == sig):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        print("đang chạy", tag, "...")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        open(f"{d}/predict.log", "w").write(r.stderr)
        if r.returncode != 0:
            print(r.stderr[-3000:]); raise RuntimeError(f"{tag}: predict lỗi, xem log ở trên")
        open(done, "w").write(sig)
    m = re.findall(r"PA=([-0-9.]+) (.*)", open(f"{d}/predict.log").read())
    print(f"{tag:12s} PA = {m[-1][0]}  {m[-1][1]}" if m else f"{tag}: không thấy PA trong log")
if PHASE == "A":
    run("noisy_rule", "--llm none")
    run("noisy_base", f"--llm vllm --model {BASE} {VLLM_FLAGS}", LLM_EVAL)
    run("noisy_lora", f"--llm vllm --model {BASE} --lora {RUN}/adapter_bf16 {VLLM_FLAGS}", LLM_EVAL,
        version="g4-sql-extract-fix")         # lần trước G4 của adapter không đọc được SQL nào
""")
code("""
# Chấm từng generator, và hiệu chuẩn độ tin cậy G2-G4 / G_select trên shop chưa thấy (chỉ dùng nhãn train).
# reliability_phaseA.json = reliability gốc của repo (giữ G1_rule) + phần LLM vừa đo. Lần chạy trước ghi đè mất G1_rule,
# nên file cũ bị xoá và tạo lại.
import re, json, shutil
REL = f"{WORK}/reliability_phaseA.json"
NOISY = f"/content/noisy_fold{HOLDOUT_FOLD}"          # (ô này chạy được cả khi bỏ qua ô chạy ở trên: chỉ cần CPU)
if PHASE == "A" and not os.path.exists(f"{NOISY}/questions.jsonl"):
    !cd {REPO} && python scripts/make_noisy_split.py data/train {NOISY} --fold {HOLDOUT_FOLD}
if PHASE == "A":
    !cd {REPO} && git checkout -- artifacts/reliability.json
    if os.path.exists(REL):
        os.remove(REL)
    print("===== base")
    !cd {REPO} && python scripts/validate.py {EVAL}/noisy_base/logs {NOISY}/answers.jsonl --folds 3 --out {EVAL}/validate_base | grep -A12 "per generator"
    print("===== lora")
    !cd {REPO} && python scripts/validate.py {EVAL}/noisy_lora/logs {NOISY}/answers.jsonl --folds 3 --out {EVAL}/validate_lora --write-reliability {REL} | grep -A14 "per generator"
    rel = json.load(open(REL, encoding="utf-8"))
    assert "G1_rule" in rel, "reliability_phaseA.json thiếu G1_rule"
    print("reliability_phaseA.json có:", sorted(rel))
""")
code("""
def _pa(tag):
    p = f"{EVAL}/{tag}/predict.log"
    m = re.findall(r"PA=([-0-9.]+)", open(p).read()) if os.path.exists(p) else []
    return float(m[-1]) if m else None
def _llm(tag):
    p = f"{EVAL}/validate_{tag}/metrics.json"
    return json.load(open(p)).get("llm_only", {}) if os.path.exists(p) else {}
f4 = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "-"
RESULTS = {}
if PHASE == "A":
    print(f"{'cấu hình':9s} {'PA bản nhiễu':>13s} | {'LLM-only PA':>12s} {'coverage':>9s} {'precision':>10s}")
    for t in ("rule", "base", "lora"):
        RESULTS[t] = {"PA_noisy": _pa(f"noisy_{t}"), "llm_only": _llm(t)}
        lo = RESULTS[t]["llm_only"]
        print(f"{t:9s} {f4(RESULTS[t]['PA_noisy']):>13s} | {f4(lo.get('PA')):>12s} {f4(lo.get('coverage')):>9s} {f4(lo.get('precision')):>10s}")
    json.dump(RESULTS, open(f"{EVAL}/results.json", "w"), indent=1)
    r_, b_, l_ = (RESULTS[t] for t in ("rule", "base", "lora"))
    no_harm = None not in (l_["PA_noisy"], r_["PA_noisy"]) and l_["PA_noisy"] >= r_["PA_noisy"]
    better = "PA" in l_["llm_only"] and "PA" in b_["llm_only"] and l_["llm_only"]["PA"] > b_["llm_only"]["PA"]
    print()
    print("không làm hỏng G1 (lora >= rule trên bản nhiễu):", no_harm)
    print("fine-tune giúp LLM (LLM-only: lora > base):     ", better)
    print("=> DÙNG adapter" if no_harm and better else "=> CHƯA nên dùng adapter: xem hướng dẫn bên dưới")
""")
code("""
# (Tuỳ chọn, chỉ CPU) Chấm lại các lần chạy cũ trên câu hỏi gốc (EVAL/base, EVAL/lora), nếu có. Các lần đó G4 không
# chạy, nên chỉ có số đo cho G2, G3.
for t in ("base", "lora"):
    if os.path.isdir(f"{EVAL}/{t}/logs"):
        print("=====", t, "(câu hỏi gốc của fold giữ riêng)")
        !cd {REPO} && python scripts/validate.py {EVAL}/{t}/logs data/train/answers.jsonl --folds 3 --out {EVAL}/validate_clean_{t} | grep -A10 "per generator"
""")

md("""
**Đọc bảng per generator:** `precision` là tỉ lệ đúng khi generator đó trả lời. `G1 silent` là các câu G1 không trả
lời được, hoặc trả lời nhưng tự đánh dấu là nghi ngờ; đây chính là phần LLM có thể thêm điểm.

**Quyết định:**
- Cả hai dòng kiểm tra đều `True`: dùng adapter. Nộp file ở ô 7. Có thể làm **pha B**: đặt `PHASE = "B"` (và
  `MAX_PER_TASK` lớn hơn nếu đủ unit) rồi chạy lại notebook, để có adapter học trên cả 30 shop. Pha B bỏ qua mục 6 và
  dùng `reliability_phaseA.json`.
- `LLM-only: lora > base` sai: fine-tune chưa giúp được. Thử `LR = 5e-5` hoặc `MAX_PER_TASK` khác, xoá thư mục `RUN`
  trên Drive rồi train lại.
- `lora >= rule` sai: LLM đang làm hỏng một vài câu G1 đã đúng. Xem `EVAL/validate_lora/errors.jsonl`.

Bản nhiễu vẫn được làm từ train, nên nó chỉ là ước lượng. **Điểm thật duy nhất trên public là leaderboard.**
""")

md("""
## 7. Tạo `submission.jsonl` cho public test bằng model fine-tune

- **7a (chỉ CPU, khoảng 1 phút):** áp bộ chọn mới lên lần chạy public đã có, không chạy lại LLM. Kết quả là file nộp
  của **bước 1** (bộ chọn mới; lần chạy cũ không có G4). Dòng `--legacy` phải in `same_as_logged_final 1000`: tức là
  replay dựng lại đúng lần chạy cũ, nên file mới chỉ khác ở những chỗ bộ chọn mới quyết định khác.
- **7b (GPU, khoảng 15 phút):** chạy lại toàn bộ với code mới (bước 1 + 2: bộ chọn mới và G4 đã sửa). Ô này tự chạy
  lại khi reliability hoặc code thay đổi.

Mỗi file nên nộp **một lần** lên leaderboard, và so với bản rule-only. Đừng thử đi thử lại theo điểm public.
""")
code("""
# 7a. Bộ chọn mới trên lần chạy public trước (không cần GPU)
PUBD = f"{WORK}/public_phase{PHASE}"
REL = f"{WORK}/reliability_phaseA.json"
REPLAY = f"{WORK}/public_phase{PHASE}_replay/submission.jsonl"
assert os.path.exists(REL), "Chưa có reliability_phaseA.json: chạy ô validate ở mục 6 trước"
if os.path.isdir(f"{PUBD}/logs"):
    OLD_REL = f"{PUBD}/reliability_used.json" if os.path.exists(f"{PUBD}/reliability_used.json") else REL
    print("== kiểm tra: bộ chọn cũ trên log cũ phải ra đúng các đáp án đã ghi")
    !cd {REPO} && python scripts/replay_selector.py {PUBD}/logs --data data/public_test --reliability {OLD_REL} --legacy
    print("== bộ chọn mới")
    !cd {REPO} && python scripts/replay_selector.py {PUBD}/logs --data data/public_test --reliability {REL} --out {REPLAY} --compare submissions/public_test_rule_only.jsonl
    !cd {REPO} && python scripts/check_submission.py {REPLAY} data/public_test/questions.jsonl
else:
    print("Chưa có lần chạy public nào ở", PUBD, "- chạy 7b")
""")
code("""
# 7b. Chạy lại public với code mới (GPU)
PUB = f"{PUBD}/submission.jsonl"
rel_now = open(REL, encoding="utf-8").read()
head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
used, ver = f"{PUBD}/reliability_used.json", f"{PUBD}/code_version.txt"
fresh = (os.path.exists(PUB) and os.path.exists(used) and open(used, encoding="utf-8").read() == rel_now
         and os.path.exists(ver) and open(ver).read() == head)
if not fresh:
    # Tạo lại khi reliability hoặc code thay đổi. Lần chạy trước được giữ ở *_prev (chạy 7a trước nếu muốn replay nó).
    if os.path.isdir(PUBD):
        shutil.rmtree(PUBD + "_prev", ignore_errors=True); shutil.move(PUBD, PUBD + "_prev")
    os.makedirs(PUBD)
    shutil.copy(REL, f"{REPO}/artifacts/reliability.json")
    r = subprocess.run(f"cd {REPO} && python -m datamate.predict --data data/public_test --llm vllm --model {BASE} "
                       f"--lora {RUN}/adapter_bf16 {VLLM_FLAGS} --out {PUB}", shell=True, capture_output=True, text=True)
    open(f"{PUBD}/predict.log", "w").write(r.stderr)
    print("\\n".join(l for l in r.stderr.splitlines() if "[datamate]" in l))
    if r.returncode != 0:
        print(r.stderr[-3000:]); raise RuntimeError("predict trên public lỗi, xem log ở trên")
    open(used, "w", encoding="utf-8").write(rel_now)
    open(ver, "w").write(head)
!cd {REPO} && python scripts/check_submission.py {PUB} data/public_test/questions.jsonl
""")
code("""
# So với bản rule-only (submissions/public_test_rule_only.jsonl), không cần nhãn: những câu khác nhau là chỗ điểm
# leaderboard có thể đổi. Với code mới, số câu "changed" (LLM thay đáp án G1) phải rất nhỏ.
# Mỗi khác biệt ghi vào diff_vs_rule_only.jsonl để đọc bằng mắt. Đừng sửa luật / prompt theo từng câu public.
!cd {REPO} && python scripts/compare_submissions.py submissions/public_test_rule_only.jsonl {PUB} --questions data/public_test/questions.jsonl --logs {PUBD}/logs --out {PUBD}/diff_vs_rule_only.jsonl
""")

md("""
## 8. Lưu về GitHub (nhánh mới, adapter qua Git LFS)

Đẩy lên **nhánh `OUT_BRANCH`**, không phải `main`. Bạn tự xem lại rồi merge trên GitHub. Token đọc từ Colab Secrets
và không bao giờ in ra màn hình. Mỗi lần đẩy dùng khoảng 0,26 GB dung lượng Git LFS (gói miễn phí của GitHub: 1 GB
lưu trữ và 1 GB băng thông mỗi tháng).

Ô này dừng lại kèm thông báo lỗi nếu có bước nào hỏng. Nó kiểm tra adapter **thật sự** đi qua Git LFS trước khi đẩy,
và sau khi đẩy thì kiểm tra nhánh đã có trên GitHub. Nhánh luôn được tạo lại từ `origin/BRANCH`, nên commit hỏng của
lần trước (adapter commit như file thường, bị GitHub từ chối) bị bỏ đi. Nếu `OUT_BRANCH` đã có trên GitHub, kết quả
được đẩy lên một nhánh mới có thêm ngày giờ, không ghi đè nhánh cũ.

Repo hiện ở chế độ **public**: đọc lại luật cuộc thi về việc công bố dữ liệu và weights học từ dữ liệu trước khi đẩy.
""")
code("""
import json, shutil, datetime
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
assert token, "Chưa có secret GITHUB_TOKEN (biểu tượng chìa khoá bên trái, bật Notebook access)"
assert shutil.which("git-lfs"), "Chưa cài git-lfs: chạy lại ô clone ở mục 2"
env = dict(os.environ, GITHUB_TOKEN=token, GIT_TERMINAL_PROMPT="0")
HELPER = 'credential.helper=!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
def git(*args, quiet=False, check=True):
    r = subprocess.run(["git", "-C", REPO, "-c", "credential.helper=", "-c", HELPER, *args],
                       env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(token, "***").strip()
    if out and not quiet:
        print(out[-2000:])
    if check and r.returncode != 0:
        if quiet:
            print(out[-2000:])
        raise RuntimeError(f"git {args[0]} lỗi (mã {r.returncode}), xem thông báo ở trên")
    return r

# 1. nhánh sạch từ origin/BRANCH; không ghi đè một nhánh kết quả đã có trên GitHub
git("lfs", "install", "--local", quiet=True)
git("fetch", "-q", "origin")
out_branch = OUT_BRANCH
if git("ls-remote", "--heads", "origin", out_branch, quiet=True).stdout.strip():
    out_branch = f"{OUT_BRANCH}-{datetime.datetime.now():%Y%m%d-%H%M}"
    print(f"{OUT_BRANCH} đã có trên GitHub, đẩy lên nhánh mới {out_branch}")
git("checkout", "-q", "-f", "-B", out_branch, f"origin/{BRANCH}")

# 2. kết quả: adapter, reliability, file nộp, ghi chú huấn luyện
ad_src, ad_dst = f"{RUN}/adapter_bf16", f"{REPO}/models/adapter"
assert os.path.exists(f"{ad_src}/adapter_model.safetensors"), f"Không thấy adapter ở {ad_src}"
shutil.rmtree(ad_dst, ignore_errors=True)
shutil.copytree(ad_src, ad_dst)
REL = f"{WORK}/reliability_phaseA.json"
if os.path.exists(REL):
    shutil.copy(REL, f"{REPO}/artifacts/reliability.json")
PUB = f"{WORK}/public_phase{PHASE}/submission.jsonl"
os.makedirs(f"{REPO}/submissions", exist_ok=True)
if os.path.exists(PUB):
    shutil.copy(PUB, f"{REPO}/submissions/public_test_lora_phase{PHASE.lower()}.jsonl")
REPLAY = f"{WORK}/public_phase{PHASE}_replay/submission.jsonl"
if os.path.exists(REPLAY):
    shutil.copy(REPLAY, f"{REPO}/submissions/public_test_lora_phase{PHASE.lower()}_replay.jsonl")
res_path = f"{WORK}/eval_phaseA/results.json"
results = json.load(open(res_path)) if os.path.exists(res_path) else {}
counts = {}
for f in (f"{DATA}/sft.jsonl", f"{DATA}/sql_sft/sql_sft.jsonl"):
    if os.path.exists(f):
        for l in open(f, encoding="utf-8"):
            t = json.loads(l)["task"]; counts[t] = counts.get(t, 0) + 1
open(f"{ad_dst}/TRAINING.md", "w", encoding="utf-8").write(f'''# LoRA adapter — phase {PHASE}

- base: Qwen/Qwen3-14B; method: {"QLoRA 4-bit" if globals().get("QLORA") else "LoRA BF16"}, rank 32, epochs {EPOCHS}, lr {LR}, max-per-task {MAX_PER_TASK}
- data: train split only{"; held-out fold " + str(HOLDOUT_FOLD) + " not used for training" if PHASE == "A" else "; all 30 shops"}
- examples per task (before the cap): {json.dumps(counts, ensure_ascii=False)}
- GPU: {globals().get("name", "?")}; date: {datetime.date.today()}
- phase A evaluation on the held-out fold {HOLDOUT_FOLD}, noisy copy (scripts/make_noisy_split.py):
  PA end-to-end and LLM generators only (G1 removed): {json.dumps(results)}
''')

# 3. kiểm tra trước khi đẩy: adapter phải là con trỏ LFS, không file nào quá giới hạn 100 MB của GitHub
git("add", "-A", "models/adapter", "artifacts/reliability.json", "submissions")
attr = git("check-attr", "filter", "--", "models/adapter/adapter_model.safetensors", quiet=True).stdout
assert "lfs" in attr, f".gitattributes không đưa adapter qua LFS: {attr}"
staged = git("diff", "--cached", "--name-only", quiet=True).stdout.split()
for p in staged:
    size = int(git("cat-file", "-s", f":{p}", quiet=True).stdout)
    assert size < 95_000_000, f"{p} có {size / 1e6:.0f} MB trong git: GitHub sẽ từ chối (git-lfs chưa hoạt động?)"
ptr = git("cat-file", "-p", ":models/adapter/adapter_model.safetensors", quiet=True).stdout
assert ptr.startswith("version https://git-lfs"), "adapter_model.safetensors chưa phải con trỏ Git LFS"
print("đã kiểm tra:", len(staged), "file; adapter đi qua Git LFS")

# 4. commit + push (Git LFS tải weights lên trong lúc push), rồi xác nhận nhánh đã có trên GitHub
if git("diff", "--cached", "--quiet", quiet=True, check=False).returncode:
    git("commit", "-q", "-m", f"Colab fine-tune phase {PHASE}: LoRA adapter, reliability, public submission")
git("lfs", "ls-files")
git("push", "origin", f"HEAD:refs/heads/{out_branch}")
assert git("ls-remote", "--heads", "origin", out_branch, quiet=True).stdout.strip(), "Push chưa tới GitHub"
web = REPO_URL[:-4] if REPO_URL.endswith(".git") else REPO_URL
print(f"XONG: {web}/tree/{out_branch}")
""")

md("""
## 9. (Tuỳ chọn) Merge thành một model và đưa lên Hugging Face Hub

Chỉ cần nếu muốn dùng `Dockerfile.finetuned` (model khoảng 30 GB). `Dockerfile.lora` đã dùng được adapter ở trên mà
không cần bước này. Cần khoảng 60 GB đĩa và runtime High-RAM. Repo trên Hub mặc định **private**.
""")
code("""
from google.colab import userdata
HF_REPO = "<hf-username>/datamate-qwen3-14b-ft"     # đổi thành tên của bạn
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!cd {REPO} && python training/sft_lora.py --merge --model {BASE} --adapter {RUN}/adapter_bf16 --out /content/merged
!cd {REPO} && python scripts/count_params.py /content/merged && python scripts/publish_hf.py /content/merged {HF_REPO}
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
out = os.path.join(ROOT, "notebooks", "colab_finetune.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print("wrote", out, len(cells), "cells")
