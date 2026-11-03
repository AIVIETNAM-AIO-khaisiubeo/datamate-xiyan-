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
4. **Đánh giá** (pha A): so rule-only, model gốc và model fine-tune trên 6 shop giữ riêng, rồi hiệu chuẩn độ tin cậy.
5. **Tạo `submission.jsonl`** cho public test bằng model đã fine-tune.
6. **Lưu về GitHub**: adapter (Git LFS, khoảng 0,26 GB), reliability, file nộp, ghi chú huấn luyện, đẩy lên một nhánh mới.

**Cần chuẩn bị**
- Runtime → Change runtime type → **A100 GPU** (tốt nhất) hoặc **L4**. **T4 không dùng được** (không có bf16, chỉ 16 GB).
- Bật **High-RAM** nếu có.
- Colab Secrets (biểu tượng chìa khoá bên trái), bật *Notebook access*:
  - `GITHUB_TOKEN`: fine-grained token, chỉ repo này, quyền *Contents: Read and write*. Cần ở bước 6.
  - `HF_TOKEN` (tuỳ chọn): token Hugging Face, chỉ cần nếu publish model merge.
  - **Không dán token vào ô code.**

**Thời gian và compute units** (ước lượng, chưa đo): tải model 5–15 phút; tạo dữ liệu 20–40 phút (CPU); train pha A
với `MAX_PER_TASK=1500` khoảng 3–5 giờ trên A100, lâu hơn khoảng 3 lần trên L4; đánh giá 3 cấu hình trên fold 0
khoảng 1 giờ. Theo dõi compute units ở góc phải trên.

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
print(PHASE, DATA, RUN)
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
if not os.path.isdir(REPO):
    !git clone --branch {BRANCH} {REPO_URL} {REPO}
else:
    !cd {REPO} && git pull --ff-only
!cd {REPO} && git lfs install --local && git log --oneline -1
""")

md("## 3. Cài thư viện và tải Qwen3-14B")
code("""
# vLLM (suy luận) kéo theo PyTorch phù hợp; sau đó là thư viện huấn luyện. Mất khoảng 5 phút.
!pip install -q "vllm==0.10.1.1"
!cd {REPO} && pip install -q -r requirements.txt -r requirements-train.txt
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
QL = "--qlora" if QLORA else ""
!cd {REPO} && python training/sft_lora.py --model {BASE} --data {DATA}/sft.jsonl {DATA}/sql_sft/sql_sft.jsonl \\
    --out {RUN} {QL} --epochs {EPOCHS} --lr {LR} --max-per-task {MAX_PER_TASK} --max-len 8192 --resume
!cd {REPO} && python scripts/count_params.py {BASE} {RUN}/adapter_bf16
""")

md("""
## 6. Đánh giá trên 6 shop giữ riêng (pha A)

**Trước tiên: Runtime → Restart session** để giải phóng GPU sau khi train. Sau đó chạy lại ô 1, ô mount Drive, rồi
chạy tiếp từ đây (thư viện và model vẫn còn trong phiên).

Ba cấu hình được chạy trên cùng 6 shop mà model chưa từng thấy: G1 rule-only, Qwen3-14B gốc, và Qwen3-14B + adapter.
""")
code("""
import re, glob
os.makedirs(EVAL, exist_ok=True)
def run(tag, extra):
    out = f"{EVAL}/{tag}/submission.jsonl"
    if not os.path.exists(out):
        r = subprocess.run(f"cd {REPO} && python -m datamate.predict --data data/train --fold {HOLDOUT_FOLD} {extra} --out {out}",
                           shell=True, capture_output=True, text=True)
        open(f"{EVAL}/{tag}.log", "w").write(r.stderr)
    pa = re.findall(r"PA=([-0-9.]+)", open(f"{EVAL}/{tag}.log").read())
    print(f"{tag:10s} PA = {pa[-1] if pa else '?'}")
if PHASE == "A":
    run("rule", "--llm none")
    run("base", f"--llm vllm --model {BASE} {VLLM_FLAGS}")
    run("lora", f"--llm vllm --model {BASE} --lora {RUN}/adapter_bf16 {VLLM_FLAGS}")
""")
code("""
# Hiệu chuẩn độ tin cậy của các generator LLM trên shop chưa thấy (chỉ dùng nhãn train). Chỉ ghi G2-G4 / G_select.
if PHASE == "A":
    !cd {REPO} && python scripts/validate.py {EVAL}/lora/logs data/train/answers.jsonl --folds 3 \\
        --out {EVAL}/validate --write-reliability {WORK}/reliability_phaseA.json
""")

md("""
**Quyết định**: chỉ dùng adapter nếu `lora` có PA **cao hơn cả** `base` và `rule`. Nếu thấp hơn, đừng nộp nó:
thử `LR` nhỏ hơn hoặc `MAX_PER_TASK` khác, xoá thư mục `RUN` trên Drive rồi train lại. Nếu tốt hơn, có thể chuyển
sang `PHASE = "B"` (train trên cả 30 shop) và chạy lại notebook. Ở pha B, bỏ qua bước đánh giá và giữ
`reliability_phaseA.json`.
""")

md("## 7. Tạo `submission.jsonl` cho public test bằng model fine-tune")
code("""
PUB = f"{WORK}/public_phase{PHASE}/submission.jsonl"
if not os.path.exists(PUB):
    !cd {REPO} && cp {WORK}/reliability_phaseA.json artifacts/reliability.json 2>/dev/null; \\
        python -m datamate.predict --data data/public_test --llm vllm --model {BASE} --lora {RUN}/adapter_bf16 {VLLM_FLAGS} --out {PUB}
!cd {REPO} && python scripts/check_submission.py {PUB} data/public_test/questions.jsonl
""")

md("""
## 8. Lưu về GitHub (nhánh mới, adapter qua Git LFS)

Đẩy lên **nhánh `OUT_BRANCH`**, không phải `main`. Bạn tự xem lại rồi merge trên GitHub. Token đọc từ Colab Secrets
và không bao giờ in ra màn hình. Mỗi lần đẩy dùng khoảng 0,26 GB dung lượng Git LFS (gói miễn phí của GitHub: 1 GB
lưu trữ và 1 GB băng thông mỗi tháng).

Repo hiện ở chế độ **public**: đọc lại luật cuộc thi về việc công bố dữ liệu và weights học từ dữ liệu trước khi đẩy.
""")
code("""
import json, shutil, datetime, tempfile
from google.colab import userdata
ad_src, ad_dst = f"{RUN}/adapter_bf16", f"{REPO}/models/adapter"
shutil.rmtree(ad_dst, ignore_errors=True)
shutil.copytree(ad_src, ad_dst)
if os.path.exists(f"{WORK}/reliability_phaseA.json"):
    shutil.copy(f"{WORK}/reliability_phaseA.json", f"{REPO}/artifacts/reliability.json")
os.makedirs(f"{REPO}/submissions", exist_ok=True)
if os.path.exists(PUB):
    shutil.copy(PUB, f"{REPO}/submissions/public_test_lora_phase{PHASE.lower()}.jsonl")
pa = {}
for tag in ("rule", "base", "lora"):
    p = f"{EVAL}/{tag}.log"
    if os.path.exists(p):
        m = re.findall(r"PA=([-0-9.]+)", open(p).read())
        pa[tag] = m[-1] if m else "?"
counts = {}
for f in (f"{DATA}/sft.jsonl", f"{DATA}/sql_sft/sql_sft.jsonl"):
    for l in open(f, encoding="utf-8"):
        t = json.loads(l)["task"]; counts[t] = counts.get(t, 0) + 1
open(f"{ad_dst}/TRAINING.md", "w", encoding="utf-8").write(f'''# LoRA adapter — phase {PHASE}

- base: Qwen/Qwen3-14B; method: {"QLoRA 4-bit" if QLORA else "LoRA BF16"}, rank 32, epochs {EPOCHS}, lr {LR}, max-per-task {MAX_PER_TASK}
- data: train split only{"; held-out fold " + str(HOLDOUT_FOLD) + " not used for training" if PHASE == "A" else "; all 30 shops"}
- examples per task (before the cap): {json.dumps(counts, ensure_ascii=False)}
- GPU: {name}; date: {datetime.date.today()}
- PA on held-out fold {HOLDOUT_FOLD} (553 questions): {json.dumps(pa)}
''')
token = userdata.get("GITHUB_TOKEN")
askpass = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
askpass.write("#!/bin/sh\\necho \\"$GITHUB_TOKEN\\"\\n"); askpass.close(); os.chmod(askpass.name, 0o700)
env = dict(os.environ, GITHUB_TOKEN=token, GIT_ASKPASS=askpass.name, GIT_TERMINAL_PROMPT="0")
def git(*args):
    r = subprocess.run(["git", "-C", REPO, *args], env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token, "***")[-1500:])
    return r.returncode
git("config", "user.name", "datamate-colab"); git("config", "user.email", "datamate-colab@users.noreply.github.com")
git("checkout", "-B", OUT_BRANCH)
git("add", "models/adapter", "artifacts/reliability.json", "submissions")
git("commit", "-m", f"Colab fine-tune phase {PHASE}: LoRA adapter, reliability, public submission")
url = "https://x-access-token@" + REPO_URL.split("https://", 1)[1]
git("push", url, f"HEAD:{OUT_BRANCH}")
os.remove(askpass.name)
""")

md("""
## 9. (Tuỳ chọn) Merge thành một model và đưa lên Hugging Face Hub

Chỉ cần nếu muốn dùng `Dockerfile.finetuned` (model khoảng 30 GB). `Dockerfile.lora` đã dùng được adapter ở trên mà
không cần bước này. Cần khoảng 60 GB đĩa và runtime High-RAM. Repo trên Hub mặc định **private**.
""")
code("""
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
