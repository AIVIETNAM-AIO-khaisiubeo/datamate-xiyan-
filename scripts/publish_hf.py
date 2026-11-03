"""Upload a model folder (LoRA adapter or merged model) to the Hugging Face Hub, private by default.

GitHub is not a place for multi-GB weights (100 MB per file, 1 GB free Git LFS): keep the ~0.26 GB adapter in the
repo (models/adapter, Git LFS) and put a merged ~30 GB model, if you make one, on the Hub.

The token is read from the environment (HF_TOKEN; on Colab: Secrets panel -> HF_TOKEN, "Notebook access" on).
It is never printed or written anywhere.

    python scripts/publish_hf.py outputs/merged <hf-user>/datamate-qwen3-14b-merged [--public]
    huggingface-cli download <hf-user>/datamate-qwen3-14b-merged --local-dir /models/merged    # later, to use it
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("repo_id", help="<user or org>/<name>")
    ap.add_argument("--public", action="store_true", help="default: private repository")
    ap.add_argument("--message", default="DataMate-XiYan fine-tuned weights (trained on the train split only)")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set (Colab: add it in the Secrets panel)")
    if not os.path.isdir(a.folder):
        sys.exit(f"{a.folder} is not a directory")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo_id, private=not a.public, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=a.folder, repo_id=a.repo_id, repo_type="model", commit_message=a.message,
                      ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "rng_state*", "scheduler*"])
    print(f"uploaded {a.folder} -> https://huggingface.co/{a.repo_id} ({'public' if a.public else 'private'})")


if __name__ == "__main__":
    main()
