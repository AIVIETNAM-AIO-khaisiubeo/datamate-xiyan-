"""Build the train-derived artifacts used at inference time (TRAIN split only):

  artifacts/known_vocab.json    question vocabulary with entity names masked (coverage check)
  artifacts/example_bank.jsonl  few-shot bank: question -> correct rule frame / canonical rewrite
  artifacts/reliability.json    per-intent reliability of the rule generator (via calibrate.py)

    python scripts/build_artifacts.py data/train
"""
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.engine.answer import FrameExecutor  # noqa: E402
from datamate.engine.executor import Executor  # noqa: E402
from datamate.ingest.builder import load_biz  # noqa: E402
from datamate.llm.generators import render_frame  # noqa: E402
from datamate.nlu.coverage import save_vocab  # noqa: E402
from datamate.nlu.rules import RuleParser  # noqa: E402
from datamate.nlu.timeparse import LunarCalendar  # noqa: E402
from datamate.scorer import score_one  # noqa: E402
from datamate.textnorm import normalize_question, repair_fold  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "training"))
from build_sft_data import frame_to_llm_json  # noqa: E402


def main(split):
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")}
    cal = LunarCalendar(os.path.join(split, "calendar.csv"))
    by = {}
    for q in qs:
        by.setdefault(q["business_id"], []).append(q)
    vocab = set()
    ivocab = {}
    art = os.path.join(ROOT, "artifacts")
    os.makedirs(art, exist_ok=True)
    nb = 0
    with open(os.path.join(art, "example_bank.jsonl"), "w", encoding="utf-8") as fb:
        for bid, items in sorted(by.items()):
            b = load_biz(os.path.join(split, "data", bid), bid)
            rp = RuleParser(b, cal)
            for q in items:
                nq = normalize_question(q["question"])
                rp._qr = repair_fold(q["question"])
                masked = rp._mask_entities(" " + nq + " ", q["question"])
                vocab |= {t for t in masked.split() if t.isalpha() and t != "_e_"}
                today = dt.date.fromisoformat(q["as_of_date"])
                fr = rp.parse(q["question"], today)
                if fr is None:
                    continue
                ans = FrameExecutor(Executor(b, today), cal).run(fr)
                if score_one(ans.to_submission(q["id"]), refs[q["id"]]) != 1:
                    continue
                ivocab.setdefault(fr.intent, set()).update(t for t in masked.split() if t.isalpha() and t != "_e_")
                item = {"q": q["question"], "nq": nq, "intent": fr.intent,
                        "frame_llm": json.dumps(frame_to_llm_json(fr, b), ensure_ascii=False),
                        "canon": render_frame(fr, b.product_name) or q["question"]}
                fb.write(json.dumps(item, ensure_ascii=False) + "\n")
                nb += 1
            print(bid, len(vocab), nb, flush=True)
    vocab -= {"_e_"}
    save_vocab(vocab, os.path.join(art, "known_vocab.json"))
    with open(os.path.join(art, "intent_vocab.json"), "w", encoding="utf-8") as fh:
        json.dump({k: sorted(v) for k, v in ivocab.items()}, fh, ensure_ascii=False)
    print("vocab", len(vocab), "bank", nb)


if __name__ == "__main__":
    main(sys.argv[1])
