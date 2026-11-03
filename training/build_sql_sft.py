"""Multi-task / multi-format SFT data for the SQL generators and the selection model (TRAIN split only).

Built from execution-verified silver SQL (scripts/build_sql_silver.py) with the inference-time prompts, on
the schema the Schema Filter gives at inference (S1 when it keeps every gold column, else S2, else full):

  nl2sql_<fmt>        (Q, E, S) -> SQL in format base / cte / alias                 XiYan Fig. 2(a) + Fig. 3
  question_inference  (SQL, E, S) -> Q                                              Fig. 2(b)
  evidence_inference  (Q, SQL, candidate rulebook rows) -> the rows the SQL relies on Fig. 2(c)
  self_refine         (Q, E, S, near-miss SQL, its execution result) -> correct SQL  Fig. 2(d)
  select              (Q, E, S_union, candidates + execution results) -> choice      §III-D, listwise, near-miss
                      negatives from controlled modifications, candidates de-formalised, random order

Question variants: the original question plus --n-aug noisy rewrites (datamate/augment.py), mimicking the
public/private style.  Outputs (both formats):
  <out>/sql_sft.jsonl           {"task", "system", "user", "target"}         -> training/sft_lora.py
  <out>/xiyan_sql_sft.json      [{"id", "conversations", "sql_type"}]        -> alibaba/XiYan-SQLTraining

Parameter budget: train ONE model on all tasks and formats (the format is chosen by the instruction in the
prompt, SQL_FORMAT_HINTS); merged it keeps 14.8B parameters.  Separate merged models per format would
multiply the parameter count and break the 15B rule; separate LoRA adapters must be counted
(scripts/count_params.py).

    python training/build_sql_sft.py --split data/train --out outputs/sql_sft --n-aug 2
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATAMATE_CACHE", os.path.join(ROOT, ".cache"))

from datamate.augment import augment  # noqa: E402
from datamate.ingest.builder import load_biz  # noqa: E402
from datamate.llm import prompts as P  # noqa: E402
from datamate.llm.evidence import EvidenceRetriever, rulebook_rows  # noqa: E402
from datamate.llm.generators import sql_result_to_value  # noqa: E402
from datamate.llm.mschema import build_mschema, build_views  # noqa: E402
from datamate.llm.schema_filter import SchemaFilter, column_recall  # noqa: E402
from datamate.llm.sql_mutations import mutate_all  # noqa: E402
from datamate.llm.sql_pipeline import SQLRunner, deformalize, exec_info  # noqa: E402
from datamate.nlu.linker import Linker  # noqa: E402
from datamate.scorer import score_one  # noqa: E402

# rulebook rows (first cell prefix) a correct query of each intent relies on
INTENT_EVIDENCE = {
    "revenue": ["Đơn tính vào doanh thu", "Doanh thu một đơn"], "monthly_list": ["Đơn tính vào doanh thu"],
    "order_count": ["Số đơn", "Đơn tính vào doanh thu"], "aov": ["Giá trị trung bình một đơn", "Doanh thu một đơn"],
    "cancel_count": ["Đơn bị huỷ"], "cancel_rate": ["Tỉ lệ đơn huỷ trong kỳ"],
    "top_products": ["Bán chạy", "Mặt hàng"], "top_customers": ["Xác định khách"],
    "best_day": ["Đơn tính vào doanh thu"], "share": ["Nhóm hàng", "Đơn tính vào doanh thu"],
    "purchase_total": ["Tiền nhập hàng trong kỳ"],
}
TIME_EVIDENCE = [("tuan", "Tuần"), ("thang nay", "Tháng này"), ("dau nam", "Từ đầu năm"), ("ngay gan nhat", "N ngày gần nhất"),
                 ("hom qua", "Ngày X"), ("hom nay", "Ngày X"), ("tet", "Tết năm Y")]


def relevant_rows(rows, intent, nq):
    keys = list(INTENT_EVIDENCE.get(intent, []))
    keys += [k for w, k in TIME_EVIDENCE if w in nq]
    return [r for r in rows if any(r.split("] ", 1)[-1].startswith(k) for k in keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--silver", default=os.path.join(ROOT, "artifacts", "sql_silver.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "sql_sft"))
    ap.add_argument("--n-aug", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-fold", type=int, default=None, help="leave the shops of this fold out (evaluation)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-shots", action="store_true", help="train without few-shot SQL in the prompt")
    a = ap.parse_args()
    from datamate.agent import load_sql_bank
    from datamate.folds import fold_shops
    from datamate.llm.sql_pipeline import MultiSQLGenerator
    rng = random.Random(a.seed)
    rb_text = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    rows_rb = rulebook_rows(rb_text)
    from datamate.textnorm import normalize_question
    items = [json.loads(l) for l in open(a.silver, encoding="utf-8")]
    by = collections.defaultdict(list)
    for it in items:
        by[it["business_id"]].append(it)
    all_shops = {json.loads(l)["business_id"] for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")}
    holdout = fold_shops(all_shops, a.holdout_fold, a.folds) if a.holdout_fold is not None else set()
    if holdout:
        print("held-out shops (not in the training data):", sorted(holdout))
    bank = None if a.no_shots else load_sql_bank(a.silver)
    shot_gen = MultiSQLGenerator(None, bank)          # only its few-shot selection: the same examples G4 sees
    os.makedirs(a.out, exist_ok=True)
    recs, stat = [], collections.Counter()

    def add(task, system, user, target):
        recs.append({"task": task, "system": system, "user": user, "target": target})
        stat[task] += 1

    for bid, lst in sorted(by.items()):
        if bid in holdout:
            continue
        if bank is not None:
            bank.exclude = {bid} | holdout
        b = load_biz(os.path.join(a.split, "data", bid), bid)
        V = build_views(b)
        ms = build_mschema(bid, V)
        sf = SchemaFilter(ms, V, Linker(b))
        ev = EvidenceRetriever(rb_text, getattr(getattr(b, "readme", None), "text", "") or "")
        runner = SQLRunner(V)
        for it in lst:
            q, gold = it["question"], it["columns"]
            evid = ev.evidence(q, it["as_of_date"])
            res = sf.run(q, evid)
            cols = next((s["columns"] for s in res.schemas if s["valid"] and column_recall(gold, s["columns"]) == 1.0), None)
            schema = ms.to_mschema(cols, extra_examples=res.values) if cols else ms.to_mschema(extra_examples=res.values)
            ref = {"type": it["answer_type"], "answer": it["answer"]}
            variants = [q] + [augment(q, rng, 0.6) for _ in range(a.n_aug)]
            base = it["sql"].get("base") or next(iter(it["sql"].values()))
            # (a) text-to-SQL, one record per verified format and question variant
            for fmt, sql in it["sql"].items():
                for v in variants:
                    shots = shot_gen._shots(v, fmt) if bank is not None else []
                    add(f"nl2sql_{fmt}", P.NL2SQL_SYSTEM, P.nl2sql_user(v, schema, evid, fmt, shots), sql + "\n```")
            # (b) question inference
            add("question_inference", "Bạn là chuyên gia SQL. Đọc 【Schema】, 【Thông tin tham khảo】 và 【SQL】, viết câu hỏi tiếng "
                "Việt tự nhiên mà SQL này trả lời.", f"【Schema】\n{schema}\n\n【Thông tin tham khảo】\n{evid}\n\n【SQL】\n{base}\n\n【Câu hỏi】", q)
            # (c) evidence inference
            rel = relevant_rows(rows_rb, it["intent"], normalize_question(q))
            if rel:
                pool = rel + rng.sample([r for r in rows_rb if r not in rel], k=min(6, len(rows_rb) - len(rel)))
                rng.shuffle(pool)
                add("evidence_inference", "Bạn là chuyên gia SQL. Chọn các luật trong 【Luật ứng viên】 mà 【SQL】 đã dùng để trả lời "
                    "【Câu hỏi】. In nguyên văn từng luật, mỗi luật một dòng.",
                    f"【Câu hỏi】\n{q}\n\n【SQL】\n{base}\n\n【Luật ứng viên】\n" + "\n".join(pool), "\n".join(rel))
            # (d) self-refine and (e) selection, from answer-changing near-miss mutations
            negs = []
            for name, m in mutate_all(base):
                try:
                    rr = runner.run(m)
                    v = sql_result_to_value(rr, it["answer_type"])
                except Exception as e:  # noqa: BLE001
                    rr, v = None, None
                    info = exec_info(None, f"{type(e).__name__}: {e}"[:200])
                else:
                    info = exec_info(rr, None)
                if v is not None and score_one({"type": it["answer_type"], "answer": v}, ref) == 1:
                    continue
                negs.append((name, m, info))
                add("self_refine", P.SELF_REFINE_SYSTEM, P.self_refine_user(q, schema, evid, m, info), base + "\n```")
            if negs:
                pos_fmt = rng.choice(list(it["sql"]))
                good = it["sql"][pos_fmt]
                cands = [(deformalize(good), exec_info(runner.run(good), None), True)]
                for name, m, info in rng.sample(negs, k=min(3, len(negs))):
                    cands.append((deformalize(m), info, False))
                rng.shuffle(cands)
                blocks = [P.candidate_block(i + 1, s, inf) for i, (s, inf, _) in enumerate(cands)]
                choice = next(i + 1 for i, c in enumerate(cands) if c[2])
                add("select", P.SELECT_SYSTEM, P.select_user(q, schema, evid, blocks), json.dumps({"choice": choice}))
        runner.close()
        print(bid, dict(stat), flush=True)
    rng.shuffle(recs)
    with open(os.path.join(a.out, "sql_sft.jsonl"), "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    xiyan = [{"id": str(i), "sql_type": r["task"],
              "conversations": [{"role": "user", "content": r["system"] + "\n\n" + r["user"]},
                                {"role": "assistant", "content": r["target"]}]} for i, r in enumerate(recs)]
    json.dump(xiyan, open(os.path.join(a.out, "xiyan_sql_sft.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", len(recs), "records:", dict(stat), "->", a.out)


if __name__ == "__main__":
    main()
