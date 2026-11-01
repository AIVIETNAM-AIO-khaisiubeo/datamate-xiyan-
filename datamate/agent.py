"""DataMate agent: orchestrates loading, the multi-generator ensemble, selection, logging.

Pipeline per business (XiYan-style, adapted to tabular QA with abstention):

  1. Ingest   : heterogeneous CSV/XLSX/SQLite -> canonical Biz (schema linking by README,
                header synonyms and value sniffing; rulebook data semantics applied once)
  2. Generate : G1 rule parser -> Frame -> executor                     (always)
                G3 LLM rewrite -> G1 on the rewrite                      (if LLM)
                G2 LLM JSON frame -> canonical text -> G1               (if LLM)
                G4 LLM text-to-SQL on M-Schema views, refined on error  (if LLM, fallback)
  3. Select   : group executed answers, calibrated vote, Chow threshold -> answer / abstain
  4. Write    : submission line (flushed immediately) + per-question log
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import traceback
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

from .engine.answer import Answer, FrameExecutor
from .engine.executor import Executor
from .ingest.builder import load_biz
from .nlu.rules import RuleParser
from .nlu.timeparse import LunarCalendar
from .rulebook import parse_rulebook
from .select.selector import Candidate, Selector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def default_config() -> dict:
    return {
        "llm": {"backend": os.environ.get("DATAMATE_LLM", "none"),
                "model_path": os.environ.get("DATAMATE_MODEL", "/models/llm"),
                "tensor_parallel_size": int(os.environ.get("DATAMATE_TP", "2")),
                "max_model_len": 16384, "gpu_memory_utilization": 0.88, "enable_thinking": False},
        "bank_path": os.path.join(ROOT, "artifacts", "example_bank.jsonl"),
        "reliability_path": os.path.join(ROOT, "artifacts", "reliability.json"),
        "tau": float(os.environ.get("DATAMATE_TAU", "0.6")),
        "time_budget_s": float(os.environ.get("DATAMATE_TIME_BUDGET", str(5.5 * 3600))),
        "use_g2": True, "use_g3": True, "use_g4": True,
        "g4_max_per_biz": 60,
    }


class DataMateAgent:
    def __init__(self, split_dir: str, config: Optional[dict] = None):
        self.split_dir = split_dir
        self.cfg = default_config()
        if config:
            self.cfg.update(config)
        rb_path = os.path.join(split_dir, "rulebook.md")
        self.rulebook_text = open(rb_path, encoding="utf-8").read() if os.path.exists(rb_path) else ""
        self.rulebook = parse_rulebook(self.rulebook_text)
        self.cal = LunarCalendar(os.path.join(split_dir, "calendar.csv"))
        self.selector = Selector(self.cfg["reliability_path"], tau=self.cfg["tau"])
        self.t_start = time.time()
        self.backend = None
        self._llm_ready = False

    # ------------------------------------------------------------------ LLM (lazy)
    def _init_llm(self):
        if self._llm_ready:
            return
        self._llm_ready = True
        from .llm.backend import make_backend
        self.backend = make_backend(self.cfg["llm"])
        if not getattr(self.backend, "available", False):
            self.backend = None
            return
        from .llm.generators import ExampleBank, FrameGenerator, RewriteGenerator, SQLGenerator
        bank = ExampleBank.load(self.cfg["bank_path"]) if os.path.exists(self.cfg["bank_path"]) else None
        digest = rulebook_digest(self.rulebook_text)
        self.g3 = RewriteGenerator(self.backend, bank)
        self.g2 = FrameGenerator(self.backend, bank, digest)
        self.g4 = SQLGenerator(self.backend, bank, digest)

    def time_left(self) -> float:
        return self.cfg["time_budget_s"] - (time.time() - self.t_start)

    # ------------------------------------------------------------------ main loop
    def run(self, questions: List[dict], out_path: str, log_dir: Optional[str] = None) -> Dict[str, dict]:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        by_biz: Dict[str, List[dict]] = OrderedDict()
        for q in questions:
            by_biz.setdefault(q["business_id"], []).append(q)
        results: Dict[str, dict] = {}
        with open(out_path, "w", encoding="utf-8") as out:
            for bid, items in by_biz.items():
                try:
                    res = self.run_business(bid, items, log_dir)
                except Exception:
                    traceback.print_exc()
                    res = {q["id"]: {"id": q["id"], "type": "abstain"} for q in items}
                for q in items:
                    line = res.get(q["id"]) or {"id": q["id"], "type": "abstain"}
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    results[q["id"]] = line
                out.flush()
        return results

    def run_business(self, bid: str, items: List[dict], log_dir: Optional[str]) -> Dict[str, dict]:
        t0 = time.time()
        biz = load_biz(os.path.join(self.split_dir, "data", bid), bid)
        rp = RuleParser(biz, self.cal)
        load_s = time.time() - t0
        cands: Dict[str, List[Candidate]] = defaultdict(list)
        logs: Dict[str, dict] = {}
        trusted = {}
        # ---------------------------------------------------------------- G1
        for q in items:
            tq = time.time()
            today = dt.date.fromisoformat(q["as_of_date"])
            log = {"id": q["id"], "business_id": bid, "question": q["question"], "as_of_date": q["as_of_date"],
                   "candidates": []}
            fr, a = self._g1(rp, biz, q["question"], today)
            if fr is not None:
                log["g1_frame"] = fr.to_json()
                cands[q["id"]].append(Candidate("G1_rule", a, fr.intent))
                trusted[fr.intent] = self.rulebook.intent_trusted(fr.intent)
                log["candidates"].append({"gen": "G1_rule", "intent": fr.intent, "answer": a.to_submission(q["id"]),
                                          "reason": a.reason})
            log["elapsed_g1_s"] = round(time.time() - tq, 4)
            logs[q["id"]] = log
        # ---------------------------------------------------------------- LLM generators
        if self.cfg["llm"].get("backend", "none") != "none" and self.time_left() > 600:
            self._init_llm()
            if self.backend is not None:
                try:
                    self._llm_stage(biz, rp, items, cands, logs)
                except Exception:
                    traceback.print_exc()
        # ---------------------------------------------------------------- select + write logs
        res = {}
        for q in items:
            ans, info = self.selector.select(cands[q["id"]], trusted)
            sub = ans.to_submission(q["id"])
            res[q["id"]] = sub
            lg = logs[q["id"]]
            lg["final"] = sub
            lg["selector"] = info
            lg["elapsed_total_s"] = round(lg.get("elapsed_g1_s", 0) + lg.get("elapsed_llm_s", 0), 4)
            lg["biz_load_s"] = round(load_s, 2)
            if log_dir:
                with open(os.path.join(log_dir, f"{q['id']}.json"), "w", encoding="utf-8") as fh:
                    json.dump(lg, fh, ensure_ascii=False, indent=1, default=str)
        return res

    def _g1(self, rp: RuleParser, biz, question: str, today: dt.date):
        try:
            fr = rp.parse(question, today)
        except Exception as e:
            return None, Answer("abstain", reason=f"parse error {e}")
        if fr is None:
            return None, Answer("abstain")
        fe = FrameExecutor(Executor(biz, today), self.cal)
        return fr, fe.run(fr)

    # ------------------------------------------------------------------ LLM stage
    def _llm_stage(self, biz, rp, items, cands, logs):
        from .llm.generators import llm_json_to_text, run_sql, sql_result_to_value
        from .llm.mschema import build_views, render_mschema
        vocab = vocab_for(biz, rp)
        tl = time.time()
        # G3: rewrite -> G1 ------------------------------------------------
        if self.cfg.get("use_g3"):
            prompts = [self.g3.build_prompt(q["question"], q["as_of_date"], vocab) for q in items]
            outs = self.g3.run(prompts)
            for q, rw in zip(items, outs):
                today = dt.date.fromisoformat(q["as_of_date"])
                for text in rw[:1]:
                    fr, a = self._g1(rp, biz, text, today)
                    logs[q["id"]].setdefault("g3_rewrite", text)
                    if fr is not None:
                        cands[q["id"]].append(Candidate("G3_rewrite", a, fr.intent, {"text": text}))
                        logs[q["id"]]["candidates"].append({"gen": "G3_rewrite", "intent": fr.intent,
                                                             "answer": a.to_submission(q["id"]), "text": text})
        # G2: JSON frame -> canonical text -> G1 ---------------------------
        if self.cfg.get("use_g2") and self.time_left() > 900:
            prompts = [self.g2.build_prompt(q["question"], q["as_of_date"], vocab) for q in items]
            outs = self.g2.run(prompts)
            for q, js_list in zip(items, outs):
                today = dt.date.fromisoformat(q["as_of_date"])
                for js in js_list[:1]:
                    logs[q["id"]]["g2_json"] = js
                    if js.get("intent") == "unanswerable":
                        a = Answer("unanswerable", reason=js.get("reason") or "")
                        cands[q["id"]].append(Candidate("G2_frame", a, "unanswerable", {"json": js}))
                        continue
                    text = llm_json_to_text(js)
                    if not text:
                        continue
                    fr, a = self._g1(rp, biz, text, today)
                    if fr is not None:
                        cands[q["id"]].append(Candidate("G2_frame", a, fr.intent, {"json": js, "text": text}))
                        logs[q["id"]]["candidates"].append({"gen": "G2_frame", "intent": fr.intent,
                                                             "answer": a.to_submission(q["id"]), "text": text})
        # G4: SQL for questions still without a confident candidate -------
        if self.cfg.get("use_g4") and self.time_left() > 1200:
            need = []
            for q in items:
                c = cands[q["id"]]
                ans, info = self.selector.select(c)
                if ans.type == "abstain":
                    need.append(q)
            need = need[: self.cfg["g4_max_per_biz"]]
            if need:
                views = build_views(biz)
                ms = render_mschema(biz.biz_id, views)
                prompts = []
                for q in need:
                    at = _guess_type(q["question"])
                    prompts.append(self.g4.build_prompt(q["question"], q["as_of_date"], ms, at))
                outs = self.g4.run(prompts)
                retry = []
                for q, sqls, pr in zip(need, outs, prompts):
                    at = _guess_type(q["question"])
                    for sql in sqls[:1]:
                        ok, err = self._try_sql(q, sql, at, views, cands, logs)
                        if not ok:
                            retry.append((q, pr, sql, err))
                # XiYan-style refiner: one repair round with the execution error
                if retry and self.time_left() > 900:
                    rprompts = [pr + f"\n\nSQL trước:\n```sql\n{sql}\n```\nLỗi khi chạy: {err}\nHãy sửa SQL." for q, pr, sql, err in retry]
                    routs = self.g4.run(rprompts)
                    for (q, _, _, _), sqls in zip(retry, routs):
                        for sql in sqls[:1]:
                            self._try_sql(q, sql, _guess_type(q["question"]), views, cands, logs)
        dt_llm = (time.time() - tl) / max(1, len(items))
        for q in items:
            logs[q["id"]]["elapsed_llm_s"] = round(dt_llm, 4)


def _sql_value(sql, at, views):
    from .llm.generators import run_sql, sql_result_to_value
    return sql_result_to_value(run_sql(sql, views), at)


def _try_sql_impl(q, sql, at, views, cands, logs):
    try:
        v = _sql_value(sql, at, views)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"[:300]
        logs[q["id"]].setdefault("g4_errors", []).append(msg)
        return False, msg
    if v is None:
        return False, "empty result"
    a = Answer(at, v)
    cands[q["id"]].append(Candidate("G4_sql", a, None, {"sql": sql}))
    logs[q["id"]]["candidates"].append({"gen": "G4_sql", "answer": a.to_submission(q["id"]), "sql": sql})
    return True, ""


DataMateAgent._try_sql = staticmethod(_try_sql_impl)


def _guess_type(q: str) -> str:
    from .textnorm import normalize_question
    s = normalize_question(q)
    if "phan tram" in s or "%" in s:
        return "percent"
    if "ngay nao" in s:
        return "date"
    if "thang nao" in s:
        return "date"
    if any(w in s for w in ("co tang lien tuc", "co luon", "khong?")):
        return "boolean"
    if any(w in s for w in ("liet ke", "top ", "nhung mat hang", "lan luot")):
        return "list"
    if any(w in s for w in ("nao", " ai ", "la gi")):
        return "string"
    return "number"


def vocab_for(biz, rp) -> str:
    from .llm.prompts import vocab_block
    lk = rp.lk
    prods = sorted({biz.product_name(k) for k in biz._names_by_key})[:150]
    v = {"Mặt hàng": prods, "Nhóm hàng": sorted(set(lk.groups.values())),
         "Nhóm khách": sorted(set(lk.cust_groups.values())), "Hoạt chất": sorted(set(lk.ingredients.values()))[:60],
         "Đơn vị": sorted(lk.units)[:60]}
    return vocab_block(v)


def rulebook_digest(text: str, max_chars: int = 6000) -> str:
    """Compact rulebook for prompts: keep table rows, drop markdown noise."""
    lines = []
    for l in text.splitlines():
        l = l.strip()
        if not l or l.startswith("|---") or l.startswith("| Luật") or l.startswith("| Cách"):
            continue
        lines.append(l.replace("**", ""))
    s = "\n".join(lines)
    return s[:max_chars]
