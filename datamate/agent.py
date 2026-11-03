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
import sys
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
from .textnorm import fold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def default_config() -> dict:
    return {
        "llm": {"backend": os.environ.get("DATAMATE_LLM", "none"),
                "model_path": os.environ.get("DATAMATE_MODEL", "/models/llm"),
                "lora_path": os.environ.get("DATAMATE_LORA") or None,      # fine-tuned adapter next to the base model
                "tensor_parallel_size": int(os.environ.get("DATAMATE_TP", "2")),
                "max_model_len": int(os.environ.get("DATAMATE_MAX_LEN", "16384")),
                "gpu_memory_utilization": float(os.environ.get("DATAMATE_GPU_UTIL", "0.88")),
                "quantization": os.environ.get("DATAMATE_QUANT") or None,       # e.g. 'fp8' on a single 24-32 GB GPU
                "enable_thinking": False},
        "bank_path": os.path.join(ROOT, "artifacts", "example_bank.jsonl"),
        "reliability_path": os.path.join(ROOT, "artifacts", "reliability.json"),
        "vocab_path": os.path.join(ROOT, "artifacts", "known_vocab.json"),
        "tau": float(os.environ.get("DATAMATE_TAU", "0.6")),
        "time_budget_s": float(os.environ.get("DATAMATE_TIME_BUDGET", str(5.5 * 3600))),
        # generators (ablation): DATAMATE_GENS="g2,g3,g4" by default; G1 always runs
        "use_g2": "g2" in os.environ.get("DATAMATE_GENS", "g2,g3,g4"),
        "use_g3": "g3" in os.environ.get("DATAMATE_GENS", "g2,g3,g4"),
        "use_g4": "g4" in os.environ.get("DATAMATE_GENS", "g2,g3,g4"),
        "g4_max_per_biz": 60,
        # XiYan pipeline for G4: filtered schemas (S1, S2) + full-schema baseline, SQL formats, selection model
        "g4_schemas": ["S1", "S2", "full"],
        "g4_scope": os.environ.get("DATAMATE_G4_SCOPE", "unresolved"),     # 'unresolved' | 'all'
        "schema_filter_llm": True, "use_selection_model": True, "g4_refine": True,
        "sql_bank_path": os.path.join(ROOT, "artifacts", "sql_silver.jsonl"),
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
        self.labelled = os.path.exists(os.path.join(split_dir, "answers.jsonl"))
        self.selector = Selector(self.cfg["reliability_path"], tau=self.cfg["tau"])
        from .nlu.coverage import load_vocab
        self.vocab = load_vocab(self.cfg.get("vocab_path"))
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
        self._banks = [x for x in (bank,) if x is not None]
        if self.labelled and bank is not None and not bank.has_shop_ids:
            print("[llm] WARNING: few-shot bank has no shop ids -> own-shop examples are not excluded on this labelled "
                  "split; rebuild with scripts/build_artifacts.py", file=sys.stderr)
        digest = rulebook_digest(self.rulebook_text)
        self.g3 = RewriteGenerator(self.backend, bank)
        self.g2 = FrameGenerator(self.backend, bank, digest)
        self.g4 = SQLGenerator(self.backend, bank, digest)
        from .llm.sql_pipeline import MultiSQLGenerator
        from .select.selection_model import SelectionModel
        self.mgen = MultiSQLGenerator(self.backend, load_sql_bank(self.cfg.get("sql_bank_path")))
        if self.mgen.bank is not None:
            self._banks.append(self.mgen.bank)
        self.selmodel = SelectionModel(self.backend) if self.cfg.get("use_selection_model") else None

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
                                          "reason": a.reason,
                                          "meta": {k: a.meta[k] for k in ("oov", "conflict", "tie") if a.meta.get(k)}})
            log["elapsed_g1_s"] = round(time.time() - tq, 4)
            logs[q["id"]] = log
        # ---------------------------------------------------------------- LLM generators
        llm_on = any(self.cfg.get(k) for k in ("use_g2", "use_g3", "use_g4"))
        if self.cfg["llm"].get("backend", "none") != "none" and llm_on and self.time_left() > 600:
            self._init_llm()
            if self.backend is not None:
                # labelled split (train): never show a shop its own examples (honest evaluation / calibration)
                for bk in getattr(self, "_banks", []):
                    bk.exclude = {bid} if self.labelled else set()
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
        a = fe.run(fr)
        # question form vs. frame answer type
        from .nlu.consistency import check as form_check
        why = form_check(fr.params.get("nq", ""), fr, a)
        if why:
            a.meta["conflict"] = why
        # semantic coverage: words the parse could not account for => do not trust it blindly
        if self.vocab is not None and fr.intent != "unanswerable":
            from .nlu.coverage import unexplained_tokens
            nq = fr.params.get("nq", "")
            masked = rp._mask_entities(" " + nq + " ", question)
            voc = self.vocab
            if biz.flags.get("repair"):
                from .domains.repair import DOMAIN_VOCAB
                voc = voc | DOMAIN_VOCAB
            if biz.flags.get("phone"):
                from .domains.phone import DOMAIN_VOCAB as PV
                # model lines are named without storage / colour ('iPhone 16 Pro Max'): their words are shop words
                names = {n for c in getattr(biz, "_names_by_key", {}).values() for n in c}
                voc = voc | PV | {w for n in names for w in fold(n).split()}
            shop_words = list(rp.lk.units) + list(rp.lk.groups) + list(rp.lk.cust_groups)
            P = biz.products
            if not P.empty and "brand" in P.columns:
                shop_words += [str(x) for x in P["brand"].dropna().unique()]
            oov = unexplained_tokens(masked, voc, shop_words)
            if oov:
                a.meta["oov"] = oov
        return fr, a

    # ------------------------------------------------------------------ LLM stage
    def _llm_stage(self, biz, rp, items, cands, logs):
        from .llm.generators import llm_json_to_text
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
        # G4: XiYan pipeline (Schema Filter -> multi-schema x multi-format SQL -> self-refine -> selection) -----
        if self.cfg.get("use_g4") and self.time_left() > 1200:
            if self.cfg.get("g4_scope") == "all":
                need = list(items)
            else:
                need = [q for q in items if self.selector.select(cands[q["id"]])[0].type == "abstain"]
            need = need[: self.cfg["g4_max_per_biz"]]
            if need:
                self._xiyan_sql(biz, rp, need, cands, logs)
        dt_llm = (time.time() - tl) / max(1, len(items))
        for q in items:
            logs[q["id"]]["elapsed_llm_s"] = round(dt_llm, 4)


def _xiyan_sql_impl(self, biz, rp, need, cands, logs):
    """Schema Filter -> Algorithm 2 (multi-schema x multi-format SQL, self-refine) -> Algorithm 3
    (reorganisation + selection model).  Every SQL candidate becomes a vote; the selector decides."""
    from .llm.evidence import EvidenceRetriever
    from .llm.mschema import build_mschema, build_views
    from .llm.schema_filter import SchemaFilter
    from .llm.sql_pipeline import SQLRunner
    from .nlu.rules import _topk
    from .nlu.timeparse import parse_time
    from .select.selection_model import reorganize
    from .textnorm import normalize_question
    views = build_views(biz)
    ms = build_mschema(biz.biz_id, views)
    runner = SQLRunner(views)
    sf = SchemaFilter(ms, views, rp.lk, backend=self.backend if self.cfg.get("schema_filter_llm") else None)
    ev = EvidenceRetriever(self.rulebook_text, getattr(getattr(biz, "readme", None), "text", "") or "")
    want = set(self.cfg.get("g4_schemas") or ["full"])
    jobs, union_cols = [], {}
    tis = [parse_time(normalize_question(q["question"]), dt.date.fromisoformat(q["as_of_date"]), self.cal) for q in need]
    evids = [ev.evidence(q["question"], q["as_of_date"]) for q in need]
    fress = sf.run_batch([q["question"] for q in need], evids, [ti.text_wo_time for ti in tis])   # batched LLM calls
    for q, ti, evid, fres in zip(need, tis, evids, fress):
        today = dt.date.fromisoformat(q["as_of_date"])
        nq = normalize_question(q["question"])
        g1 = next((c for c in cands[q["id"]] if c.gen == "G1_rule"), None)
        at = g1.answer.type if g1 is not None and g1.answer.type not in ("abstain", "unanswerable") \
            else _guess_type(q["question"])
        schemas, seen_cols = [], set()
        for sc in fres.schemas:
            key = tuple(sorted(sc["columns"]))
            if sc["id"] in want and sc["valid"] and key not in seen_cols:
                seen_cols.add(key)
                schemas.append((sc["id"], len(sc["columns"]), ms.to_mschema(sc["columns"], extra_examples=fres.values)))
        if "full" in want or not schemas:
            schemas.append(("full", len(fres.full_columns), ms.to_mschema(extra_examples=fres.values)))
        union_cols[q["id"]] = sorted({c for sc in fres.schemas if sc["valid"] for c in sc["columns"]}) or None
        logs[q["id"]]["schema_filter"] = fres.to_log()
        logs[q["id"]]["evidence"] = evid
        jobs.append({"qid": q["id"], "question": q["question"], "nq": nq, "evidence": evid, "answer_type": at,
                     "periods": ti.periods, "k": _topk(nq), "today": today, "schemas": schemas})
    deadline = self.t_start + self.cfg["time_budget_s"] - 900
    out = self.mgen.run(jobs, runner, refine=self.cfg.get("g4_refine", True), deadline=deadline)
    sel_jobs = []
    order = {g["name"]: self.selector.r(g["name"], None) for g in self.mgen.gens}
    for j in jobs:
        lst = out.get(j["qid"], [])
        logs[j["qid"]]["sql_candidates"] = [c.log() for c in lst]
        for c in lst:
            if c.ok:
                a = Answer(c.answer_type, c.value)
                det = {"sql": c.sql, "family": c.family, "flags": c.flags, "schema_id": c.schema_id, "fmt": c.fmt}
                cands[j["qid"]].append(Candidate(c.generator, a, None, det))
                logs[j["qid"]]["candidates"].append({"gen": c.generator, "answer": a.to_submission(j["qid"]),
                                                     "sql": c.sql, "schema": c.schema_id, "flags": c.flags})
        shown, mode, top = reorganize(lst, order)
        logs[j["qid"]]["selection"] = {"mode": mode, "largest_cluster": top, "n_shown": len(shown)}
        if self.selmodel is not None and len(shown) > 1 and self.time_left() > 600:
            u = union_cols.get(j["qid"])
            sel_jobs.append({"qid": j["qid"], "question": j["question"], "evidence": j["evidence"], "shown": shown,
                             "schema_union": ms.to_mschema(u) if u else ms.to_mschema()})
    if sel_jobs:
        chosen = self.selmodel.choose(sel_jobs)
        for sj in sel_jobs:
            c = chosen.get(sj["qid"])
            if c is None:
                continue
            a = Answer(c.answer_type, c.value)
            cands[sj["qid"]].append(Candidate("G_select", a, None,
                                              {"sql": c.sql, "family": "selector:" + c.family, "flags": c.flags}))
            logs[sj["qid"]]["selection"]["chosen"] = c.sql
            logs[sj["qid"]]["candidates"].append({"gen": "G_select", "answer": a.to_submission(sj["qid"]), "sql": c.sql})
    runner.close()


DataMateAgent._xiyan_sql = _xiyan_sql_impl


def load_sql_bank(path: Optional[str]):
    """Few-shot bank of execution-verified SQL (train only) for the ICL path of the SQL generators."""
    from .llm.generators import ExampleBank
    if not path or not os.path.exists(path):
        return None
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            items.append({"q": r["question"], "intent": r["intent"], "sql": r["sql"], "biz": r["business_id"]})
    return ExampleBank(items) if items else None


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
