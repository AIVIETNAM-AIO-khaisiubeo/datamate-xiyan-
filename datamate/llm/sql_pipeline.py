"""Multiple SQL generation over multiple filtered schemas (XiYan-SQL Algorithm 2) with full provenance.

    for each schema S_i in {S1, S2, full}:          (Schema Filter output + full-schema baseline)
      for each generator M_j:                        (one model, several SQL formats; LoRA variants if trained)
        l_ij = M_j(Q, E, S_i); r = execute(l_ij)
        if error / anomalous result: l_ij' = M_j(Q, E, S_i, l_ij, r)      (self-refine, once)
    deduplicate (de-formalised SQL), lint, keep provenance

Each candidate records: SQL, schema id and size, generator / format / model family, execution result or
error, refine flag, generation time and token counts.  Candidates of one model family are *not*
independent votes; the selector counts a family once per answer group.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..select.sql_checks import lint
from . import prompts as P
from .generators import ExampleBank, extract_sql, sql_result_to_value

DEFAULT_GENERATORS = [
    {"name": "G4_base", "fmt": "base", "icl": True},
    {"name": "G4_cte", "fmt": "cte", "icl": True},
    {"name": "G4_alias", "fmt": "alias", "icl": True},
]


# ---------------------------------------------------------------------------- SQL execution
class SQLRunner:
    """One read-only DuckDB connection per shop, views registered once, per-query timeout."""

    FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|"
                           r"import|call|set|read_csv|read_parquet|read_json|glob)\b")

    def __init__(self, views: Dict, timeout_s: float = 5.0, max_rows: int = 1000):
        import duckdb
        self.con = duckdb.connect(database=":memory:")
        for name, df in views.items():
            self.con.register(name, df)
        self.timeout_s, self.max_rows = timeout_s, max_rows

    def run(self, sql: str):
        if self.FORBIDDEN.search(re.sub(r"'(?:[^']|'')*'", "''", sql.lower())):
            raise ValueError("only read-only SELECT queries are allowed")
        timer = threading.Timer(self.timeout_s, self.con.interrupt)
        timer.start()
        try:
            cur = self.con.execute(sql)
            rows = cur.fetchmany(self.max_rows + 1)
        finally:
            timer.cancel()
        return rows

    def close(self):
        try:
            self.con.close()
        except Exception:  # noqa: BLE001
            pass


def deformalize(sql: str) -> str:
    """Remove superficial style (case, spacing, trailing ';') but keep string literals: used for dedup and
    for presenting candidates to the selection model (XiYan §III-D)."""
    parts = re.split(r"('(?:[^']|'')*')", sql.strip().rstrip(";"))
    out = []
    for i, p in enumerate(parts):
        if i % 2:
            out.append(p)                                   # string literal: untouched
        else:
            c = re.sub(r"\s+", " ", p.lower())
            c = re.sub(r"\s*([(,=<>])\s*", r"\1", c)
            out.append(re.sub(r"\s*\)", ")", c))
    return "".join(out).strip()


def exec_info(rows, err: Optional[str], max_rows: int = 10) -> str:
    if err:
        return f"Lỗi: {err}"
    if not rows:
        return "Không có dòng nào"
    shown = "\n".join(str(tuple(r)) for r in rows[:max_rows])
    more = f"\n... ({len(rows)} dòng)" if len(rows) > max_rows else ""
    return shown + more


@dataclass
class SQLCandidate:
    qid: str
    sql: str
    schema_id: str
    schema_cols: int
    generator: str
    fmt: str
    family: str
    refined: bool = False
    ok: bool = False
    error: Optional[str] = None
    n_rows: int = 0
    preview: str = ""
    value: object = None
    answer_type: str = "number"
    flags: List[Tuple[str, float]] = field(default_factory=list)
    t_gen_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    also: List[str] = field(default_factory=list)     # other (schema/generator) that produced the same SQL

    def log(self) -> Dict:
        d = asdict(self)
        d["value"] = self.value if isinstance(self.value, (int, float, str, bool, list, type(None))) else str(self.value)
        return d


# ---------------------------------------------------------------------------- generation
class MultiSQLGenerator:
    def __init__(self, backend, bank: Optional[ExampleBank] = None, generators: Sequence[Dict] = DEFAULT_GENERATORS,
                 n_shots: int = 3, max_tokens: int = 700):
        self.be, self.bank = backend, bank
        self.gens = list(generators)
        self.n_shots, self.max_tokens = n_shots, max_tokens
        self.family = getattr(backend, "family", "llm")

    def _shots(self, question: str, fmt: str) -> List[Dict[str, str]]:
        if self.bank is None:
            return []
        out = []
        for it in self.bank.topk(question, self.n_shots * 2):
            sql = (it.get("sql") or {}).get(fmt) or (it.get("sql") or {}).get("base")
            if sql:
                out.append({"question": it["q"], "sql": sql})
            if len(out) >= self.n_shots:
                break
        return out

    def _generate(self, prompts: List[str]) -> Tuple[List[str], List[Tuple[int, int]], float]:
        t0 = time.time()
        outs = self.be.generate(prompts, n=1, temperature=0.0, max_tokens=self.max_tokens, seed=3)
        dt_ = (time.time() - t0) / max(1, len(prompts))
        usage = getattr(self.be, "last_usage", None) or []
        toks = [(u[0], (u[1] or [0])[0]) for u in usage] if len(usage) == len(prompts) else [(0, 0)] * len(prompts)
        return [(o[0] if o else "") for o in outs], toks, dt_

    def run(self, jobs: List[Dict], runner: SQLRunner, today=None, refine: bool = True,
            deadline: Optional[float] = None) -> Dict[str, List[SQLCandidate]]:
        """jobs: [{qid, question, nq, evidence, answer_type, periods, k, schemas: [(id, n_cols, mschema_text)]}]"""
        tasks, seen_prompt = [], set()
        for j in jobs:
            for sid, ncols, ms_text in j["schemas"]:
                for g in self.gens:
                    shots = self._shots(j["question"], g["fmt"]) if g.get("icl") else []
                    user = P.nl2sql_user(j["question"], ms_text, j["evidence"], g["fmt"], shots)
                    prompt = self.be.chat_prompt(P.NL2SQL_SYSTEM, user)
                    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()
                    if h in seen_prompt:
                        continue
                    seen_prompt.add(h)
                    tasks.append((j, sid, ncols, ms_text, g, prompt))
        res: Dict[str, List[SQLCandidate]] = {j["qid"]: [] for j in jobs}
        seen_sql: Dict[str, Dict[str, SQLCandidate]] = {j["qid"]: {} for j in jobs}
        if not tasks:
            return res
        texts, toks, t_each = self._generate([t[-1] for t in tasks])
        retry = []
        for (j, sid, ncols, ms_text, g, _), text, (pt, ct) in zip(tasks, texts, toks):
            sql = extract_sql(text) if text else None
            c = SQLCandidate(j["qid"], sql or "", sid, ncols, g["name"], g["fmt"], self.family, t_gen_s=round(t_each, 3),
                             prompt_tokens=pt, completion_tokens=ct, answer_type=j["answer_type"])
            if not sql:
                c.error = "no SQL in output"
                res[j["qid"]].append(c)
                continue
            key = deformalize(sql)
            if key in seen_sql[j["qid"]]:                       # duplicate: not re-executed, provenance kept
                seen_sql[j["qid"]][key].also.append(f"{sid}/{g['name']}")
                continue
            seen_sql[j["qid"]][key] = c
            self._execute(c, j, runner, today)
            res[j["qid"]].append(c)
            if refine and (not c.ok or c.value is None):
                retry.append((j, ms_text, g, c))
        if retry and refine and (deadline is None or time.time() < deadline):
            prompts = [self.be.chat_prompt(P.SELF_REFINE_SYSTEM,
                                           P.self_refine_user(j["question"], ms_text, j["evidence"], c.sql,
                                                              exec_info(None, c.error) if c.error else c.preview or "Không có dòng nào"))
                       for j, ms_text, g, c in retry]
            texts, toks, t_each = self._generate(prompts)
            for (j, ms_text, g, c0), text, (pt, ct) in zip(retry, texts, toks):
                sql = extract_sql(text) if text else None
                if not sql or deformalize(sql) in seen_sql[j["qid"]]:
                    continue
                c = SQLCandidate(j["qid"], sql, c0.schema_id, c0.schema_cols, c0.generator, c0.fmt, self.family,
                                 refined=True, t_gen_s=round(t_each, 3), prompt_tokens=pt, completion_tokens=ct,
                                 answer_type=j["answer_type"])
                seen_sql[j["qid"]][deformalize(sql)] = c
                self._execute(c, j, runner, today)
                res[j["qid"]].append(c)
        return res

    @staticmethod
    def _execute(c: SQLCandidate, j: Dict, runner: SQLRunner, today):
        try:
            rows = runner.run(c.sql)
        except Exception as e:  # noqa: BLE001
            c.error = f"{type(e).__name__}: {e}"[:300]
            return
        c.n_rows = len(rows)
        c.preview = exec_info(rows, None)
        try:
            c.value = sql_result_to_value(rows, c.answer_type)
        except (TypeError, ValueError) as e:
            c.error = f"result conversion: {e}"
            return
        c.ok = c.value is not None
        c.flags = lint(c.sql, j.get("nq", ""), j.get("periods") or (), j.get("today", today), c.answer_type, j.get("k"), c.n_rows)
