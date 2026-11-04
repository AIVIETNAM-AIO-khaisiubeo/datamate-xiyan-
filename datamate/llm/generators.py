"""LLM generators of the multi-generator ensemble.

G2  FrameGenerator   – LLM -> JSON query frame (DSL) -> deterministic executor
G3  RewriteGenerator – LLM rewrites a noisy question into canonical Vietnamese, the rule
                       parser (G1) then parses the rewrite (neuro-symbolic pipeline)
G4  SQLGenerator     – LLM -> DuckDB SQL over rulebook-encoded canonical views (M-Schema
                       prompt, execution-feedback refiner as in XiYan-SQL)

Few-shot examples are retrieved from the training bank with character n-gram TF-IDF
(no extra embedding model, so the whole parameter budget stays with the LLM).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from ..nlu.frame import Frame
from ..nlu.timeparse import Period
from ..textnorm import fold, normalize_question
from .backend import LLMBackend, extract_json
from . import prompts as P


# ============================================================================ example bank
class ExampleBank:
    def __init__(self, items: List[dict]):
        self.items = items
        # shops whose examples must not be shown (leave-one-business-out when evaluating on a labelled split:
        # otherwise a train question would retrieve its own gold frame / SQL as a few-shot example)
        self.exclude: set = set()
        self.df = Counter()
        self.vecs = []
        for it in items:
            v = self._grams(it.get("nq") or normalize_question(it["q"]))
            self.vecs.append(v)
            self.df.update(set(v))
        self.n = max(1, len(items))
        self.norms = [math.sqrt(sum((c * self._idf(g)) ** 2 for g, c in v.items())) or 1.0 for v in self.vecs]

    @staticmethod
    def _grams(s: str) -> Counter:
        s = re.sub(r"\d+", "#", s)
        s = f" {s} "
        return Counter(s[i:i + 4] for i in range(len(s) - 3))

    def _idf(self, g: str) -> float:
        return math.log((1 + self.n) / (1 + self.df.get(g, 0))) + 1

    def topk(self, q: str, k: int = 6, diverse: bool = True) -> List[dict]:
        v = self._grams(normalize_question(q))
        nv = math.sqrt(sum((c * self._idf(g)) ** 2 for g, c in v.items())) or 1.0
        scores = []
        for i, u in enumerate(self.vecs):
            dot = sum(c * u.get(g, 0) * self._idf(g) ** 2 for g, c in v.items())
            scores.append((dot / (nv * self.norms[i]), i))
        scores.sort(reverse=True)
        out, seen = [], Counter()
        for sc, i in scores:
            it = self.items[i]
            if self.exclude and it.get("biz") in self.exclude:
                continue
            if diverse and seen[it.get("intent")] >= 3:
                continue
            seen[it.get("intent")] += 1
            out.append(it)
            if len(out) >= k:
                break
        return out

    @property
    def has_shop_ids(self) -> bool:
        """False for a bank built before shop ids were recorded: leave-one-business-out would then exclude nothing."""
        return bool(self.items) and all("biz" in it for it in self.items)

    @classmethod
    def load(cls, path: str) -> "ExampleBank":
        with open(path, encoding="utf-8") as fh:
            return cls([json.loads(l) for l in fh if l.strip()])


# ============================================================================ canonical renderer
_PAY_VI = {"cash": "tiền mặt", "transfer": "chuyển khoản", "ewallet": "ví điện tử", "cod": "COD", "debt": "ghi nợ"}
_CH_VI = {"phone": "điện thoại", "zalo": "Zalo OA", "store": "tại quầy", "ecommerce": "sàn TMĐT", "facebook": "Facebook"}


def render_time(p: Optional[Period]) -> str:
    if p is None:
        return ""
    lab = p.label or ""
    if lab == "today":
        return "hôm nay"
    if lab == "yesterday":
        return "hôm qua"
    if lab == "this_week":
        return "tuần này"
    if lab == "last_week":
        return "tuần trước"
    if lab == "this_month" or p.kind == "mtd":
        return "tháng này"
    if lab == "ytd":
        return "từ đầu năm đến hết hôm nay"
    m = re.fullmatch(r"last(\d+)d", lab)
    if m:
        return f"{m.group(1)} ngày gần nhất"
    if p.kind == "month":
        return f"tháng {p.start.month}/{p.start.year}"
    if p.kind == "quarter":
        return f"quý {(p.start.month - 1) // 3 + 1}/{p.start.year}"
    if p.kind == "tet":
        return f"dịp Tết {p.end.year}"
    if p.kind == "day" or p.start == p.end:
        return f"ngày {p.start:%d/%m/%Y}"
    return f"từ ngày {p.start:%d/%m/%Y} đến hết ngày {p.end:%d/%m/%Y}"


def render_frame(fr: Frame, name_of=lambda k: k, cust_of=lambda k: k) -> Optional[str]:
    """Frame -> canonical Vietnamese question that the rule parser understands."""
    t = render_time(fr.periods[0]) if fr.periods else ""
    f = fr.filters
    parts = []
    if f.get("product"):
        parts.append(f"của {name_of(f['product'])}")
    if f.get("group"):
        parts.append(f"nhóm {f['group']}")
    if f.get("ingredient"):
        parts.append(f"các thuốc có hoạt chất {f['ingredient']}")
    if f.get("rx") is True:
        parts.append("thuốc kê đơn")
    if f.get("rx") is False:
        parts.append("hàng không kê đơn")
    if f.get("cust_group"):
        parts.append(f"các đơn của khách nhóm {f['cust_group']}")
    if f.get("channel"):
        parts.append(f"các đơn bán qua {_CH_VI.get(f['channel'], f['channel'])}")
    if f.get("pay"):
        parts.append(f"các đơn trả bằng {_PAY_VI.get(f['pay'], f['pay'])}")
    if f.get("shift"):
        parts.append(f"ca {'sáng' if f['shift'] == 'sang' else 'chiều'}")
    if f.get("staff"):
        parts.append(f"của nhân viên mã {f['staff']}")
    flt = " ".join(parts)
    it = fr.intent
    if it == "revenue":
        return f"Doanh thu {flt} {t} là bao nhiêu?".replace("  ", " ")
    if it == "order_count":
        return f"Số đơn {flt} {t} là bao nhiêu?".replace("  ", " ")
    if it == "cancel_count":
        return f"Số đơn bị huỷ {t} là bao nhiêu?"
    if it == "cancel_rate":
        return f"Tỉ lệ đơn huỷ {t} là bao nhiêu %?"
    if it == "aov":
        return f"Giá trị trung bình một đơn {flt} {t} là bao nhiêu?".replace("  ", " ")
    if it == "top_products":
        k = fr.k or 1
        crit = {"profit": "lời nhất", "orders": "được ưa chuộng nhất", "qty": "bán chạy nhất"}.get(fr.by or "qty", "bán chạy nhất")
        return f"{'Mặt hàng' if k == 1 else f'Liệt kê {k} mặt hàng'} {crit} {t} là gì?"
    if it == "top_customers":
        k = fr.k or 1
        crit = {"orders": "mua nhiều đơn nhất", "revenue": "chi tiêu nhiều nhất", "days": "trung thành nhất"}.get(fr.by or "revenue")
        return f"{'Khách nào' if k == 1 else f'{k} khách nào'} {crit} {t}?"
    if it == "best_day":
        return f"Ngày nào trong {t} có doanh thu cao nhất?"
    if it == "units" and f.get("product"):
        return f"{t.capitalize()} bán được bao nhiêu {fr.unit or ''} {name_of(f['product'])}?".replace("  ", " ")
    if it == "growth":
        mode = fr.params.get("mode")
        if mode == "prev":
            return f"Doanh thu {t} thay đổi bao nhiêu % so với kỳ trước?"
        if mode == "yoy":
            return f"So với cùng kỳ năm trước, doanh thu {t} tăng hay giảm bao nhiêu %?"
        if mode == "mtd":
            return "Từ đầu tháng này đến hết hôm nay, doanh thu thay đổi bao nhiêu % so với cùng kỳ tháng trước?"
        if mode == "shift":
            return f"{t.capitalize()} doanh thu ca chiều hơn hay kém ca sáng bao nhiêu %?"
    if it == "share":
        return f"Doanh thu {flt} chiếm bao nhiêu % tổng doanh thu {t}?"
    if it == "debt_total" and fr.snapshot:
        return f"Tính đến hết ngày {fr.snapshot:%d/%m/%Y}, khách còn nợ cửa hàng tổng cộng bao nhiêu tiền?"
    if it == "purchase_total":
        return f"Tổng tiền nhập hàng {t} là bao nhiêu?"
    return None


# ============================================================================ G3 rewrite
class RewriteGenerator:
    name = "G3_rewrite"

    def __init__(self, backend: LLMBackend, bank: Optional[ExampleBank]):
        self.be, self.bank = backend, bank

    def build_prompt(self, question: str, as_of: str, vocab: str) -> str:
        shots = [s for s in (self.bank.topk(question, 6) if self.bank else []) if s.get("canon")]
        return self.be.chat_prompt(P.REWRITE_SYSTEM, P.rewrite_user(question, as_of, vocab, shots))

    def run(self, prompts: List[str], n: int = 1) -> List[List[str]]:
        outs = self.be.generate(prompts, n=n, temperature=0.0 if n == 1 else 0.6, max_tokens=160, seed=11)
        res = []
        for o in outs:
            res.append([re.sub(r"^(viết lại|câu hỏi)\s*:\s*", "", x.strip().split("\n")[0], flags=re.I) for x in o if x.strip()])
        return res


# ============================================================================ G2 frame
class FrameGenerator:
    name = "G2_frame"

    def __init__(self, backend: LLMBackend, bank: Optional[ExampleBank], rulebook_digest: str):
        self.be, self.bank, self.rb = backend, bank, rulebook_digest

    def build_prompt(self, question: str, as_of: str, vocab: str) -> str:
        shots = [s for s in (self.bank.topk(question, 6) if self.bank else []) if s.get("frame_llm")]
        shots = [{"q": s["q"], "frame": s["frame_llm"]} for s in shots]
        return self.be.chat_prompt(P.FRAME_SYSTEM, P.frame_user(question, as_of, self.rb, vocab, shots))

    def run(self, prompts: List[str], n: int = 1) -> List[List[dict]]:
        outs = self.be.generate(prompts, n=n, temperature=0.0 if n == 1 else 0.5, max_tokens=320,
                                json_schema=P.FRAME_SCHEMA, seed=7)
        return [[j for j in (extract_json(x) for x in o) if j] for o in outs]


def llm_json_to_text(js: dict) -> Optional[str]:
    """Turn the LLM's JSON frame into a canonical question for the rule parser (keeps all
    rule semantics in one deterministic place)."""
    it = js.get("intent")
    t = (js.get("time") or "").strip()
    pr = js.get("product")
    flt = []
    if js.get("group"):
        flt.append(f"nhóm {js['group']}")
    if js.get("customer_group"):
        flt.append(f"các đơn của khách nhóm {js['customer_group']}")
    if js.get("channel"):
        flt.append(f"các đơn bán qua {_CH_VI.get(js['channel'], js['channel'])}")
    if js.get("payment"):
        flt.append(f"các đơn trả bằng {_PAY_VI.get(js['payment'], js['payment'])}")
    if js.get("shift"):
        flt.append(f"ca {js['shift']}")
    if js.get("ingredient"):
        flt.append(f"thuốc có hoạt chất {js['ingredient']}")
    if js.get("rx") is True:
        flt.append("thuốc kê đơn")
    if js.get("rx") is False:
        flt.append("hàng không kê đơn")
    if js.get("staff"):
        flt.append(f"của nhân viên mã {js['staff']}")
    f = " ".join(flt)
    if it == "revenue":
        return f"Doanh thu {('của ' + pr) if pr else ''} {f} {t} là bao nhiêu?"
    if it == "order_count":
        return f"Số đơn {f} {t} là bao nhiêu?"
    if it == "cancel_count":
        return f"Số đơn bị huỷ {t} là bao nhiêu?"
    if it == "cancel_rate":
        return f"Tỉ lệ đơn huỷ {t} là bao nhiêu %?"
    if it == "aov":
        return f"Giá trị trung bình một đơn {f} {t} là bao nhiêu?"
    if it == "units" and pr:
        return f"{t} bán được bao nhiêu {js.get('unit') or ''} {pr}?"
    if it == "top_products":
        k = js.get("k") or 1
        crit = {"profit": "lời nhất", "orders": "được ưa chuộng nhất"}.get(js.get("by") or "", "bán chạy nhất")
        return f"{'Mặt hàng' if k == 1 else f'Liệt kê {k} mặt hàng'} {crit} {t} là gì?"
    if it == "top_customers":
        k = js.get("k") or 1
        crit = {"orders": "mua nhiều đơn nhất", "days": "trung thành nhất"}.get(js.get("by") or "", "chi tiêu nhiều nhất")
        return f"{'Khách nào' if k == 1 else f'{k} khách nào'} {crit} {t}?"
    if it == "best_day":
        return f"Ngày nào trong {t} có doanh thu cao nhất?"
    if it == "share":
        return f"Doanh thu {f} chiếm bao nhiêu % tổng doanh thu {t}?"
    if it == "purchase_total":
        return f"Tổng tiền nhập hàng {t} là bao nhiêu?"
    if it == "debt_total":
        return f"Tính đến hết {t or 'hôm nay'}, khách còn nợ cửa hàng tổng cộng bao nhiêu tiền?"
    if it == "growth":
        c = js.get("compare") or "prev"
        if c == "yoy":
            return f"So với cùng kỳ năm trước, doanh thu {t} tăng hay giảm bao nhiêu %?"
        if c == "mtd":
            return "Từ đầu tháng này đến hết hôm nay, doanh thu thay đổi bao nhiêu % so với cùng kỳ tháng trước?"
        if c == "shift":
            return f"{t} doanh thu ca chiều hơn hay kém ca sáng bao nhiêu %?"
        return f"So với tháng liền trước, doanh thu {t} thay đổi bao nhiêu %?"
    return None


# ============================================================================ G4 text-to-SQL
class SQLGenerator:
    name = "G4_sql"

    def __init__(self, backend: LLMBackend, bank: Optional[ExampleBank], rulebook_digest: str):
        self.be, self.bank, self.rb = backend, bank, rulebook_digest

    def build_prompt(self, question: str, as_of: str, mschema: str, answer_type: str) -> str:
        shots = [s for s in (self.bank.topk(question, 4) if self.bank else []) if s.get("sql")]
        return self.be.chat_prompt(P.SQL_SYSTEM, P.sql_user(question, as_of, mschema, self.rb, answer_type, shots))

    def run(self, prompts: List[str], n: int = 1) -> List[List[str]]:
        outs = self.be.generate(prompts, n=n, temperature=0.0 if n == 1 else 0.7, max_tokens=600, seed=3)
        return [[s for s in (extract_sql(x) for x in o) if s] for o in outs]


def extract_sql(text: str) -> Optional[str]:
    """SQL from a model answer.  The prompts end with an opened ```sql fence, so a fine-tuned model answers with the
    SQL followed by the closing fence only ("SELECT ...\\n```"); a base model often writes the whole fenced block.
    The closing fence used to stay in the SQL, so every SQL of the fine-tuned model failed to parse (no G4 at all)."""
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = re.search(r"```(?:sql|duckdb)?[ \t]*\n?(.+?)```", text, re.S | re.I)
    body = m.group(1) if m and re.search(r"\b(select|with)\b", m.group(1), re.I) else None
    if body is None:
        m = (re.search(r"^[ \t]*(select|with)\b.+", text, re.S | re.I | re.M)
             or re.search(r"\b(select|with)\b.+", text, re.S | re.I))
        if not m:
            return None
        body = m.group(0).split("```", 1)[0]
    sql = body.strip().rstrip(";").strip()
    return sql or None


def run_sql(sql: str, views: Dict, timeout_rows: int = 100000):
    """Execute read-only SQL on in-memory views with DuckDB; returns list of row tuples."""
    import duckdb
    low = sql.lower()
    if re.search(r"\b(insert|update|delete|drop|create|alter|attach|copy|pragma|install|load)\b", low):
        raise ValueError("only SELECT is allowed")
    con = duckdb.connect(database=":memory:")
    try:
        for name, df in views.items():
            con.register(name, df)
        res = con.execute(sql).fetchmany(timeout_rows)
        return res
    finally:
        con.close()


def sql_result_to_value(rows, answer_type: str):
    if not rows:
        return None
    if answer_type in ("number", "percent"):
        v = rows[0][0]
        if v is None:
            return None
        v = float(v)
        return round(v, 2) if answer_type == "percent" else (int(round(v)) if abs(v - round(v)) < 1e-6 or abs(v) > 1000 else round(v, 2))
    if answer_type == "string":
        return str(rows[0][0])
    if answer_type == "list":
        if len(rows) == 1 and len(rows[0]) > 1:
            return [x for x in rows[0]]
        return [r[0] for r in rows]
    if answer_type == "date":
        v = rows[0][0]
        return v.isoformat() if hasattr(v, "isoformat") else str(v)
    if answer_type == "boolean":
        return bool(rows[0][0])
    return None
