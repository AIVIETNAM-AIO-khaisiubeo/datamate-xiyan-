"""Per-question Schema Filter (XiYan-SQL §III-B: multi-path retrieval + iterative column selection).

    K        = keywords(Q, E)                                   LLM M_s if available, else rule keywords
    Score(k,c) = cos(V_{Q||E}, V_tab(c)) · cos(V_k, V_c)         table-column retrieval  (Eq. 1)
    values   = edit-distance match of k against column values    value retrieval (with column provenance)
    S_rtrv   = top-k columns per keyword ∪ value columns
    for i in 1..ps:                                              Algorithm 1
        S_slct_i = M_s(S_rtrv, Q, E)                            LLM column selection (heuristic without LLM)
        P_i      = PFKeyIdentifier(S_slct_i)                    complete composite keys + join path keys
        S_i      = ∪_{k<i} S_k ∪ S_slct_i ∪ P_i ∪ G_i           G_i = business-condition columns (is_valid, kind,
                                                                 status, cancelled) and the date column of
                                                                 every table used
        S_rtrv   = S_rtrv \\ (S_slct_i \\ P_i)

Deviations from the paper, forced by the competition rules (≤ 15B parameters for *all* models, offline):
the embedding model is replaced by character 4-gram TF-IDF over Vietnamese column descriptions + a small
business glossary, and LSH by an n-gram inverted index; M_s is the same Qwen3-14B as the generators.
Business-condition columns are always kept: a query that selects `amount` but not `is_valid` runs fine and
answers wrongly.  The full canonical schema stays available as baseline and as fallback when a filtered
schema is invalid (no table, or tables that cannot be joined).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from ..textnorm import fold, normalize_question
from .mschema import MSchema

# Vietnamese business vocabulary per column / table (the "metadata" that the paper embeds).
COLUMN_GLOSS: Dict[str, str] = {
    "orders.order_id": "don ma don so don bao nhieu don dem don",
    "orders.order_date": "ngay dat thang tuan quy nam ky hom nay hom qua",
    "orders.order_hour": "gio ca sang ca chieu khung gio cao diem",
    "orders.status": "trang thai huy don huy tra hang dang giao ti le huy",
    "orders.is_valid": "don hop le khong huy khong tra",
    "orders.customer_key": "khach hang khach mua so khach",
    "orders.customer_name": "ten khach ai khach nao",
    "orders.customer_group": "nhom khach than thiet si vip le",
    "orders.staff": "nhan vien nguoi ban nv tho",
    "orders.channel": "kenh ban tai quay zalo dien thoai san thuong mai dien tu facebook shopee",
    "orders.payment_method": "hinh thuc thanh toan chuyen khoan tien mat vi dien tu cod ghi no",
    "orders.revenue": "doanh thu tien ban ban duoc bao nhieu tien thu ve tien hang doanh so trung binh moi don "
                      "mua nhieu tien chi tieu chi nhieu tien nhat",
    "orders.total_due": "khach phai tra tong tien don cong no",
    "sales_lines.order_id": "don co mat hang",
    "sales_lines.line_no": "dong",
    "sales_lines.order_date": "ngay ban thang tuan",
    "sales_lines.product_key": "mat hang san pham ma hang",
    "sales_lines.product_name": "mat hang san pham mon hang ten hang nao",
    "sales_lines.product_group": "nhom hang loai hang nhom",
    "sales_lines.kind": "dong hang phi phuc vu voucher diem ship",
    "sales_lines.unit": "don vi tinh thung hop chai goi",
    "sales_lines.qty": "so luong",
    "sales_lines.base_qty": "so luong ban chay chay nhat hot nhat ban nhieu nhat bao nhieu cai ly chai goi lon",
    "sales_lines.base_unit": "don vi co ban",
    "sales_lines.unit_price": "don gia gia ban",
    "sales_lines.amount": "doanh thu mat hang thanh tien tien ban nhieu tien",
    "sales_lines.cost_per_unit": "gia von lai gop loi nhuan loi nhat",
    "sales_lines.is_valid": "don hop le",
    "products.name": "ten mat hang", "products.product_group": "nhom hang", "products.price": "gia ban",
    "products.cost": "gia von gia tri ton kho", "products.active_ingredient": "hoat chat thuoc",
    "products.is_prescription": "thuoc ke don khong ke don", "products.base_unit": "don vi co ban",
    "payments.pay_date": "ngay tra no ngay thu", "payments.amount": "tra no thu no tien tra khach tra bot",
    "payments.method": "hinh thuc tra",
    "purchases.receipt_date": "ngay nhap phieu nhap", "purchases.supplier": "nha cung cap ncc nhap cua ai",
    "purchases.total": "tien nhap hang tong phieu nhap", "purchases.cancelled": "phieu huy",
    "purchase_lines.qty": "so luong nhap", "purchase_lines.base_qty": "so luong nhap ve",
    "purchase_lines.unit_cost": "gia nhap binh quan", "purchase_lines.amount": "tien nhap",
    "purchase_lines.lot": "lo hang so lo", "purchase_lines.expiry": "han dung het han hsd can date",
    "expenses.amount": "chi phi lai rong", "expenses.category": "loai chi phi", "expenses.expense_date": "ngay chi",
    "deliveries.trip_no": "dot giao chuyen giao", "deliveries.delivery_date": "ngay giao cho hang",
    "deliveries.base_qty": "so luong giao da cho",
}
TABLE_GLOSS: Dict[str, str] = {
    "orders": "don hang doanh thu so don khach ban hang huy",
    "sales_lines": "mat hang ban chay so luong ban doanh thu nhom hang",
    "products": "danh muc hang gia von ton kho",
    "payments": "tra no thu no cong no",
    "purchases": "nhap hang phieu nhap nha cung cap",
    "purchase_lines": "nhap hang lo han dung",
    "expenses": "chi phi",
    "deliveries": "giao hang dot giao",
}
# entity name <-> identifier: selecting one makes the other necessary (filter by key, display by name)
ENTITY_PAIRS = [("sales_lines.product_name", "sales_lines.product_key"), ("orders.customer_name", "orders.customer_key"),
                ("products.name", "products.product_key")]
# lexicon path of the multi-path retriever: synonym -> column mappings (XiYan's evidence 'synonym mappings')
LEXICON = [
    (r"\b(?:so|bao nhieu|may|dem|tong cong|tong so)\s+(?:don|hoa don|bill)\b|\bdon hang\b", ["orders.order_id"]),
    (r"\bca (?:sang|chieu)\b|\bgio cao diem\b|\bkhung gio\b", ["orders.order_hour"]),
    (r"\bkhach\b.*\b(?:nhieu tien|chi tieu|chi nhieu|mua nhieu|tong tien)\b", ["orders.revenue", "orders.customer_name"]),
    (r"\bhuy\b", ["orders.status"]),
    (r"\b(?:chuyen khoan|tien mat|vi dien tu|cod|ghi no)\b", ["orders.payment_method"]),
    (r"\b(?:tai quay|zalo|san thuong mai|facebook|shopee|qua dien thoai)\b", ["orders.channel"]),
    (r"\bnhom khach\b|\bkhach (?:than thiet|si|vip)\b", ["orders.customer_group"]),
    (r"\bnhap hang\b|\bphieu nhap\b|\btien nhap\b", ["purchases.total", "purchases.receipt_date"]),
]
_STOP = set("""bao nhieu bn may la gi nao co duoc dc cua cho toi minh shop tiem quan thi va voi vay nhi
ha a nhe nha roi tong cong tat ca cac nhung mot trong den toi het tu ngay thang nam tuan quy hom nay qua
""".split())


# ---------------------------------------------------------------------------- TF-IDF ("embedding")
def _grams(s: str) -> Counter:
    s = " " + re.sub(r"\d+", "#", fold(s)) + " "
    return Counter(s[i:i + 4] for i in range(len(s) - 3))


class _Space:
    def __init__(self, docs: Dict[str, str]):
        self.vecs = {k: _grams(v) for k, v in docs.items()}
        df = Counter()
        for v in self.vecs.values():
            df.update(set(v))
        n = max(1, len(docs))
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.norm = {k: self._norm(v) for k, v in self.vecs.items()}

    def _norm(self, v: Counter) -> float:
        return math.sqrt(sum((c * self.idf.get(g, 1.0)) ** 2 for g, c in v.items())) or 1.0

    def cos(self, text_vec: Counter, key: str) -> float:
        u = self.vecs[key]
        dot = sum(c * u.get(g, 0) * self.idf.get(g, 1.0) ** 2 for g, c in text_vec.items())
        return dot / (self._norm(text_vec) * self.norm[key])


# ---------------------------------------------------------------------------- result
@dataclass
class FilterResult:
    keywords: List[str]
    retrieved: List[Tuple[str, float]]                    # (table.column, score) of S_rtrv
    values: Dict[str, List[str]]                          # table.column -> matched values
    schemas: List[Dict]                                   # [{"id": "S1", "columns": [...], "valid": bool}]
    full_columns: List[str]
    fallback: bool = False
    notes: List[str] = field(default_factory=list)

    def to_log(self) -> Dict:
        return {"keywords": self.keywords, "retrieved": [c for c, _ in self.retrieved][:40],
                "values": self.values, "schemas": [{"id": s["id"], "n_cols": len(s["columns"]), "valid": s["valid"],
                                                     "selector": s.get("selector"), "columns": s["columns"]}
                                                    for s in self.schemas],
                "full_cols": len(self.full_columns), "fallback": self.fallback, "notes": self.notes}


# ---------------------------------------------------------------------------- the filter
class SchemaFilter:
    def __init__(self, ms: MSchema, views: Dict, linker=None, backend=None, top_k: int = 5, ps: int = 2,
                 max_values_per_col: int = 3, value_threshold: float = 0.82):
        self.ms, self.views, self.lk, self.be = ms, views, linker, backend
        self.top_k, self.ps = top_k, ps
        self.max_values, self.vthr = max_values_per_col, value_threshold
        cols = ms.columns()
        self.col_space = _Space({c: self._col_doc(c) for c in cols})
        self.tab_space = _Space({t: f"{t} {ti.get('comment', '')} {TABLE_GLOSS.get(t, '')}" for t, ti in ms.tables.items()})
        self._value_index = None

    def _col_doc(self, c: str) -> str:
        t, col = c.split(".", 1)
        f = self.ms.tables[t]["fields"][col]
        return f"{col.replace('_', ' ')} {f.get('comment', '')} {COLUMN_GLOSS.get(c, '')}"

    # -- keywords ---------------------------------------------------------------
    def rule_keywords(self, question: str, text_wo_time: Optional[str] = None) -> List[str]:
        s = normalize_question(question) if text_wo_time is None else text_wo_time
        kws: List[str] = []
        ents = []
        if self.lk is not None:
            for fn in (self.lk.products, self.lk.customers):
                try:
                    ents += [n for n, _, _ in fn(question)]
                except Exception:  # noqa: BLE001
                    pass
            for d in (self.lk.groups, self.lk.cust_groups, self.lk.ingredients, self.lk.staff, self.lk.suppliers):
                ents += [n for n in d if n and (" " + n + " ") in " " + s + " "]
        for e in sorted(set(ents), key=len, reverse=True):
            kws.append(e)
            s = s.replace(e, " | ")
        for chunk in re.split(r"\|", s):
            words = [w for w in chunk.split() if w not in _STOP and not w.isdigit() and not re.fullmatch(r"[\d/.,%-]+", w)]
            if words:
                kws.append(" ".join(words))
            kws += [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]        # sliding bigrams ('ca sang')
        return [k for k in dict.fromkeys(kws) if len(k) >= 2]

    def llm_keywords_batch(self, questions: List[str], evidences: List[str]) -> List[List[str]]:
        if self.be is None or not questions:
            return [[] for _ in questions]
        from . import prompts as P
        from .backend import extract_json
        out = self.be.generate([self.be.chat_prompt(P.KEYWORD_SYSTEM, P.keyword_user(q, e)) for q, e in zip(questions, evidences)],
                               max_tokens=160, json_schema=P.KEYWORD_SCHEMA, seed=5)
        res = []
        for o in out:
            js = extract_json(o[0]) if o else None
            res.append([str(k) for k in (js or {}).get("keywords", []) if str(k).strip()][:12])
        return res

    # -- retrieval --------------------------------------------------------------
    def retrieve(self, question: str, evidence: str, keywords: List[str]) -> List[Tuple[str, float]]:
        vqe = _grams(question + " " + evidence)
        tab_score = {t: self.tab_space.cos(vqe, t) for t in self.ms.tables}
        best: Dict[str, float] = {}
        for k in keywords:
            vk = _grams(k)
            sc = []
            for c in self.ms.columns():
                s = tab_score[c.split(".")[0]] * self.col_space.cos(vk, c)
                if s > 0:
                    sc.append((s, c))
            sc.sort(reverse=True)
            for s, c in sc[:self.top_k]:
                best[c] = max(best.get(c, 0.0), s)
        return sorted(best.items(), key=lambda kv: -kv[1])

    def _build_value_index(self):
        idx = []
        for t, df in self.views.items():
            for col in df.columns:
                if col.endswith(("_id", "_key", "_date", "_no")) or df[col].dtype != object:
                    continue
                vals = [v for v in df[col].dropna().unique() if isinstance(v, str)]
                if not vals or len(vals) > 5000:
                    continue
                inv = defaultdict(set)
                fv = [fold(v) for v in vals]
                for i, f in enumerate(fv):
                    for g in set(_grams(f)):
                        inv[g].add(i)
                idx.append((f"{t}.{col}", vals, fv, inv))
        self._value_index = idx

    def retrieve_values(self, keywords: List[str]) -> Dict[str, List[str]]:
        if self._value_index is None:
            self._build_value_index()
        out: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for k in keywords:
            fk = fold(k).strip()
            if len(fk) < 3:
                continue
            gk = set(_grams(fk))
            for col, vals, fv, inv in self._value_index:
                cnt = Counter()
                for g in gk:
                    for i in inv.get(g, ()):
                        cnt[i] += 1
                for i, c in cnt.most_common(30):                       # n-gram prefilter (LSH stand-in)
                    if c < 0.5 * len(gk):
                        break
                    v = fv[i]
                    contained = (" " + fk + " ") in (" " + v + " ") or (" " + v + " ") in (" " + fk + " ")
                    r = 1.0 if contained else SequenceMatcher(None, fk, v).ratio()
                    if r >= self.vthr:
                        out[col].append((r, vals[i]))
        res = {}
        for col, lst in out.items():
            lst.sort(key=lambda x: -x[0])
            res[col] = list(dict.fromkeys(v for _, v in lst))[:self.max_values]
        return res

    # -- keys and business conditions ------------------------------------------
    def pf_key_identifier(self, cols: List[str]) -> List[str]:
        """Complete composite primary keys of every table used and add the key columns of a join path that
        connects all used tables (BFS over the verified foreign keys)."""
        tabs = list(dict.fromkeys(c.split(".")[0] for c in cols))
        keys: List[str] = []
        for t in tabs:
            keys += [f"{t}.{c}" for c in self.ms.primary_key(t)]
        if len(tabs) > 1:
            root = tabs[0]
            for t in tabs[1:]:
                path = self._join_path(root, t)
                if path is None:
                    continue
                for a, ca, b, cb in path:
                    keys += [f"{a}.{ca}", f"{b}.{cb}"]
                    keys += [f"{b}.{c}" for c in self.ms.primary_key(b)]
        return list(dict.fromkeys(keys))

    def _join_path(self, src: str, dst: str):
        prev = {src: None}
        q = [src]
        while q:
            t = q.pop(0)
            if t == dst:
                break
            for a, ca, b, cb in self.ms.edges(t):
                if b not in prev:
                    prev[b] = (a, ca, b, cb)
                    q.append(b)
        if dst not in prev:
            return None
        path, t = [], dst
        while prev[t] is not None:
            e = prev[t]
            path.append(e)
            t = e[0]
        return list(reversed(path))

    def business_columns(self, cols: List[str]) -> List[str]:
        """Date column and business-condition columns of every table used, and the key / name partner of an
        entity column."""
        out = []
        have = set(cols)
        for a, b in ENTITY_PAIRS:
            if a in have and b.split(".")[0] in self.ms.tables and b in self.col_space.vecs:
                out.append(b)
            if b in have and a in self.col_space.vecs:
                out.append(a)
        for t in dict.fromkeys(c.split(".")[0] for c in cols):
            d = self.ms.date_column(t)
            if d:
                out.append(f"{t}.{d}")
            out += [f"{t}.{g}" for g in self.ms.guards(t)]
        return out

    def connected(self, cols: List[str]) -> bool:
        tabs = list(dict.fromkeys(c.split(".")[0] for c in cols))
        return all(self._join_path(tabs[0], t) is not None for t in tabs[1:]) if tabs else False

    # -- column selection (Algorithm 1) ---------------------------------------
    @staticmethod
    def _heuristic_select(rem: List[Tuple[str, float]], valued: Set[str], it: int) -> List[str]:
        """M_s without an LLM: first round the clearly relevant columns, later rounds widen recall."""
        if not rem:
            return []
        top = rem[0][1]
        if it == 0:
            return [c for c, s in rem if s >= 0.55 * top or c in valued]
        return [c for c, _ in rem[:6]]

    def _llm_select_batch(self, todo: List[Dict], it: int) -> List[List[str]]:
        from . import prompts as P
        from .backend import extract_json
        prompts = [self.be.chat_prompt(P.COLUMN_SELECT_SYSTEM, P.column_select_user(
            st["question"], st["evidence"], self.ms.to_mschema([c for c, _ in st["rem"]], extra_examples=st["values"])))
            for st in todo]
        out = self.be.generate(prompts, max_tokens=300, json_schema=P.COLUMN_SELECT_SCHEMA, seed=13 + it)
        picks = []
        for st, o in zip(todo, out):
            js = extract_json(o[0]) if o else None
            allowed = {c for c, _ in st["rem"]}
            picks.append([c for c in (js or {}).get("columns", []) if c in allowed])
        return picks

    def run(self, question: str, evidence: str = "", text_wo_time: Optional[str] = None) -> FilterResult:
        return self.run_batch([question], [evidence], [text_wo_time])[0]

    def run_batch(self, questions: List[str], evidences: List[str], texts_wo_time: Optional[List] = None) -> List[FilterResult]:
        """Algorithm 1 for many questions at once: one LLM call per stage (keywords, each selection round)."""
        texts_wo_time = texts_wo_time or [None] * len(questions)
        notes = [[] for _ in questions]
        try:
            llm_kws = self.llm_keywords_batch(questions, evidences)
        except Exception as e:  # noqa: BLE001
            llm_kws = [[] for _ in questions]
            for n in notes:
                n.append(f"keyword LLM failed: {e}")
        states = []
        for q, e, t, lk in zip(questions, evidences, texts_wo_time, llm_kws):
            kws = list(dict.fromkeys(lk + self.rule_keywords(q, t)))
            nq = normalize_question(q)
            rtrv = self.retrieve(nq, e, kws)
            values = self.retrieve_values(kws)
            lex = [c for rx, cols in LEXICON if re.search(rx, nq) for c in cols if c in self.col_space.vecs]
            valued = set(values) | set(lex)
            have = {c for c, _ in rtrv}
            for c in valued:
                if c not in have:
                    rtrv.append((c, max([s for _, s in rtrv] or [0.1])))
            states.append({"question": q, "evidence": e, "kws": kws, "rtrv": rtrv, "values": values, "valued": valued,
                           "rem": list(rtrv), "union": [], "schemas": [], "done": False})
        for it in range(self.ps):
            todo = [st for st in states if not st["done"] and st["rem"]]
            if not todo:
                break
            picks = [[] for _ in todo]
            if self.be is not None:
                try:
                    picks = self._llm_select_batch(todo, it)
                except Exception as e:  # noqa: BLE001
                    for st in todo:
                        st.setdefault("_notes", []).append(f"column-selection LLM failed: {e}")
            for st, pick in zip(todo, picks):
                slct = pick or self._heuristic_select(st["rem"], st["valued"], it)
                if not slct:
                    st["done"] = True
                    continue
                pk = self.pf_key_identifier(st["union"] + slct)
                si = list(dict.fromkeys(st["union"] + slct + pk + self.business_columns(st["union"] + slct)))
                st["schemas"].append({"id": f"S{it + 1}", "columns": si, "valid": bool(si) and self.connected(si),
                                      "selector": "llm" if pick else "heuristic"})
                st["union"] = si
                drop = set(slct) - set(pk)
                st["rem"] = [(c, s) for c, s in st["rem"] if c not in drop]
        full = self.ms.columns()
        out = []
        for st, n in zip(states, notes):
            n += st.get("_notes", [])
            fallback = not any(s["valid"] for s in st["schemas"])
            if fallback:
                n.append("no valid filtered schema -> full schema")
            out.append(FilterResult(st["kws"], st["rtrv"], st["values"], st["schemas"], full, fallback, n))
        return out


def column_recall(gold: List[str], cols: List[str]) -> float:
    g = {c.lower() for c in gold}
    return 1.0 if not g else len(g & {c.lower() for c in cols}) / len(g)


def schema_json(res: FilterResult) -> str:
    return json.dumps(res.to_log(), ensure_ascii=False)
