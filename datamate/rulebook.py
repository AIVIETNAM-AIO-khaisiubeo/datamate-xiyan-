"""Split-specific rulebook reader.

The deterministic engine implements the default interpretations of the train/public
rulebooks.  Because the private rulebook may differ, every rule row is fingerprinted; the
agent only trusts an engine intent when all rule rows that intent depends on are present
and unchanged (or their parameters could be re-extracted, e.g. new shift hours).  Rows that
changed are reported so the selector can abstain / defer to the LLM generators, which get
the full rulebook text in their prompt.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .textnorm import fold

# rule row (folded first cell) -> intents that rely on it
DEPENDS = {
    "don tinh vao doanh thu": "*",
    "doanh thu mot don": "*",
    "don thuoc ky nao": "*",
    "so don": {"order_count", "aov", "cust_min_orders", "cust_min_orders_pct", "top_customers", "orders_min_amount",
               "products_min_orders"},
    "ti le don huy trong ky": {"cancel_rate"},
    "phan tram thay doi, tang truong": {"growth", "price_effect"},
    "doanh thu theo hinh thuc thanh toan": {"revenue", "share", "growth"},
    "doanh thu theo nhom khach": {"revenue", "share", "aov"},
    "tuan": "*",
    "thang nay": "*",
    "ca sang, ca chieu": {"revenue", "order_count", "growth"},
    "tet nam y, dip tet, mua tet": {"growth", "revenue", "top_products"},
    "ban chay": {"top_products", "units"},
    "hang loi nhat": {"top_products"},
    "xac dinh khach": {"top_customers", "cust_min_orders", "cust_min_orders_pct", "debtor_count", "top_debtors"},
    "nhan vien ban gioi nhat": {"top_staff"},
    "gia tri trung binh mot don": {"aov"},
    "no mot don tai ngay x": {"debt_total", "debtor_count", "top_debtors", "debt_aging", "debt_overdue"},
    "tuoi no": {"debt_aging", "debt_overdue"},
    "no them, tra no, thu no trong ky": {"debt_flow", "repayments"},
    "tien nhap hang trong ky": {"purchase_total", "top_suppliers"},
    "gia nhap binh quan mot mat hang trong thang": {"avg_purchase_price"},
    "gia tri hang ton kho": {"stock_value"},
    "lai gop": {"gross_profit", "top_products", "net_profit"},
    "lai rong mot thang": {"net_profit"},
    "gia ban x ngay d": {"price_on"},
    "hieu ung tang gia": {"price_effect"},
    "mat hang khong ban duoc trong ky (e)": {"unsold_products"},
    "mat hang het hang cuoi ngay": {"out_of_stock"},
    "hang het han trong n ngay toi": {"expiring"},
    "lo gan het han nhat cua mot thuoc": {"nearest_expiry"},
    "giao nhieu dot": {"delivery_trips"},
}


@dataclass
class RuleRow:
    section: str
    name: str
    text: str

    @property
    def key(self) -> str:
        return fold(re.sub(r"\*", "", self.name)).strip()

    @property
    def fp(self) -> str:
        return hashlib.md5(fold(self.text).encode()).hexdigest()[:10]


@dataclass
class Rulebook:
    text: str
    rows: List[RuleRow] = field(default_factory=list)
    shift_morning: Tuple[int, int] = (6, 13)
    shift_afternoon: Tuple[int, int] = (14, 21)
    week_start: int = 0       # Monday
    tet_from: Tuple[int, int] = (12, 23)   # lunar (month, day) of previous lunar year
    tet_to: Tuple[int, int] = (1, 10)
    changed: Set[str] = field(default_factory=set)
    unknown: Set[str] = field(default_factory=set)

    def row(self, key: str) -> Optional[RuleRow]:
        for r in self.rows:
            if r.key == key:
                return r
        return None

    def intent_trusted(self, intent: str) -> bool:
        for k in self.changed:
            dep = DEPENDS.get(k, set())
            if dep == "*" or (dep and intent in dep):
                return False
        return True


# Key-phrase assertions: the default each engine intent implements.  If the split's rulebook
# no longer contains the phrase (rule changed), dependent intents are no longer trusted.
CHECKS: List[Tuple[str, List[str]]] = [
    ("don tinh vao doanh thu", ["khong huy", "khong tra hang", "ke ca chi tra mot phan", "bo ca don"]),
    ("doanh thu mot don", ["thanh tien cac dong hang", "khong tru giam gia tren ca don", "khong cong thue vat", "khong gom phi giao hang"]),
    ("don thuoc ky nao", ["theo ngay dat"]),
    ("so don", ["chi tinh don khong huy, khong tra"]),
    ("ti le don huy trong ky", ["moi don dat trong ky", "lam tron 2 chu so"]),
    ("phan tram thay doi, tang truong", ["so voi ky truoc", "giam thi ra so am"]),
    ("tuan", ["bat dau thu hai"]),
    ("thang nay", ["tu ngay 1 den het hom nay", "cung so ngay"]),
    ("ca sang, ca chieu", ["theo gio tao don"]),
    ("tet nam y, dip tet, mua tet", ["23 thang chap", "mung 10 thang gieng"]),
    ("ban chay", ["so luong quy ve don vi co ban"]),
    ("hang loi nhat", ["tong lai gop"]),
    ("nhan vien ban gioi nhat", ["theo doanh thu"]),
    ("gia tri trung binh mot don", ["lam tron ve dong"]),
    ("no mot don tai ngay x", ["phan chua tra cua tong don"]),
    ("tuoi no", ["tinh tu ngay dat"]),
    ("tien nhap hang trong ky", ["ngay phieu", "bo phieu huy"]),
    ("gia tri hang ton kho", ["gia von mot don vi co ban ghi trong danh muc hang"]),
    ("lai gop", ["gia von ghi tren dong ban"]),
    ("hieu ung tang gia", ["n ngay ke tu d", "n ngay ngay truoc d"]),
]

_CHECK_DEPS = {
    "don tinh vao doanh thu": "*", "doanh thu mot don": "*", "don thuoc ky nao": "*", "tuan": "*", "thang nay": "*",
}


def _check(rb: "Rulebook"):
    for key, phrases in CHECKS:
        r = rb.row(key)
        if r is None:
            rb.changed.add(key)
            continue
        t = fold(re.sub(r"\*", "", r.text))
        if not all(p in t for p in phrases):
            rb.changed.add(key)


def parse_rulebook(text: str) -> Rulebook:
    rb = Rulebook(text=text)
    section = ""
    for line in text.splitlines():
        if line.startswith("#"):
            section = line.strip("# ").strip()
            continue
        if line.startswith("|") and not re.match(r"^\|\s*-", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] not in ("Luật", "Cách nói", "Cách ghi", "Từ", "type"):
                rb.rows.append(RuleRow(section, cells[0], " | ".join(cells[1:])))
    f = fold(text)
    m = re.search(r"ca sang (\d{1,2}):\d{2}\s*[–-]\s*(\d{1,2}):\d{2}", f)
    if m:
        rb.shift_morning = (int(m.group(1)), int(m.group(2)))
    m = re.search(r"ca chieu (\d{1,2}):\d{2}\s*[–-]\s*(\d{1,2}):\d{2}", f)
    if m:
        rb.shift_afternoon = (int(m.group(1)), int(m.group(2)))
    m = re.search(r"tu ngay duong cua \*?\*?(\d{1,2}) thang chap", f)
    if m:
        rb.tet_from = (12, int(m.group(1)))
    m = re.search(r"mung (\d{1,2}) thang gieng", f)
    if m:
        rb.tet_to = (1, int(m.group(1)))
    _check(rb)
    return rb


# ---------------------------------------------------------------------------- rows the engine has never seen
# The engine (G1) implements the rows of the rulebooks read while it was written (train, then public).  A private
# rulebook may define new concepts ("Khách ... nhất", "Trung bình mỗi ...").  A question that names such a concept
# must not be forced onto the nearest known intent: the rule routes are flagged and only the SQL route, which reads
# the row as evidence, may answer (or the agent abstains).
KNOWN_ROWS_PATH = "artifacts/known_rulebook_rows.json"


def concept_tokens(key: str) -> List[str]:
    """Content words of a folded row name ('khach sop nhat' -> ['khach', 'sop', 'nhat']); placeholders X / N / D
    and punctuation are dropped."""
    return [t for t in re.sub(r"[^a-z0-9]+", " ", key).split() if len(t) > 1]


def row_id(row: "RuleRow") -> str:
    return " ".join(concept_tokens(row.key))


def mentions(nq: str, toks: List[str], max_gap: int = 2) -> bool:
    """The folded question contains the row's words in order, at most max_gap other words between two of them."""
    q = nq.split()
    if len(toks) < 2:
        return False
    for i, w in enumerate(q):
        if w != toks[0]:
            continue
        j, ok = i, True
        for t in toks[1:]:
            nxt = next((k for k in range(j + 1, min(len(q), j + 2 + max_gap)) if q[k] == t), None)
            if nxt is None:
                ok = False
                break
            j = nxt
        if ok:
            return True
    return False


def new_rows(rb: "Rulebook", known: Set[str]) -> List["RuleRow"]:
    """Rows of this split's rulebook that the engine never saw (and specific enough to be matched in a question)."""
    return [r for r in rb.rows if len(concept_tokens(r.key)) >= 2 and row_id(r) not in known]
