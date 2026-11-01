"""Value linking: find products, customers, groups, ingredients, staff, suppliers and units
mentioned in a question (XiYan 'value retrieval', done with exact/fuzzy n-gram matching
over the shop's own vocabulary)."""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from ..ingest.builder import Biz, strip_phone
from ..textnorm import fold, fold_key, fold_words


def _fw(s: str) -> str:
    return " " + fold_words(s) + " "


class Linker:
    def __init__(self, biz: Biz):
        self.b = biz
        self.prod_names: Dict[str, List[str]] = defaultdict(list)   # folded words -> keys
        for k, cnt in biz._names_by_key.items():
            for n in cnt:
                self.prod_names[fold_words(n)].append(k)
        if not biz.products.empty:
            for k, n, c in zip(biz.products["key"], biz.products["name"], biz.products["code"]):
                if isinstance(n, str):
                    self.prod_names[fold_words(n)].append(k)
                if isinstance(c, str):
                    self.prod_names[fold_words(c)].append(k)
                al = None
        if not biz.products.empty and "aliases" in biz.products.columns:
            for k, al in zip(biz.products["key"], biz.products["aliases"]):
                if isinstance(al, str):
                    for a in re.split(r"[,;/]", al):
                        if a.strip():
                            self.prod_names[fold_words(a)].append(k)
        for k in list(self.prod_names):
            self.prod_names[k] = sorted(set(self.prod_names[k]))
        self.prod_sorted = sorted(self.prod_names, key=len, reverse=True)
        # customers
        self.cust_names: Dict[str, List[str]] = defaultdict(list)
        for k, n in getattr(biz, '_cust_name', {}).items():
            self.cust_names[fold_words(strip_phone(n))].append(k)
            self.cust_names[fold_words(n)].append(k)
        if not biz.orders.empty and "customer_name" in biz.orders.columns and "cust_key" in biz.orders.columns:
            for k, n in zip(biz.orders["cust_key"], biz.orders["customer_name"]):
                if k and isinstance(n, str):
                    fw = fold_words(strip_phone(n))
                    if k not in self.cust_names[fw]:
                        self.cust_names[fw].append(k)
        for k in list(self.cust_names):
            self.cust_names[k] = sorted(set(self.cust_names[k]))
        self.cust_sorted = sorted((c for c in self.cust_names if len(c) >= 5), key=len, reverse=True)
        # groups
        self.groups: Dict[str, str] = {}
        if not biz.products.empty and "group" in biz.products.columns:
            for g in biz.products["group"].dropna().unique():
                self.groups[fold_words(g)] = g
        if not self.groups and not biz.lines.empty and "product_group" in biz.lines.columns:
            for g in biz.lines["product_group"].dropna().unique():
                if isinstance(g, str):
                    self.groups[fold_words(g)] = g
        self.cust_groups: Dict[str, str] = {}
        O = biz.orders
        if not O.empty and "cust_group" in O.columns:
            for g in O["cust_group"].dropna().unique():
                if isinstance(g, str):
                    self.cust_groups[fold_words(g)] = g
        if not biz.customers.empty and "customer_group" in biz.customers.columns:
            for g in biz.customers["customer_group"].dropna().unique():
                if isinstance(g, str):
                    self.cust_groups.setdefault(fold_words(g), g)
        self.ingredients: Dict[str, str] = {}
        for code, (a, rx) in (getattr(biz, "_drug", {}) or {}).items():
            if isinstance(a, str) and a.strip():
                self.ingredients[fold_words(a)] = a
        self.staff: Dict[str, str] = {}
        if not biz.staff.empty:
            for c, n in zip(biz.staff.get("staff", [None] * len(biz.staff)), biz.staff.get("staff_name", [None] * len(biz.staff))):
                if isinstance(n, str):
                    self.staff[fold_words(n)] = c if isinstance(c, str) else n
        self.suppliers: Dict[str, str] = {}
        H = biz.purchases
        if not H.empty and "supplier_name" in H.columns:
            for n in H["supplier_name"].dropna().unique():
                self.suppliers[fold_words(n)] = n
        # units seen in the data
        units = set()
        L = biz.lines
        if not L.empty and "unit_eff" in L.columns:
            units |= {fold_words(u) for u in L["unit_eff"].dropna().unique() if isinstance(u, str)}
        for (k, u) in biz.unit_factor:
            units.add(fold_words(u) if not re.fullmatch(r"[0-9a-z]+", u) else u)
        for u in (getattr(biz, '_pbase', None) or {}).values():
            if isinstance(u, str):
                units.add(fold_words(u))
        self.units = {u for u in units if u}

    # ------------------------------------------------------------------
    def products(self, q: str) -> List[Tuple[str, List[str], int]]:
        """[(matched folded name, keys, position)] longest-first, non-overlapping."""
        s = _fw(q)
        found = []
        used = [False] * len(s)
        for name in self.prod_sorted:
            if len(name) < 3:
                continue
            pat = " " + name + " "
            i = s.find(pat)
            while i >= 0:
                if not any(used[i + 1:i + len(pat) - 1]):
                    for j in range(i + 1, i + len(pat) - 1):
                        used[j] = True
                    found.append((name, self.prod_names[name], i))
                    break
                i = s.find(pat, i + 1)
        found.sort(key=lambda x: x[2])
        return found

    def product_fuzzy(self, q: str, min_ratio: float = 0.86) -> Optional[Tuple[str, List[str], float]]:
        """Fuzzy fallback: best product name whose tokens nearly all appear in the question."""
        s = fold_words(q)
        toks = s.split()
        best = None
        for name in self.prod_sorted:
            nt = name.split()
            if len(nt) < 2:
                continue
            n = len(nt)
            for i in range(0, max(1, len(toks) - n + 1)):
                cand = " ".join(toks[i:i + n])
                r = SequenceMatcher(None, cand, name).ratio()
                if r >= min_ratio and (best is None or r > best[2] or (r == best[2] and len(name) > len(best[0]))):
                    best = (name, self.prod_names[name], r)
        return best

    def customers(self, q: str) -> List[Tuple[str, List[str], int]]:
        s = _fw(q)
        out = []
        for name in self.cust_sorted:
            i = s.find(" " + name + " ")
            if i >= 0:
                if any(i < p + len(n) + 1 and p < i + len(name) + 1 for n, _, p in out):
                    continue
                out.append((name, self.cust_names[name], i))
        return out

    def group(self, q: str) -> Optional[str]:
        s = _fw(q)
        best = None
        for fg, g in self.groups.items():
            if not fg:
                continue
            for pre in ("nhom hang hoa ", "nhom hang ", "nhom ", "cac mat hang nhom ", "mat hang nhom ", "loai "):
                if (" " + pre + fg + " ") in s:
                    if best is None or len(fg) > len(best[0]):
                        best = (fg, g)
        return best[1] if best else None

    def brand(self, q: str) -> Optional[str]:
        P = self.b.products
        if P.empty or "brand" not in P.columns:
            return None
        s = _fw(q)
        best = None
        for br in P["brand"].dropna().unique():
            fb = fold_words(str(br))
            if len(fb) >= 3 and (" " + fb + " ") in s and (best is None or len(fb) > len(fold_words(best))):
                best = str(br)
        return best

    def group_loose(self, q: str) -> Optional[str]:
        """Group named without the word 'nhóm' ('bỉm' -> 'bỉm tã', 'sữa bột')."""
        s = _fw(q)
        firsts = {}
        for fg, g in self.groups.items():
            if (" " + fg + " ") in s:
                return g
            w = fg.split()[0] if fg else ""
            firsts.setdefault(w, []).append(g)
        for w, gs in firsts.items():
            if len(w) >= 3 and len(gs) == 1 and (" " + w + " ") in s:
                return gs[0]
        return None

    def cust_group(self, q: str) -> Optional[str]:
        s = _fw(q)
        best = None
        for fg, g in self.cust_groups.items():
            for pre in ("nhom khach hang ", "nhom khach ", "khach nhom ", "khach thuoc nhom ", "khach hang nhom ",
                        "khach hang thuoc nhom ", "nhom ", "khach ", "khach hang "):
                if (" " + pre + fg + " ") in s:
                    if best is None or len(fg) > len(best[0]):
                        best = (fg, g)
        return best[1] if best else None

    def ingredient(self, q: str) -> Optional[str]:
        s = _fw(q)
        best = None
        for fi, a in self.ingredients.items():
            if (" hoat chat " + fi + " ") in s or (" chat " + fi + " ") in s:
                if best is None or len(fi) > len(best[0]):
                    best = (fi, a)
        return best[1] if best else None

    def staff_ref(self, q: str) -> Optional[str]:
        m = re.search(r"\b(nv\s*\d{2,4})\b", fold(q))
        if m:
            return m.group(1).replace(" ", "").upper()
        s = _fw(q)
        for fn, code in self.staff.items():
            if (" " + fn + " ") in s:
                return code
        return None

    def supplier(self, q: str) -> Optional[str]:
        s = _fw(q)
        for fn, n in sorted(self.suppliers.items(), key=lambda kv: -len(kv[0])):
            if (" " + fn + " ") in s:
                return n
        return None

    def unit_after(self, q: str, words=("so", "bao nhieu", "may", "nhieu")) -> Optional[str]:
        """Unit word right after a quantity word: 'bao nhiêu chai', 'số ly', 'mấy hộp'."""
        s = _fw(q)
        cands = sorted(self.units | _COMMON_UNITS, key=len, reverse=True)
        for w in words:
            for u in cands:
                if (" " + w + " " + u + " ") in s:
                    return u
        return None


_COMMON_UNITS = {"ly", "chai", "lon", "hop", "vien", "goi", "kg", "bao", "thung", "cay", "m3", "cai", "dia", "phan", "to",
                 "binh", "cuon", "tuyp", "lo", "vi", "ong", "mieng", "m", "m2", "tam", "bo", "xe 4m3", "tan", "loc",
                 "bich", "chiec", "may", "suat", "qua", "cuc", "thien", "o", "lit", "can", "nai", "trai", "ky"}
