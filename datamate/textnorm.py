"""Vietnamese text normalisation for noisy, everyday questions.

Everything downstream matches in *folded* space: lower-case, no diacritics, 'đ' -> 'd'.
On top of that we repair Telex typing slips ("tieenf" -> "tien", "dduwowcj" -> "duoc"),
expand chat abbreviations / teen-code / regional words ("dthu", "bnhiu", "hnay", "mik",
"rứa", "mô"...), and canonicalise date shorthands ("Th7/2026", "6-2025", "Q1/26").
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional, Set

_D_MAP = str.maketrans({"đ": "d", "Đ": "d"})


@lru_cache(maxsize=200000)
def fold(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.translate(_D_MAP).lower()
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def fold_key(s: str) -> str:
    """Fold and drop punctuation/space: used to compare names ("BẠC XỈU (M)" == "Bac xiu (m)")."""
    return re.sub(r"[^0-9a-z]+", "", fold(s))


def fold_words(s: str) -> str:
    """Fold and turn punctuation into single spaces (keeps tokens)."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z%]+", " ", fold(s))).strip()


# ---------------------------------------------------------------- Telex repair
_TONE_TAIL = set("sfrxj")
_VOWEL_MODS = [("aa", "a"), ("aw", "a"), ("ee", "e"), ("oo", "o"), ("ow", "o"), ("uw", "u"), ("dd", "d")]

# accent-less syllables that are valid Vietnamese words in this domain; extended at runtime
_BASE_VOCAB = set("""
a ai an ang anh ao ba bac ban bang banh bao bat bay be ben bi bia bien biet binh bo boi bon bong bot bu bua bui
ca cac cach cai cam can canh cao cap cat cau cay chai chan chang chao chat chay che chen chi chia chien chieu chin
chinh cho choi chon chu chua chuan chuc chung chuyen co coc coi com con cong cu cua cuc cui cung cuoi cuon cuong
da dai dan dang danh dao dap dat dau day de den deo dem den di dia diem dien dieu dinh do doan doc doi don dong
du dua duc dung duoc duong duy em gan gao gas gi gia giai giam gian giao gio giua giup gium goi gop gui ha hai
ham han hang hanh hao hat hay he hen het hieu hinh ho hoa hoan hoat hoc hoi hom hon hong hop hu hua huy huong
it keo kem ken kenh ket khac khach khai kham khi kho khoa khoan khoang khoi khong khu kia kien kiem kieu kim ky
la lai lam lan lang lanh lao lau le len lien lieu linh lo loai loc loi lon long luc lui luon luong luot ly
ma mai man mang mat mau may me men mi mien minh mo moc moi mon mot mua muc muoi muon my na nao nam nang nay
ne nen ngay nghi ngoai ngoc ngon ngu nguoi nguyen nha nhan nhanh nhap nhat nhau nhe nhi nhieu nhieu nhom nhu
nhung no noi nuoc o on ong pha phai phan phat phi phien phieu phong phu phuong qua quan quang quay que qui quy
ra rang rau re rieng roi rong rot sa sach sai san sang sao sap sau se si so soi son song sua sui sung suot ta
tai tam tan tang tao tat tau te tem ten thai tham than thang thanh thao thap that thay the them then thep thi
thich thiet thieu thit tho thoai thoi thom thong thu thua thuan thuc thue thuoc thuong ti tiec tien tiep tieu
tim tinh to toan toi tom ton tong tra trai tram tran trang tre tren tri trinh tro troi trong truoc truong tu
tua tuan tuc tui tung tuoi tuong tuyet tuyp uong va vai van vang vao vat vay ve vi viec vien viet vinh vo voi
vu vua vui vuong xa xac xam xang xanh xao xay xe xem xet xin xoai xuat xuong y yen
""".split())


def _strip_telex(tok: str) -> str:
    t = tok
    if len(t) >= 3 and t[-1] in _TONE_TAIL:
        t = t[:-1]
    for a, b in _VOWEL_MODS:
        t = t.replace(a, b)
    # 'uow' / 'uwow' style leftovers
    t = t.replace("w", "")
    return t


_SYL = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
                  r"(uye|uya|uyu|ieu|yeu|uoi|uou|oai|oay|uay|uai|oao|oeo|ua|ue|ui|uo|uu|uy|ia|ie|iu|ye|ai|ao|au|ay|eo|eu|oa|oe|oi|a|e|i|o|u|y)"
                  r"(ch|ng|nh|c|m|n|p|t)?$")


def is_syllable(tok: str) -> bool:
    return bool(_SYL.match(tok))


def telex_repair(tok: str, vocab: Set[str]) -> str:
    if not tok.isalpha() or tok in vocab:
        return tok
    if is_syllable(tok) and tok not in ("bans",):
        return tok
    t = _strip_telex(tok)
    if t != tok and (t in vocab or is_syllable(t)):
        return t
    return tok


def repair_fold(q: str) -> str:
    """fold_words + Telex repair, without abbreviation expansion (used for value linking)."""
    return " ".join(telex_repair(t, _BASE_VOCAB) for t in fold_words(q).split())


# ---------------------------------------------------------------- abbreviations
# token-level (folded) replacements; value may be several words or "" (filler)
_ABBR = {
    "dthu": "doanh thu", "dthuu": "doanh thu", "doanhthu": "doanh thu", "revenue": "doanh thu", "dso": "doanh so",
    "bn": "bao nhieu", "bnhiu": "bao nhieu", "bnhieu": "bao nhieu", "bnh": "bao nhieu", "baonhieu": "bao nhieu",
    "bnhiue": "bao nhieu", "bnhieuu": "bao nhieu", "bnhju": "bao nhieu",
    "nhiu": "nhieu", "nhju": "nhieu", "nhieuu": "nhieu", "nhiuu": "nhieu",
    "hnay": "hom nay", "hnai": "hom nay", "homnay": "hom nay", "hqua": "hom qua", "homqua": "hom qua",
    "hwa": "qua", "wa": "qua", "hum": "hom",
    "dc": "duoc", "dk": "duoc", "duoc": "duoc", "dcj": "duoc", "dduoc": "duoc",
    "ko": "khong", "k0": "khong", "hok": "khong", "kg0": "khong", "khog": "khong", "kh": "khach hang",
    "sp": "san pham", "mh": "mat hang", "ncc": "nha cung cap", "nv": "nhan vien", "tb": "trung binh",
    "sl": "so luong", "dh": "don hang", "hd": "hoa don", "kl": "khach le", "trc": "truoc", "trk": "truoc",
    "truoc": "truoc", "vs": "voi", "z": "vay", "v": "vay", "zay": "vay", "dzay": "vay", "j": "gi", "ji": "gi",
    "mik": "minh", "mk": "minh", "mjk": "minh", "mjnh": "minh", "cko": "cho", "chok": "cho",
    "jup": "giup", "jum": "gium", "giumf": "gium", "jump": "giup", "bit": "biet", "bjet": "biet",
    "kon": "con", "ko0": "khong", "wan": "quan", "nhug": "nhung", "tih": "tinh", "kua": "cua", "cuar": "cua",
    "ni": "nay", "nay": "nay", "mo": "nao", "mi": "may", "rua": "vay", "nhut": "nhat", "nhat": "nhat", "tieng": "tien",
    "dong": "dong", "huong": "huong", "lanh": "lanh", "chec": "check", "chek": "check", "tops": "top", "tob": "top",
    "diem": "diem", "dien": "dien", "voucher": "voucher", "vouchers": "voucher", "ship": "ship",
    "pt": "phan tram", "ptram": "phan tram", "%": "phan tram", "phantram": "phan tram",
    "order": "don", "orders": "don", "bill": "don", "bills": "don", "total": "tong", "tongg": "tong",
    "thangs": "thang", "thg": "thang", "thag": "thang", "thng": "thang", "tuan": "tuan",
    "ngayf": "ngay",
    "rùi": "roi", "rui": "roi", "r": "roi",
    "dau": "dau", "toi": "toi",
    "lun": "luon", "lum": "luon", "nhiu": "nhieu",
    "qa": "qua", "ntn": "nhu the nao", "sao": "sao",
    # frequent typing slips / teen-code of common words
    "trog": "trong", "trg": "trong", "nuot": "luot", "lout": "luot", "nhta": "nhat", "nhatt": "nhat", "nhaat": "nhat",
    "bna": "ban", "bnaf": "ban", "lne": "len", "nheiu": "nhieu", "nhieeu": "nhieu", "dup": "giup", "dum": "giup",
    "cog": "cong", "tog": "tong", "duco": "duoc", "dduoc": "duoc", "ngui": "nguoi", "nguoj": "nguoi", "zi": "vay",
    "dza": "vay", "dzi": "vay", "mun": "muon", "xo": "so", "xố": "so", "bin": "binh", "nguog": "nguong",
    "dag": "dang", "kac": "cac", "doanhthu": "doanh thu", "dthuu": "doanh thu", "bnhieu": "bao nhieu",
    "thg": "thang", "nhug": "nhung", "zo": "vao",
}
# filler / politeness words that carry no meaning for the parser
_FILLERS = {"a", "ah", "ạ", "oi", "ơi", "e", "em", "anh", "chi", "ad", "admin", "alo", "ê", "he", "hen", "nhe", "nha",
            "nhi", "ha", "ta", "the", "thế", "vay", "do", "đó", "nhe", "nhá", "hả", "hè", "hen", "ne", "nè",
            "check", "xem", "giup", "gium", "coi", "cho", "minh", "toi", "biet", "hoi", "voi", "di", "ik", "xin",
            "xiu", "dum", "giu", "jum", "luon", "vs", "roi"}
# words that are fillers only at the very end of a question
_TAIL_FILLERS = {"vay", "z", "v", "nhi", "ha", "the", "do", "a", "nha", "nhe", "he", "hen", "ne", "ta", "roi",
                 "rua", "khong", "ko", "hong", "chua"}


def expand_tokens(tokens: List[str], vocab: Optional[Set[str]] = None) -> List[str]:
    vocab = vocab or _BASE_VOCAB
    out: List[str] = []
    for t in tokens:
        if t in _ABBR:
            rep = _ABBR[t]
            out.extend(rep.split())
            continue
        r = telex_repair(t, vocab)
        if r != t and r in _ABBR:
            out.extend(_ABBR[r].split())
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------- dates / numbers in questions
_DATE_PATTERNS = [
    # Th7/2026, T8/2025, th 7 / 2026, thg7-2025
    (re.compile(r"\b(?:thang|th|thg|t)\s*\.?\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})\b"), lambda m: f" thang {int(m.group(1))}/{_yy(m.group(2))} "),
    # Q1/2026, quy 3 nam 2025
    (re.compile(r"\bq\s*([1-4])\s*[/\-.]\s*(\d{4}|\d{2})\b"), lambda m: f" quy {m.group(1)}/{_yy(m.group(2))} "),
    (re.compile(r"\bquy\s*([1-4])\s*(?:/|-|nam)\s*(\d{4}|\d{2})\b"), lambda m: f" quy {m.group(1)}/{_yy(m.group(2))} "),
    (re.compile(r"\bthang\s*(\d{1,2})\s*nam\s*(\d{4})\b"), lambda m: f" thang {int(m.group(1))}/{m.group(2)} "),
]


def _yy(s: str) -> str:
    return s if len(s) == 4 else str(2000 + int(s))


def canon_dates(s: str) -> str:
    for rx, fn in _DATE_PATTERNS:
        s = rx.sub(fn, s)
    # bare month-year at sentence start or after a space: "6-2025", "12-2025", "06/2026", "1/26", "5/26"
    def bare(m):
        mo, yr = int(m.group(1)), m.group(2)
        if not 1 <= mo <= 12:
            return m.group(0)
        if len(yr) == 2 and not (20 <= int(yr) <= 35):
            return m.group(0)
        before = m.string[max(0, m.start() - 6):m.start()]
        if re.search(r"(quy|ngay)\s*$", before):
            return f" {mo}/{_yy(yr)} "
        return f" thang {mo}/{_yy(yr)} "
    s = re.sub(r"(?<![\d/\-.])(\d{1,2})\s*[/\-]\s*(20\d{2}|2[0-9]|3[0-5])(?![\d/\-.])", bare, s)
    s = re.sub(r"\bthang\s+thang\b", "thang", s)
    # 'th nay', 'th truoc', 'th roi' -> thang
    s = re.sub(r"\bth\b(?=\s+(nay|truoc|roi|sau|lien|nao|vua|ke|kia|\d))", "thang", s)
    s = re.sub(r"\b(?:thang|th)(\d{1,2})\b(?!\s*/)", lambda m: f"thang {int(m.group(1))}" if 1 <= int(m.group(1)) <= 12 else m.group(0), s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_question(q: str, vocab: Optional[Set[str]] = None) -> str:
    """Folded, abbreviation-expanded, date-canonical form of a question."""
    s = fold(q)
    # 'dt' / 'đt': phone after 'máy / qua / chiếc / gọi', otherwise revenue (doanh thu)
    s = re.sub(r"\b(may|qua|chiec|goi|ban qua|so)\s+dt\b", r"\1 dien thoai", s)
    s = re.sub(r"\bdt\b", "doanh thu", s)
    s = s.replace("%", " % ")
    # glue errors like "nhieudon" are rare; split letters from digits only for unit-ish tokens
    s = re.sub(r"(?<=[a-z])(?=\d{1,2}/\d{2,4})", " ", s)
    s = re.sub(r"[?!,;:\"“”…()\[\]{}]+", " ", s)
    s = re.sub(r"(?<=\D)\.(?=\D)|(?<=\D)\.$", " ", s)
    toks = s.split()
    toks = expand_tokens(toks, vocab)
    s = " ".join(toks)
    s = canon_dates(s)
    s = re.sub(r"\b(bua|hom) (nao|mo)\b", "ngay nao", s)
    return s


def strip_fillers(s: str) -> str:
    toks = s.split()
    while toks and toks[-1] in _TAIL_FILLERS:
        toks.pop()
    return " ".join(toks)


def build_vocab(texts: Iterable[str]) -> Set[str]:
    v = set(_BASE_VOCAB)
    for t in texts:
        for w in re.findall(r"[a-z]+", fold(t)):
            if len(w) <= 7:
                v.add(w)
    return v
