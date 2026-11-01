"""Facts extracted from a shop README: data end date, unit conversions, conventions."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..textnorm import fold, fold_key
from .roles import parse_readme_columns


@dataclass
class ReadmeInfo:
    text: str
    columns: Dict[str, Dict[str, str]] = field(default_factory=dict)
    data_end: Optional[dt.date] = None
    conversions: List[Tuple[str, str, float, str]] = field(default_factory=list)  # (product, big_unit, factor, base_unit)
    bundle_rules: List[Tuple[str, str, float, str]] = field(default_factory=list)
    conventions: Dict[str, str] = field(default_factory=dict)  # bold title -> text
    factor_is_chain: bool = False   # 'hệ số ghi theo đơn vị ngay dưới'

    def has(self, *words: str) -> bool:
        f = fold(self.text)
        return any(fold(w) in f for w in words)


_CONV_RE = re.compile(r"^\s*-\s+(.+?):\s*1\s+(.+?)\s*=\s*([0-9.,]+)\s+(.+?)\s*$")


def parse_readme(text: str) -> ReadmeInfo:
    info = ReadmeInfo(text=text, columns=parse_readme_columns(text))
    m = re.search(r"Dữ liệu có đến hết\s+(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        info.data_end = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    for line in text.splitlines():
        mm = _CONV_RE.match(line)
        if mm:
            try:
                f = float(mm.group(3).replace(",", "."))
            except ValueError:
                continue
            info.conversions.append((mm.group(1).strip(), mm.group(2).strip(), f, mm.group(4).strip()))
        mc = re.match(r"^\s*-\s+\*\*(.+?)\*\*\s*:\s*(.+)$", line)
        if mc and not re.match(r"^\s*-\s+\*\*.+?\*\*\s*:\s*(mã|tên|số|ngày|đơn|giá|thành|trạng|nhóm|kênh)", line):
            info.conventions[mc.group(1).strip()] = mc.group(2).strip()
    # "A, B, C bán theo bó, 1 bó = 10 cây"
    for mb in re.finditer(r"\*\*Hàng theo bó\*\*:\s*(.+?) bán theo (\S+), 1 \S+ = (\d+) (\S+?)[;.]", text):
        for name in mb.group(1).split(", "):
            info.bundle_rules.append((name.strip(), mb.group(2), float(mb.group(3)), mb.group(4)))
    if re.search(r"hệ số ghi theo đơn vị ngay dưới|bằng bấy nhiêu đơn vị ngay dưới", text):
        info.factor_is_chain = True
    return info
