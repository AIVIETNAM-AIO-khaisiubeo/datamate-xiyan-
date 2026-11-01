"""The query-frame DSL shared by every generator (rule parser, LLM frame generator).

A Frame is the 'multi-format' intermediate representation in the XiYan sense: the rule
parser and the LLM both emit Frames; the executor turns a Frame into a typed answer.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .timeparse import Period

INTENTS = {
    "revenue": "doanh thu (tổng thành tiền dòng hàng của đơn hợp lệ) trong kỳ",
    "revenue_multi": "danh sách doanh thu theo từng kênh / hình thức thanh toán nêu trong câu",
    "order_count": "số đơn hợp lệ (không huỷ, không trả)",
    "cancel_count": "số đơn huỷ",
    "cancel_rate": "% đơn huỷ trên mọi đơn",
    "aov": "giá trị trung bình một đơn",
    "units": "số lượng bán (đơn vị cơ bản)",
    "growth": "% thay đổi so với kỳ so sánh",
    "diff": "chênh lệch tuyệt đối kỳ hỏi − kỳ so",
    "share": "% tỉ trọng doanh thu của một phần trên toàn shop",
    "top_products": "xếp hạng mặt hàng",
    "top_customers": "xếp hạng khách",
    "top_staff": "xếp hạng nhân viên",
    "top_suppliers": "xếp hạng nhà cung cấp",
    "best_day": "ngày có doanh thu/số đơn cao nhất",
    "best_month": "tháng có doanh thu cao nhất trong các tháng nêu",
    "monthly_list": "doanh thu từng kỳ theo thứ tự",
    "increasing": "doanh thu có tăng liên tục không",
    "price_on": "đơn giá một đơn vị cơ bản ngày D",
    "price_effect": "% thay đổi lượng bán N ngày sau vs N ngày trước lần tăng giá",
    "purchase_total": "tiền nhập hàng trong kỳ",
    "avg_purchase_price": "giá nhập bình quân gia quyền",
    "purchase_minus_sales": "số nhập trừ số bán",
    "stock": "tồn kho cuối ngày",
    "stock_split": "tồn kho theo [đơn vị lớn, đơn vị cơ bản lẻ]",
    "stock_value": "giá trị tồn kho theo giá vốn",
    "out_of_stock": "số mặt hàng hết hàng",
    "expiring": "lô sắp hết hạn",
    "nearest_expiry": "hạn dùng lô gần hết hạn nhất",
    "debt_total": "tổng nợ khách tại ngày X",
    "debtor_count": "số khách còn nợ",
    "top_debtors": "bảng khách nợ nhiều nhất",
    "debt_aging": "nợ theo nhóm tuổi",
    "debt_overdue": "nợ của đơn đặt quá N ngày",
    "debt_flow": "[nợ thêm, trả được] của khách trong kỳ",
    "repayments": "tiền trả nợ thu được trong kỳ",
    "cust_min_orders": "số khách có ≥N đơn",
    "cust_min_orders_pct": "% khách có ≥N đơn trong số khách đã mua",
    "products_min_orders": "số mặt hàng có mặt trong ≥N đơn",
    "unsold_products": "số mặt hàng không bán được trong kỳ",
    "days_revenue": "số ngày có doanh thu thoả ngưỡng",
    "orders_min_amount": "số đơn có tiền hàng thoả ngưỡng",
    "gross_profit": "lãi gộp",
    "net_profit": "lãi ròng",
    "delivery_trips": "số đợt giao",
    "unanswerable": "dữ liệu không trả lời được",
}


@dataclass
class Frame:
    intent: str
    periods: List[Period] = field(default_factory=list)
    snapshot: Optional[dt.date] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    k: Optional[int] = None
    by: Optional[str] = None
    unit: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    answer_type: Optional[str] = None
    source: str = "rule"
    reason: str = ""
    confidence: float = 1.0

    def to_json(self) -> str:
        d = asdict(self)
        d["periods"] = [[str(p.start), str(p.end), p.kind] for p in self.periods]
        d["snapshot"] = str(self.snapshot) if self.snapshot else None
        return json.dumps(d, ensure_ascii=False, default=str)

    def signature(self) -> str:
        """Canonical string used for candidate clustering."""
        ps = ";".join(f"{p.start}:{p.end}" for p in self.periods)
        fs = ";".join(f"{k}={self.filters[k]}" for k in sorted(self.filters) if self.filters[k] is not None)
        return f"{self.intent}|{ps}|{self.snapshot}|{fs}|{self.k}|{self.by}|{self.unit}|{sorted(self.params.items())}"
