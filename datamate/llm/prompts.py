"""Prompt templates (Vietnamese) for the LLM generators."""
from __future__ import annotations

from typing import Dict, List

from ..nlu.frame import INTENTS

SYSTEM_BASE = (
    "Bạn là trợ lý phân tích dữ liệu bán hàng cho cửa hàng nhỏ ở Việt Nam. Bạn trả lời câu hỏi tiếng Việt đời thường "
    "dựa DUY NHẤT trên dữ liệu của cửa hàng và bộ luật (rulebook) được cung cấp. Không dùng hiểu biết bên ngoài. "
    "Nếu dữ liệu không đủ để trả lời (thiếu loại thông tin, kỳ nằm ngoài dữ liệu, câu hỏi chủ quan) thì phải nói là "
    "không trả lời được. Trả lời sai bị trừ điểm, bỏ trống không bị trừ: chỉ trả lời khi chắc chắn."
)

REWRITE_SYSTEM = SYSTEM_BASE + (
    "\nNhiệm vụ: viết lại câu hỏi cho chuẩn, đủ dấu tiếng Việt, không viết tắt, không tiếng lóng, giữ nguyên ý. "
    "Giữ NGUYÊN VĂN tên mặt hàng / khách / nhóm hàng như trong danh sách tên của cửa hàng. "
    "Viết mốc thời gian theo dạng chuẩn: 'tháng M/YYYY', 'quý Q/YYYY', 'ngày DD/MM/YYYY', 'hôm nay', 'hôm qua', "
    "'tuần này', 'tuần trước', 'tháng này', 'N ngày gần nhất', 'từ đầu năm đến hết hôm nay', 'đến hết ngày DD/MM/YYYY'. "
    "Chỉ in ra một dòng là câu hỏi đã viết lại."
)

FRAME_SYSTEM = SYSTEM_BASE + (
    "\nNhiệm vụ: chuyển câu hỏi thành một khung truy vấn JSON (DSL) để hệ thống tính toán chính xác. "
    "Chỉ in ra JSON hợp lệ, không giải thích."
)

SQL_SYSTEM = SYSTEM_BASE + (
    "\nNhiệm vụ: viết MỘT câu truy vấn DuckDB SQL trên các view đã chuẩn hoá (đã áp luật đơn hợp lệ, đơn vị cơ bản) "
    "để trả lời câu hỏi. Chỉ dùng các bảng/cột có trong schema. Lọc kỳ theo ngày đặt đơn, không lấy bản ghi sau hôm nay. "
    "Doanh thu và số đơn chỉ tính đơn is_valid = 1. In ra SQL trong khối ```sql ... ```."
)

ANSWERABILITY_SYSTEM = SYSTEM_BASE + (
    "\nNhiệm vụ: quyết định câu hỏi có trả lời được bằng dữ liệu của cửa hàng hay không. "
    "Suy nghĩ ngắn gọn: câu hỏi cần loại thông tin nào, dữ liệu có cột/bảng đó không, kỳ hỏi có nằm trong khoảng dữ liệu không, "
    "câu hỏi có chủ quan hay cần kiến thức ngoài không. Sau đó in JSON: "
    '{"answerable": true|false|"unsure", "reason": "..."}.'
)

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": sorted(INTENTS)},
        "time": {"type": "string"},
        "compare": {"type": "string", "enum": ["none", "prev", "yoy", "mtd", "shift", "tet_yoy", "before_tet", "explicit"]},
        "product": {"type": ["string", "null"]},
        "customer": {"type": ["string", "null"]},
        "group": {"type": ["string", "null"]},
        "customer_group": {"type": ["string", "null"]},
        "ingredient": {"type": ["string", "null"]},
        "staff": {"type": ["string", "null"]},
        "channel": {"type": ["string", "null"]},
        "payment": {"type": ["string", "null"]},
        "shift": {"type": ["string", "null"]},
        "rx": {"type": ["boolean", "null"]},
        "size": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "k": {"type": ["integer", "null"]},
        "by": {"type": ["string", "null"]},
        "threshold": {"type": ["number", "null"]},
        "threshold_op": {"type": ["string", "null"]},
        "n": {"type": ["integer", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["intent", "time"],
}


def intents_doc() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())


def vocab_block(v: Dict[str, List[str]], max_items: int = 120) -> str:
    out = []
    for k, items in v.items():
        if not items:
            continue
        shown = items[:max_items]
        more = f" … (+{len(items) - max_items})" if len(items) > max_items else ""
        out.append(f"{k}: " + "; ".join(shown) + more)
    return "\n".join(out)


def rewrite_user(question: str, as_of: str, vocab: str, shots: List[Dict[str, str]]) -> str:
    ex = "\n".join(f"Câu gốc: {s['q']}\nViết lại: {s['canon']}" for s in shots)
    return (f"Tên trong dữ liệu cửa hàng:\n{vocab}\n\nHôm nay: {as_of}\n\nVí dụ:\n{ex}\n\n"
            f"Câu gốc: {question}\nViết lại:")


def frame_user(question: str, as_of: str, rulebook_digest: str, vocab: str, shots: List[Dict[str, str]]) -> str:
    ex = "\n".join(f"Câu hỏi: {s['q']}\nJSON: {s['frame']}" for s in shots)
    return (f"Các intent:\n{intents_doc()}\n\nLuật chính:\n{rulebook_digest}\n\nTên trong dữ liệu:\n{vocab}\n\n"
            f"Trường 'time' ghi mốc thời gian dạng chuẩn tiếng Việt (vd 'tháng 7/2025', 'tuần trước', 'đến hết ngày 05/08/2026'). "
            f"Tên mặt hàng/khách/nhóm phải chép đúng từ danh sách. Câu không trả lời được: intent='unanswerable' và ghi reason.\n\n"
            f"Ví dụ:\n{ex}\n\nHôm nay: {as_of}\nCâu hỏi: {question}\nJSON:")


def sql_user(question: str, as_of: str, mschema: str, rulebook_digest: str, answer_type: str, shots: List[Dict[str, str]]) -> str:
    ex = "\n\n".join(f"Câu hỏi: {s['q']}\n```sql\n{s['sql']}\n```" for s in shots if s.get("sql"))
    return (f"{mschema}\n\nLuật chính:\n{rulebook_digest}\n\nHôm nay: {as_of}. Kiểu đáp án cần: {answer_type}.\n"
            f"{('Ví dụ:' + chr(10) + ex) if ex else ''}\n\nCâu hỏi: {question}\n")


def answerability_user(question: str, as_of: str, data_profile: str) -> str:
    return f"Dữ liệu cửa hàng có:\n{data_profile}\n\nHôm nay: {as_of}\nCâu hỏi: {question}\n"
