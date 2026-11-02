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


# ============================================================================ XiYan-SQL pipeline prompts
# Templates follow alibaba/XiYan-SQL (XiYan-SQLTraining/data/data_utils/prompt_utils.py): 【用户问题】/【数据库schema】/
# 【参考信息】 blocks, a self-refine template with the failing SQL and its execution result, and a listwise
# selection template over candidates with their execution results.  Text is Vietnamese, dialect DuckDB.
KEYWORD_SYSTEM = (
    "Bạn là chuyên gia phân tích câu hỏi dữ liệu. Trích các từ khoá cần để tìm cột và giá trị trong cơ sở dữ liệu: "
    "khái niệm nghiệp vụ (doanh thu, số đơn, nợ, nhập hàng...), tên thực thể (mặt hàng, khách, nhóm, nhân viên), "
    "giá trị lọc (kênh, hình thức thanh toán...). Không lấy mốc thời gian. In JSON {\"keywords\": [...]}."
)
KEYWORD_SCHEMA = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
                  "required": ["keywords"]}


def keyword_user(question: str, evidence: str) -> str:
    return f"【Câu hỏi】\n{question}\n\n【Thông tin tham khảo】\n{evidence}\n\nJSON:"


COLUMN_SELECT_SYSTEM = (
    "Bạn là chuyên gia SQL. Từ 【Schema】 dưới đây, chọn MỌI cột cần để viết SQL trả lời câu hỏi: cột đo lường, cột lọc, "
    "cột ngày, cột điều kiện hợp lệ và cột nối bảng. Chỉ chọn cột có trong schema, ghi dạng bảng.cột. "
    "In JSON {\"columns\": [...]}."
)
COLUMN_SELECT_SCHEMA = {"type": "object", "properties": {"columns": {"type": "array", "items": {"type": "string"}}},
                        "required": ["columns"]}


def column_select_user(question: str, evidence: str, mschema: str) -> str:
    return f"{mschema}\n\n【Thông tin tham khảo】\n{evidence}\n\n【Câu hỏi】\n{question}\n\nJSON:"


NL2SQL_SYSTEM = (
    "Bạn là chuyên gia DuckDB SQL. Đọc và hiểu 【Schema】 (các view đã chuẩn hoá theo luật của cuộc thi), 【Thông tin "
    "tham khảo】 (luật nghiệp vụ, ngày hôm nay) rồi viết MỘT câu SQL trả lời 【Câu hỏi】. Chỉ dùng bảng/cột có trong "
    "schema; lọc kỳ theo cột ngày của bảng; doanh thu và số đơn chỉ tính is_valid = 1; không lấy bản ghi sau hôm nay. "
    "In SQL trong khối ```sql ... ```."
)
# multi-format generation styles (XiYan Fig. 3): one model, three writing preferences
SQL_FORMAT_HINTS = {
    "base": "",
    "cte": "Viết theo cấu trúc: đặt các dòng đã lọc trong một CTE (WITH ... AS (...)), tính tổng hợp ở câu SELECT ngoài.",
    "alias": "Viết theo phong cách: từ khoá chữ thường, mỗi bảng có bí danh ngắn (o, l, p...), ghi rõ bí danh trước cột.",
}


def nl2sql_user(question: str, mschema: str, evidence: str, fmt: str = "base", shots: List[Dict[str, str]] = ()) -> str:
    ex = "\n\n".join(f"【Câu hỏi】\n{s['question']}\n```sql\n{s['sql']}\n```" for s in shots if s.get("sql"))
    hint = SQL_FORMAT_HINTS.get(fmt, "")
    return (f"【Câu hỏi】\n{question}\n\n【Schema】\n{mschema}\n\n【Thông tin tham khảo】\n{evidence}\n"
            f"{(chr(10) + '【Ví dụ】' + chr(10) + ex + chr(10)) if ex else ''}"
            f"{(chr(10) + hint + chr(10)) if hint else ''}\n【Câu hỏi】\n{question}\n\n```sql")


SELF_REFINE_SYSTEM = (
    "Bạn là chuyên gia DuckDB SQL. 【SQL】 trước đó chưa trả lời đúng 【Câu hỏi】. Dựa vào 【Schema】, 【Thông tin tham khảo】 "
    "và 【Kết quả thực thi】 của SQL sai, hãy viết lại một SQL đúng. In SQL trong khối ```sql ... ```."
)


def self_refine_user(question: str, mschema: str, evidence: str, sql: str, exec_info: str) -> str:
    return (f"【Schema】\n{mschema}\n\n【Thông tin tham khảo】\n{evidence}\n\n【Câu hỏi】\n{question}\n\n【SQL】\n{sql}\n\n"
            f"【Kết quả thực thi】\n{exec_info}\n\n【SQL đã sửa】\n```sql")


SELECT_SYSTEM = (
    "Bạn là chuyên gia DuckDB SQL. Với 【Câu hỏi】 có nhiều 【SQL】 ứng viên cùng 【Kết quả thực thi】 (tối đa 10 dòng). "
    "So sánh các ứng viên, phân tích khác biệt (điều kiện is_valid, cột ngày và khoảng thời gian, phép JOIN có nhân bản dòng, "
    "doanh thu hay tiền đã trả, LIMIT...) dựa trên 【Schema】 và 【Thông tin tham khảo】, rồi chọn MỘT ứng viên đúng."
)


def candidate_block(i: int, sql: str, exec_info: str) -> str:
    return f"Ứng viên {i}\n【SQL】\n{sql}\n【Kết quả thực thi】\n{exec_info}"


def select_user(question: str, mschema: str, evidence: str, candidates: List[str]) -> str:
    ids = ", ".join(str(i) for i in range(1, len(candidates) + 1))
    return (f"【Schema】\n{mschema}\n\n【Thông tin tham khảo】\n{evidence}\n\n【Câu hỏi】\n{question}\n\n==========\n\n"
            + "\n\n".join(candidates) + f"\n\nỨng viên được chọn là (một trong {ids}). In JSON {{\"choice\": <số>}}:")


SELECT_SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]}
