"""Regression tests for phrasings first seen on the public split (no data needed)."""
import datetime as dt
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamate.domains.books import Contract
from datamate.domains.repair import job_items, question_job
from datamate.engine.answer import Answer
from datamate.nlu.consistency import check
from datamate.nlu.timeparse import parse_time
from datamate.textnorm import normalize_question as nq

TODAY = dt.date(2026, 3, 18)


def _periods(q, today=TODAY):
    return [(p.start, p.end, p.kind) for p in parse_time(nq(q), today).periods]


def test_today_only_bounds_a_longer_period():
    assert _periods("Tính luôn hnay thì 7 ngày gần nhất quán có bnhiu đơn") == \
        [(dt.date(2026, 3, 12), TODAY, "range")]
    assert _periods("doanh thu từ mùng 1 tới hết hôm nay")[0][2] == "mtd"
    assert _periods("Tới hôm nay, mấy đơn đặt tháng 2/2026 đã giao 2 đợt")[0][2] == "month"


def test_month_ranges_and_last_year_month():
    p = _periods("T1/2026 tới T3/2026, tháng nào bán nhiều tiền nhất")
    assert p == [(dt.date(2026, 1, 1), dt.date(2026, 3, 31), "month_range")]
    p = _periods("Doanh thu tháng 2/2026 so với tháng 2 năm rồi")      # relative to today (rulebook)
    assert p == [(dt.date(2026, 2, 1), dt.date(2026, 2, 28), "month"), (dt.date(2025, 2, 1), dt.date(2025, 2, 28), "month")]
    assert len(_periods("Dthu th4, th5, th6/2025 mỗi tháng")) == 3


def test_normalisation_variants():
    assert "may ngay" in nq("có mấy bữa bán được từ 10 củ trở lên") and "10 trieu" in nq("10 củ")
    assert "don" in nq("bình quân mỗi o đờ")
    assert nq("tháng sau có cao hơn tháng trước") .startswith("thang sau")     # not 'tháng sáu'
    assert "thang 3" in nq("Tháng ba quán có mấy đơn")
    assert "cao nhat" in nq("dthu caonhất")


def _fr(intent, periods=1, filters=None):
    return NS(intent=intent, params={}, filters=filters or {}, periods=[NS(start=i, end=i) for i in range(periods)])


def test_guard_blocks_form_conflicts():
    assert check(nq("nhóm gia vị mặt hàng nào có doanh thu cao nhất"), _fr("revenue"), Answer("number", 1))
    assert check(nq("thuốc kê đơn tháng ni mấy % tổng doanh thu"), _fr("revenue"), Answer("number", 1))
    assert check(nq("dthu th7, th8 lần lượt bnhiu"), _fr("revenue"), Answer("number", 1))
    assert check(nq("có mấy bữa bán được từ 9 triệu"), _fr("revenue"), Answer("number", 1))
    # product names with '%' / 'nhôm' are not cues
    assert check(nq("bán ra bao nhiêu tuýp Acyclovir kem 5%"), _fr("units", filters={"product": "x"}), Answer("number", 1)) is None
    assert check(nq("bán được mấy cái Thước thủy nhôm"), _fr("units", filters={"product": "x"}), Answer("number", 1)) is None


def test_repair_job_taxonomy():
    assert job_items("vá ruột, thay nhớt láp") == ["va_ruot", "nhot_lap"]
    assert job_items("thay bóng đèn, ắc quy") == ["den", "ac_quy"]          # not 'sửa điện'
    assert job_items("rửa kim phun") == ["kim_phun"]
    assert question_job("thay dau lap cho bao nhieu luot xe") == "nhot_lap"
    assert question_job("bao nhieu phieu co ruot") == "thay_ruot"            # plain 'ruột' = new tube


def test_instalment_schedule():
    c = Contract(dt.date(2025, 7, 2), "33", "A", 19_980_000, 4_380_000, 6, 2_600_000, 3,
                 pays=[(dt.date(2025, 8, 5), 2_600_000)])
    sch = c.schedule()
    assert sch[0][0] == dt.date(2025, 8, 3) and sch[-1][0] == dt.date(2026, 1, 3)
    assert c.due_count(dt.date(2025, 9, 3)) == 2 and c.due_count(dt.date(2025, 9, 3), inclusive=False) == 1
    assert c.overdue(dt.date(2025, 9, 4)) == 2_600_000
    assert c.outstanding(dt.date(2025, 9, 4)) == 13_000_000
