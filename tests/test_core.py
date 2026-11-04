"""Unit tests that need no competition data (run: python -m pytest -q tests)."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamate.engine.answer import Answer, _mtd_prev, _prev_period  # noqa: E402
from datamate.nlu.timeparse import Period, month_period, parse_time  # noqa: E402
from datamate.scorer import penalized_accuracy, score_one  # noqa: E402
from datamate.select.selector import Candidate, Selector  # noqa: E402
from datamate.textnorm import fold, normalize_question, repair_fold  # noqa: E402
from datamate.values import parse_datetime, parse_money, parse_qty, round_half_up  # noqa: E402

D = dt.date


def test_money_formats():
    cases = {"560,000": 560000, "30.500đ": 30500, "25k": 25000, "1tr440": 1440000, "1tr050": 1050000,
             "398,8604tr": 398860400, "3,255tr": 3255000, "29200k": 29200000, "1.016.000đ": 1016000,
             "45,000.0000": 45000, "129 000": 129000, "-30.500đ": -30500, "-45k": -45000, "7,5k": 7500}
    for s, v in cases.items():
        assert parse_money(s) == v, s


def test_dates_and_qty():
    assert parse_datetime("1 thg 6 2025").date() == D(2025, 6, 1)
    assert parse_datetime(45689.0).date() == D(2025, 2, 1)
    assert parse_datetime("45748").date() == D(2025, 4, 1)
    assert parse_datetime("1/1", 2026).date() == D(2026, 1, 1)
    assert parse_qty("1 bao 25kg") == (1.0, "bao 25kg")
    assert parse_qty("2 xe 4m3") == (2.0, "xe 4m3")


def test_rounding():
    assert round_half_up(2.5) == 3 and round_half_up(160128.5) == 160129 and round_half_up(-2.5) == -3
    assert round_half_up(1.005, 2) == 1.01


def test_normalisation():
    assert normalize_question("Th7/2026 mat hang nao ban chay nhat the?").startswith("thang 7/2026")
    assert "quy 1/2026" in normalize_question("Q1/2026 tiem co bao nhieu don?")
    assert "doanh thu" in normalize_question("dthu hqua bnhiu")
    assert "hom qua" in normalize_question("dthu hqua bnhiu")
    assert repair_fold("tieenf bans hangf").split() == ["tien", "ban", "hang"]
    assert fold("Bạc Xỉu (M)") == "bac xiu (m)"


def test_time_rules():
    today = D(2026, 3, 4)  # Wednesday
    ti = parse_time(normalize_question("doanh thu tuần rồi"), today)
    assert ti.periods[0].start == D(2026, 2, 23) and ti.periods[0].end == D(2026, 3, 1)
    ti = parse_time(normalize_question("doanh thu tuần này"), today)
    assert ti.periods[0].start == D(2026, 3, 2) and ti.periods[0].end == today
    ti = parse_time(normalize_question("7 ngày gần nhất"), today)
    assert ti.periods[0].start == D(2026, 2, 26)
    ti = parse_time(normalize_question("doanh thu tháng 11"), today)  # month without year after today -> last year
    assert ti.periods[0].start == D(2025, 11, 1)
    p = _mtd_prev(Period(D(2026, 3, 1), D(2026, 3, 31), "this_month", "mtd"))
    assert p.end == D(2026, 2, 28)
    assert _prev_period(month_period(2026, 1)).start == D(2025, 12, 1)


def test_scorer():
    ref = {"type": "list", "answer": ["Cà phê sữa (M)", "Bạc xỉu"], "alias": [["SP0003"], []]}
    assert score_one({"type": "list", "answer": ["SP0003", "bạc xỉu"]}, ref) == 1
    assert score_one({"type": "list", "answer": ["Bạc xỉu", "Cà phê sữa (M)"]}, ref) == -1
    assert score_one({"type": "abstain"}, ref) == 0
    assert score_one({"type": "number", "answer": 48250000}, {"type": "number", "answer": 48250000.0}) == 1
    assert score_one({"type": "string", "answer": "Sting dầu 330ml"}, {"type": "string", "answer": "Sting dâu 330ml"}) == -1
    tab = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 1], ["B", 2]]}}
    assert score_one({"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["B", 2], ["A", 1]]}}, tab) == 1
    pa, c = penalized_accuracy({"1": {"type": "number", "answer": 1}, "2": {"type": "number", "answer": 3}},
                               {"1": {"type": "number", "answer": 1}, "2": {"type": "number", "answer": 2},
                                "3": {"type": "number", "answer": 2}})
    assert abs(pa - 0.0) < 1e-9 and c == {"correct": 1, "wrong": 1, "abstain": 1}


def test_selector_threshold():
    s = Selector(None, tau=0.6)
    a, _ = s.select([Candidate("G1_rule", Answer("number", 5), "revenue")])
    assert a.type == "number"
    a, _ = s.select([Candidate("G4_sql", Answer("number", 5))])
    assert a.type == "abstain"                      # a lone weak generator is not trusted
    a, _ = s.select([Candidate("G4_sql", Answer("number", 5)), Candidate("G3_rewrite", Answer("number", 5), "revenue")])
    assert a.type == "number"                       # one LLM: its more reliable route speaks for it
    a, _ = s.select([Candidate("G1_rule", Answer("number", 5), "revenue"),
                     Candidate("G3_rewrite", Answer("number", 6), "revenue"),
                     Candidate("G2_frame", Answer("number", 6), "revenue")])
    assert a.type in ("number", "abstain")
