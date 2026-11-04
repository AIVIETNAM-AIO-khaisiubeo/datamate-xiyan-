"""Unit tests (no competition data) for the XiYan-SQL components: M-Schema keys, PFKeyIdentifier,
Schema Filter guards, SQL de-formalisation, SQL checks, candidate reorganisation, family-aware voting."""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamate.engine.answer import Answer
from datamate.llm.mschema import build_mschema
from datamate.llm.schema_filter import SchemaFilter
from datamate.llm.sql_mutations import drop_is_valid, fanout_join, shift_month
from datamate.llm.sql_pipeline import SQLCandidate, SQLRunner, deformalize
from datamate.nlu.timeparse import Period
from datamate.select.selection_model import reorganize
from datamate.select.selector import Candidate, Selector
from datamate.select.sql_checks import lint

D = dt.date


def _views():
    orders = pd.DataFrame({"order_id": ["A", "B", "C"], "order_date": [D(2025, 7, 1), D(2025, 7, 2), D(2025, 8, 1)],
                           "status": ["completed", "cancelled", "completed"], "is_valid": [1, 0, 1],
                           "revenue": [100.0, 50.0, 70.0], "total_due": [110.0, 50.0, 70.0]})
    lines = pd.DataFrame({"order_id": ["A", "A", "B", "C"], "line_no": [1, 2, 1, 1],
                          "order_date": [D(2025, 7, 1)] * 2 + [D(2025, 7, 2), D(2025, 8, 1)],
                          "product_key": ["p1", "p2", "p1", "p2"], "product_name": ["Trà", "Cà phê", "Trà", "Cà phê"],
                          "kind": ["item"] * 4, "amount": [60.0, 40.0, 50.0, 70.0], "is_valid": [1, 1, 0, 1]})
    products = pd.DataFrame({"product_key": ["p1", "p2"], "name": ["Trà", "Cà phê"]})
    return {"orders": orders, "sales_lines": lines, "products": products}


def test_keys_verified_and_rendered():
    V = _views()
    ms = build_mschema("t", V)
    assert ms.primary_key("orders") == ["order_id"]
    assert ms.primary_key("sales_lines") == ["order_id", "line_no"]          # composite key kept
    assert ["sales_lines", "order_id", "orders", "order_id"] in ms.foreign_keys
    V2 = dict(V)
    V2["sales_lines"] = V["sales_lines"].assign(line_no=1)                    # not unique any more
    ms2 = build_mschema("t", V2)
    assert ms2.primary_key("sales_lines") == [] and ms2.issues
    txt = ms.to_mschema(["orders.revenue", "orders.order_date"])
    assert "# Table: orders" in txt and "sales_lines" not in txt and "【Foreign keys】" not in txt


def test_pfkey_completes_composite_keys_and_join_path():
    V = _views()
    sf = SchemaFilter(build_mschema("t", V), V)
    keys = sf.pf_key_identifier(["sales_lines.amount", "orders.status"])
    assert {"sales_lines.order_id", "sales_lines.line_no", "orders.order_id"} <= set(keys)
    # business-condition columns travel with the table
    assert {"sales_lines.is_valid", "sales_lines.kind", "sales_lines.order_date"} <= set(sf.business_columns(["sales_lines.amount"]))


def test_schema_filter_keeps_measure_and_guards():
    V = _views()
    sf = SchemaFilter(build_mschema("t", V), V)
    res = sf.run("doanh thu tháng 7/2025 là bao nhiêu")
    s1 = set(res.schemas[0]["columns"])
    assert "orders.revenue" in s1 and "orders.is_valid" in s1 and "orders.order_date" in s1
    assert res.schemas[0]["valid"]


def test_deformalize_and_runner():
    a = "SELECT  SUM(revenue) FROM orders WHERE is_valid = 1 AND status = 'A B';"
    b = "select sum(revenue)\nfrom orders where is_valid=1 and status='A B'"
    assert deformalize(a) == deformalize(b)
    r = SQLRunner(_views())
    assert r.run("SELECT SUM(revenue) FROM orders WHERE is_valid = 1")[0][0] == 170.0
    try:
        r.run("DROP TABLE orders")
        assert False
    except ValueError:
        pass


def test_sql_checks_flag_near_misses_not_correct_sql():
    p = [Period(D(2025, 7, 1), D(2025, 7, 31), "m", "month")]
    good = "SELECT SUM(revenue) FROM orders WHERE is_valid = 1 AND order_date BETWEEN DATE '2025-07-01' AND DATE '2025-07-31'"
    nq = "doanh thu thang 7/2025 la bao nhieu"
    assert lint(good, nq, p, D(2025, 9, 1)) == []
    assert ("missing_is_valid", 0.9) in lint(drop_is_valid(good), nq, p, D(2025, 9, 1))
    assert any(f == "period_mismatch" for f, _ in lint(shift_month(good), nq, p, D(2025, 9, 1)))
    assert any(f == "join_fanout" for f, _ in lint(fanout_join(good), nq, p, D(2025, 9, 1)))
    assert any(f == "paid_not_revenue" for f, _ in lint(good.replace("revenue", "total_due"), nq, p, D(2025, 9, 1)))


def _c(gen, v, sql):
    return SQLCandidate("q", sql, "S1", 5, gen, "base", "m", ok=True, value=v, answer_type="number")


def test_reorganize_algorithm3():
    cs = [_c("G4_base", 1, "select 1"), _c("G4_cte", 1, "with x as (select 1) select * from x"), _c("G4_alias", 2, "select 2")]
    shown, mode, top = reorganize(cs, {"G4_base": 0.6, "G4_cte": 0.5, "G4_alias": 0.5})
    assert mode == "consensus" and top == 2 and shown[0].generator == "G4_base"
    shown, mode, _ = reorganize(cs[:1] + cs[2:], {})
    assert mode == "consensus" or mode == "one_per_cluster"
    shown, mode, _ = reorganize(cs[:2], {})
    assert mode == "consistent" and shown[0].sql == "select 1"          # shortest of a consistent set


def test_one_model_family_is_one_vote():
    sel = Selector(None, tau=0.6)
    fam = {"family": "qwen"}
    three = [Candidate(g, Answer("number", 5), None, dict(fam)) for g in ("G4_base", "G4_cte", "G4_alias")]
    _, info3 = sel.select(three)
    _, info1 = sel.select(three[:1])
    assert abs(info3["p"] - info1["p"]) < 1e-9                          # formats of one model are not 3 votes
    flagged = [Candidate("G4_base", Answer("number", 5), None, {"family": "qwen", "flags": [("missing_is_valid", 0.9)]})]
    _, infof = sel.select(flagged)
    assert infof["p"] < info1["p"]


def test_few_shot_excludes_own_shop_and_folds_are_shared():
    from datamate.folds import fold_of, fold_shops
    from datamate.llm.generators import ExampleBank
    bank = ExampleBank([{"q": "doanh thu tháng 7", "intent": "revenue", "biz": "a"},
                        {"q": "doanh thu tháng 8", "intent": "revenue", "biz": "b"}])
    assert {x["biz"] for x in bank.topk("doanh thu tháng 7", 5)} == {"a", "b"}
    bank.exclude = {"a"}
    assert {x["biz"] for x in bank.topk("doanh thu tháng 7", 5)} == {"b"}
    shops = [f"biz_{i}" for i in range(101, 131)]
    f = fold_of(shops, 5)
    assert fold_shops(shops, 0, 5) == {b for b in shops if f[b] == 0} and len(fold_shops(shops, 0, 5)) == 6
