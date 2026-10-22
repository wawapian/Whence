"""记忆统计：收集与确定性渲染（Task 2）。"""
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.memory_stats import STATS_MISSING_SAMPLES, STATS_ROWS_MAX, collect, render
from app.memory import store
from app.models.db import Fact, init_db

# 冻结"现在"：本机（UTC+8）2026-10-14 周三 15:30
NOW = datetime(2026, 10, 14, 7, 30, tzinfo=timezone.utc)


def _session(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    return Session(engine)


def _add(session, *, amount=None, currency=None, occurred=datetime(2026, 10, 3, 4, 0, tzinfo=timezone.utc),
         summary="午饭", type_="consumption", raw_content=None):
    raw = store.insert_raw(session, kind="text", content=raw_content or summary)
    session.flush()
    fields = {"amount": amount, "currency": currency}
    fact = Fact(id=store.new_id(), type=type_, summary=summary, fields=fields,
                valid_from=occurred, raw_id=raw.id)
    session.add(fact)
    session.flush()
    return fact


def test_spending_sums_by_currency_and_orders_desc(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=32.0, currency="CNY", occurred=datetime(2026, 10, 3, 4, 0, tzinfo=timezone.utc),
         summary="午饭")
    _add(s, amount=28.0, currency="CNY", occurred=datetime(2026, 10, 5, 4, 0, tzinfo=timezone.utc),
         summary="打车")
    _add(s, amount=20.0, currency="USD", occurred=datetime(2026, 10, 6, 4, 0, tzinfo=timezone.utc),
         summary="订阅")
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    assert out.totals == {"CNY": 60.0, "USD": 20.0}
    assert out.count == 3
    assert [r.summary for r in out.rows] == ["订阅", "打车", "午饭"]  # 时间倒序


def test_missing_amount_disclosed_not_counted(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=32.0, currency="CNY", summary="午饭")
    _add(s, amount=None, summary="点外卖", occurred=datetime(2026, 10, 6, 4, 0, tzinfo=timezone.utc))
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    assert out.count == 1 and out.totals == {"CNY": 32.0}
    assert out.missing == 1 and out.missing_samples == ["点外卖"]
    text = render(out)
    assert "另有 1 条记录未识别到金额，未计入" in text
    assert "点外卖" in text


def test_state_fact_amount_excluded(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=5000.0, currency="CNY", summary="卡里余额", type_="bank_card")
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    assert out.count == 0 and out.totals == {}
    assert render(out) == "该时间范围内没有符合条件的记录。"


def test_negative_and_zero_amount_included(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=-38.0, currency="CNY", summary="退款")
    _add(s, amount=0.0, currency="CNY", summary="凑单")
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    assert out.totals == {"CNY": -38.0} and out.count == 2
    assert "-38.00" in render(out) and "0.00" in render(out)


def test_currency_missing_gets_own_bucket(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=12.0, currency=None, summary="买水")
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    assert out.totals == {"未标币种": 12.0}


def test_count_kind_counts_events(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=None, summary="去公园", type_="trip")
    _add(s, amount=None, summary="交报告", type_="todo")
    _add(s, amount=32.0, currency="CNY", summary="午饭")
    out = collect(s, period="this_month", start=None, end=None, kind="count", now=NOW)
    assert out.kind == "count" and out.count == 3 and out.rows == []
    assert "符合条件的记录：3 条" in render(out)


def test_keyword_matches_summary_or_raw(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=32.0, currency="CNY", summary="午饭", raw_content="和同事吃了牛肉面")
    _add(s, amount=28.0, currency="CNY", summary="出行",
         raw_content="打车去机场", occurred=datetime(2026, 10, 5, 4, 0, tzinfo=timezone.utc))
    out = collect(s, period="this_month", start=None, end=None, kind="spending",
                  keyword="打车", now=NOW)
    assert out.count == 1 and out.rows[0].summary == "出行"


def test_out_of_range_excluded(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=99.0, currency="CNY", summary="上月的", occurred=datetime(2026, 9, 20, 4, 0, tzinfo=timezone.utc))
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    assert out.count == 0
    out_last = collect(s, period="last_month", start=None, end=None, kind="spending", now=NOW)
    assert out_last.count == 1 and out_last.totals == {"CNY": 99.0}


def test_render_spending_template(tmp_path):
    s = _session(tmp_path)
    _add(s, amount=32.0, currency="CNY", summary="午饭")
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    lines = render(out).splitlines()
    assert lines[0] == "统计口径：按记录时间 2026-10-01 ~ 2026-10-31"
    assert lines[1] == "消费合计：CNY 32.00（1 笔）"
    assert lines[2] == "明细（最多 20 条，按时间倒序）："
    assert lines[3] == "1. 2026-10-03 32.00 CNY —— 午饭"
    assert STATS_ROWS_MAX == 20 and STATS_MISSING_SAMPLES == 3


def test_render_has_cross_currency_total_count(tmp_path):
    """终审 I1：跨币种时"几笔"必须由代码给出总数——不能让模型把各币种笔数相加（数字不经模型）。"""
    s = _session(tmp_path)
    _add(s, amount=32.0, currency="CNY", summary="午饭")
    _add(s, amount=28.0, currency="CNY", summary="打车", occurred=datetime(2026, 10, 4, 4, 0, tzinfo=timezone.utc))
    _add(s, amount=20.0, currency="USD", summary="订阅", occurred=datetime(2026, 10, 5, 4, 0, tzinfo=timezone.utc))
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    text = render(out)
    assert "共 3 笔" in text


def test_render_truncates_long_detail_list(tmp_path):
    s = _session(tmp_path)
    for i in range(25):
        _add(s, amount=1.0, currency="CNY", summary=f"第{i}笔",
             occurred=datetime(2026, 10, 3, 4, 0, tzinfo=timezone.utc))
    out = collect(s, period="this_month", start=None, end=None, kind="spending", now=NOW)
    text = render(out)
    assert "…共 25 条" in text
    assert text.count("\n") <= 2 + STATS_ROWS_MAX + 1 + 1  # 头部 3 行 + 20 明细 + 截断行（+尾行）
