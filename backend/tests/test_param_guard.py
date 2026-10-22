"""param_guard：以问题原文为尺子的矛盾检测（spec 2026-10-16-stats-param-guard §4）。"""
from datetime import datetime, timezone

from app.core import param_guard

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)  # 本机 +08 → 2026-10-16


def test_expected_window_single_phrase():
    got = param_guard.expected_window("上个月我一共花了多少钱？", now=NOW)
    assert got is not None
    period, start, end = got
    assert period == "last_month"
    assert param_guard.range_label(start, end) == "2026-09-01 ~ 2026-09-30"


def test_expected_window_multiple_targets_is_none():
    assert param_guard.expected_window("上个月和这个月各花了多少", now=NOW) is None


def test_expected_window_no_phrase_is_none():
    assert param_guard.expected_window("前一阵子花了多少", now=NOW) is None


def test_expected_window_last_week_not_in_table():
    assert param_guard.expected_window("上周花了多少", now=NOW) is None


def test_period_mismatch_reports_expected():
    msg = param_guard.period_mismatch("上个月花了多少钱", "this_month", None, None, now=NOW)
    assert msg is not None
    assert "last_month" in msg and "2026-09-01 ~ 2026-09-30" in msg


def test_period_mismatch_custom_with_correct_dates_passes():
    assert param_guard.period_mismatch(
        "上个月花了多少钱", "custom", "2026-09-01", "2026-09-30", now=NOW
    ) is None


def test_period_mismatch_no_phrase_passes():
    assert param_guard.period_mismatch("前一阵子花了多少", "this_month", None, None, now=NOW) is None


def test_quoted_entities_extracts_and_dedups():
    q = "「样例信息有限公司」的岗位，「样例信息有限公司」有哪些"
    assert param_guard.quoted_entities(q) == ["样例信息有限公司"]


def test_quoted_entities_ignores_single_char_and_unquoted():
    assert param_guard.quoted_entities("样例公司有哪些岗位？「甲」") == []


def test_entity_mismatch_detects_rewritten_name():
    msg = param_guard.entity_mismatch("「样例信息有限公司」的岗位", "样例科技有限公司")
    assert msg is not None and "样例信息有限公司" in msg


def test_entity_mismatch_allows_substring_and_unrelated():
    assert param_guard.entity_mismatch("「样例信息有限公司」的岗位", "样例信息") is None
    assert param_guard.entity_mismatch("「样例信息有限公司」的岗位", "调度员") is None
    assert param_guard.entity_mismatch("样例公司有哪些岗位", "样例科技有限公司") is None


def test_with_stats_caliber_appends_when_missing():
    out = param_guard.with_stats_caliber(
        "这个月花了 483 元",
        [{"label": "按记录时间 2026-10-01 ~ 2026-10-31", "probe": "2026-10-01 ~ 2026-10-31"}],
    )
    assert out.startswith("这个月花了 483 元")
    assert "统计口径：" in out


def test_with_stats_caliber_skips_when_already_present():
    answer = "统计口径：按记录时间 2026-10-01 ~ 2026-10-31，共 3 笔"
    assert param_guard.with_stats_caliber(
        answer,
        [{"label": "按记录时间 2026-10-01 ~ 2026-10-31", "probe": "2026-10-01 ~ 2026-10-31"}],
    ) == answer


def test_with_stats_caliber_no_calls_is_identity():
    assert param_guard.with_stats_caliber("普通回答", []) == "普通回答"
