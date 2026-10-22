"""记忆类统计：时间窗解析 / 收集 / 确定性渲染（spec 2026-10-14-memory-stats-design）。"""
from datetime import datetime, timezone

import pytest

from app.core.memory_stats import resolve_period
from app.memory.store import to_local

NOW = datetime(2026, 10, 14, 7, 30, tzinfo=timezone.utc)  # 本机（UTC+8）2026-10-14 周三 15:30


def _local(dt):
    local = to_local(dt)
    return (local.date().isoformat(), local.strftime("%H:%M"))


def test_this_month_starts_at_first_day_local_midnight():
    s, e = resolve_period("this_month", now=NOW)
    assert _local(s) == ("2026-10-01", "00:00")
    assert _local(e) == ("2026-11-01", "00:00")  # 左闭右开


def test_last_month_ends_at_this_month_start():
    s, e = resolve_period("last_month", now=NOW)
    assert _local(s) == ("2026-09-01", "00:00")
    assert _local(e) == ("2026-10-01", "00:00")


def test_this_week_starts_monday_local():
    s, _ = resolve_period("this_week", now=NOW)  # 2026-10-14 是周三
    assert _local(s) == ("2026-10-12", "00:00")


def test_last_7_days_includes_today():
    s, e = resolve_period("last_7_days", now=NOW)
    assert _local(s) == ("2026-10-08", "00:00")
    assert _local(e) == ("2026-10-15", "00:00")


def test_last_month_crosses_year_boundary():
    jan = datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)
    s, e = resolve_period("last_month", now=jan)  # 本机 2026-01-05 → 上月=2025-12
    assert _local(s) == ("2025-12-01", "00:00")
    assert _local(e) == ("2026-01-01", "00:00")


def test_custom_end_day_is_inclusive():
    s, e = resolve_period("custom", start="2026-10-01", end="2026-10-03", now=NOW)
    assert _local(s) == ("2026-10-01", "00:00")
    assert _local(e) == ("2026-10-04", "00:00")  # end 含当日 → 次日 00:00


def test_all_covers_far_past():
    s, _ = resolve_period("all", now=NOW)
    assert s.year <= 1980


def test_invalid_inputs_raise():
    for args in (
        {"period": "last_year"},
        {"period": "custom"},
        {"period": "custom", "start": "昨", "end": "2026-10-03"},
        {"period": "custom", "start": "2026-10-03", "end": "2026-10-01"},
    ):
        with pytest.raises(ValueError):
            resolve_period(now=NOW, **args)
