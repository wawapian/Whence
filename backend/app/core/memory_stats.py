"""记忆类统计：时间窗解析（本模块）→ 收集 → 确定性渲染（spec 2026-10-14-memory-stats-design）。

数字一律由代码计算与渲染；模型只负责选参数与组织语言（项目铁律：数字不经模型）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.memory.store import to_local
from app.models.db import Fact, RawEvidence
from app.models.schemas import EVENT_TYPES

STATS_ROWS_MAX = 20       # 明细渲染上限（超出给「…共 N 条」）
STATS_MISSING_SAMPLES = 3  # 「未识别到金额」示例条数

PERIODS = ("this_week", "last_7_days", "this_month", "last_month",
           "last_30_days", "this_year", "all", "custom")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _add_month(day: date, delta: int) -> date:
    """月首日 ± delta 月（跨年安全）。"""
    months = day.year * 12 + (day.month - 1) + delta
    return date(months // 12, months % 12 + 1, 1)


def resolve_period(period: str, start: str | None = None, end: str | None = None,
                   *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """period → `(start, end)`（UTC aware，左闭右开 `start <= t < end`）。

    区间边界按**本机时区**（与展示层 to_local 同口径）计算，再折算 UTC 与库内比较；
    非法 period / custom 缺参或日期不可解析 / end 早于 start → `ValueError`（调用方转可读文案）。
    """
    now = now or datetime.now(timezone.utc)
    local_now = to_local(now)
    today = local_now.date()
    tz = local_now.tzinfo

    def at(day: date) -> datetime:
        return datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(timezone.utc)

    if period == "this_week":
        monday = today - timedelta(days=today.weekday())
        return at(monday), at(monday + timedelta(days=7))
    if period == "last_7_days":
        return at(today - timedelta(days=6)), at(today + timedelta(days=1))
    if period == "this_month":
        first = today.replace(day=1)
        return at(first), at(_add_month(first, 1))
    if period == "last_month":
        first = _add_month(today.replace(day=1), -1)
        return at(first), at(_add_month(first, 1))
    if period == "last_30_days":
        return at(today - timedelta(days=29)), at(today + timedelta(days=1))
    if period == "this_year":
        return at(date(today.year, 1, 1)), at(date(today.year + 1, 1, 1))
    if period == "all":
        return _EPOCH, at(today + timedelta(days=1))
    if period == "custom":
        if not start or not end:
            raise ValueError("custom 需要 start 与 end（YYYY-MM-DD）")
        try:
            start_day = date.fromisoformat(start)
            end_day = date.fromisoformat(end)
        except ValueError as exc:
            raise ValueError(f"日期不可解析：{start!r} / {end!r}") from exc
        if end_day < start_day:
            raise ValueError("end 早于 start")
        return at(start_day), at(end_day + timedelta(days=1))  # end 含当日
    raise ValueError(f"未知 period：{period!r}")


@dataclass
class SpendRow:
    fact_id: str
    occurred: datetime
    amount: float
    currency: str
    summary: str


@dataclass
class StatsOutcome:
    kind: str                  # spending | count
    start: datetime
    end: datetime
    rows: list[SpendRow] = field(default_factory=list)   # spending：金额明细（时间倒序）
    totals: dict[str, float] = field(default_factory=dict)  # 币种 → 合计
    count: int = 0             # spending=笔数；count=条数
    missing: int = 0           # spending：区间内事件类但未识别到金额的条数（不计入合计）
    missing_samples: list[str] = field(default_factory=list)


def collect(session: Session, *, period: str, start: str | None, end: str | None, kind: str,
            keyword: str | None = None, type_filter: str | None = None,
            now: datetime | None = None) -> StatsOutcome:
    """按时间窗收集记忆统计（spec §4）：事件类事实、可选类型/关键词、spending 分币种合计。

    已进回收站的记录（其 raw 软删）不参与统计——与检索可见性同口径。
    """
    if kind not in ("spending", "count"):
        raise ValueError(f"未知 kind：{kind!r}")
    start_dt, end_dt = resolve_period(period, start, end, now=now)
    stmt = (
        select(Fact)
        .outerjoin(RawEvidence, RawEvidence.id == Fact.raw_id)
        .where(
            Fact.valid_from >= start_dt,
            Fact.valid_from < end_dt,
            Fact.type.in_(EVENT_TYPES),  # 状态类（如银行卡余额）不算"花销/次数"
            or_(RawEvidence.id.is_(None), RawEvidence.deleted_at.is_(None)),
        )
        .order_by(Fact.valid_from.desc())
    )
    if type_filter:
        stmt = stmt.where(Fact.type == type_filter)
    if keyword and keyword.strip():
        kw = keyword.strip()
        stmt = stmt.where(or_(Fact.summary.contains(kw), RawEvidence.content.contains(kw)))
    facts = session.scalars(stmt).all()

    if kind == "count":
        return StatsOutcome(kind="count", start=start_dt, end=end_dt, count=len(facts))

    rows: list[SpendRow] = []
    totals: dict[str, float] = {}
    missing = 0
    missing_samples: list[str] = []
    for fact in facts:
        fields = fact.fields or {}
        amount = fields.get("amount")
        if amount is None:
            missing += 1
            if len(missing_samples) < STATS_MISSING_SAMPLES:
                missing_samples.append(fact.summary or "")
            continue
        currency = fields.get("currency") or "未标币种"
        rows.append(SpendRow(fact_id=fact.id, occurred=fact.valid_from, amount=float(amount),
                             currency=currency, summary=fact.summary or ""))
        totals[currency] = totals.get(currency, 0.0) + float(amount)
    return StatsOutcome(kind="spending", start=start_dt, end=end_dt, rows=rows, totals=totals,
                        count=len(rows), missing=missing, missing_samples=missing_samples)


def _range_label(outcome: StatsOutcome) -> str:
    """区间标签（本机日期，含末日）；`all` 用「截至 X」。"""
    last = to_local(outcome.end - timedelta(days=1)).date().isoformat()
    if outcome.start.year <= 1970:
        return f"截至 {last}"
    return f"{to_local(outcome.start).date().isoformat()} ~ {last}"


def render(outcome: StatsOutcome) -> str:
    """确定性渲染（数字与文案全部由代码产出；模型只组织语言）。"""
    if outcome.kind == "count":
        if outcome.count == 0:
            return "该时间范围内没有符合条件的记录。"
        return f"统计口径：按记录时间 {_range_label(outcome)}\n符合条件的记录：{outcome.count} 条"

    if outcome.count == 0 and not outcome.missing:
        return "该时间范围内没有符合条件的记录。"
    lines = [f"统计口径：按记录时间 {_range_label(outcome)}"]
    per_currency = Counter(row.currency for row in outcome.rows)
    if outcome.totals:
        lines.append("消费合计：" + "；".join(
            f"{cur} {outcome.totals[cur]:.2f}（{per_currency[cur]} 笔）"
            for cur in sorted(outcome.totals)
        ))
    else:
        lines.append("消费合计：无可计入金额的记录")
    if len(outcome.totals) > 1:  # 跨币种：笔数总须由代码给出（别让模型把各币种笔数相加；终审 I1）
        lines.append(f"共 {outcome.count} 笔")
    if outcome.rows:
        lines.append(f"明细（最多 {STATS_ROWS_MAX} 条，按时间倒序）：")
        for i, row in enumerate(outcome.rows[:STATS_ROWS_MAX], start=1):
            day = to_local(row.occurred).strftime("%Y-%m-%d")
            lines.append(f"{i}. {day} {row.amount:.2f} {row.currency} —— {row.summary}")
        if len(outcome.rows) > STATS_ROWS_MAX:
            lines.append(f"…共 {len(outcome.rows)} 条")
    if outcome.missing:
        sample = "；".join(s for s in outcome.missing_samples if s)
        lines.append(f"另有 {outcome.missing} 条记录未识别到金额，未计入"
                     + (f"（例：{sample}）" if sample else ""))
    return "\n".join(lines)
