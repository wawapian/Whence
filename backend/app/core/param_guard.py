"""统计类问答的参数守卫：以「问题原文」为尺子做**矛盾检测**（spec 2026-10-16-stats-param-guard）。

只做矛盾检测，不做语义解析——不要求理解「前一阵子」这类口语，只要求发现
「问题说的是上个月、参数却填了 this_month」这种自相矛盾。

三条纪律：
- 校验**只在统计工具被调用时**发生，不改写问题语义；
- 多目标 / 无命中一律**不校验**（放行由口径可见性兜底）；
- 打回消息是给**模型**看的（它据此重试），不是给用户的最终文案。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.core.memory_stats import resolve_period
from app.memory.store import to_local

# 问题里的说法 → period（子串匹配；只收**无第二种解释**的高频词）。
# 刻意不收「上周」：period 枚举里没有"上一个自然周"这一项（this_week 与 last_7_days 都不等于它），
# 强行映射会**算错**——记为已知边界，不校验。
TIME_PHRASES: dict[str, str] = {
    "上个月": "last_month",
    "上月": "last_month",
    "这个月": "this_month",
    "本月": "this_month",
    "本周": "this_week",
    "这周": "this_week",
    "这星期": "this_week",
    "最近七天": "last_7_days",
    "最近7天": "last_7_days",
    "过去一周": "last_7_days",
    "最近一个月": "last_30_days",
    "最近30天": "last_30_days",
    "近30天": "last_30_days",
    "今年": "this_year",
    "本年度": "this_year",
}

PERIOD_CN: dict[str, str] = {
    "this_week": "本周",
    "last_7_days": "最近七天",
    "this_month": "这个月",
    "last_month": "上个月",
    "last_30_days": "最近30天",
    "this_year": "今年",
    "all": "不限",
}

_QUOTE_PAIRS = (("「", "」"), ("“", "”"), ('"', '"'), ("'", "'"))


def _targets(question: str) -> set[str]:
    return {period for phrase, period in TIME_PHRASES.items() if phrase in question}


def range_label(start: datetime, end: datetime) -> str:
    """区间标签（本机日期，含末日；`end` 为**右开**边界）。"""
    last = to_local(end - timedelta(days=1)).date().isoformat()
    return f"{to_local(start).date().isoformat()} ~ {last}"


def expected_window(
    question: str, *, now: datetime | None = None
) -> tuple[str, datetime, datetime] | None:
    """问题里的时间说法 → `(period, start, end)`；命中多个**不同**目标或无命中 → `None`。"""
    targets = _targets(question)
    if len(targets) != 1:
        return None
    period = next(iter(targets))
    start, end = resolve_period(period, None, None, now=now)
    return period, start, end


def period_mismatch(
    question: str,
    period: str | None,
    start: str | None,
    end: str | None,
    *,
    now: datetime | None = None,
) -> str | None:
    """模型给的窗口与问题里的时间说法**区间不等** → 打回消息；相等或不校验 → `None`。"""
    expected = expected_window(question, now=now)
    if expected is None:
        return None
    exp_period, exp_start, exp_end = expected
    exp_cn = PERIOD_CN.get(exp_period, exp_period)
    exp_label = range_label(exp_start, exp_end)
    try:
        got_start, got_end = resolve_period(period or "", start, end, now=now)
    except ValueError:
        return (f"（时间窗参数非法：{period!r}。问题说的是「{exp_cn}」"
                f"＝{exp_label}（{exp_period}），请改正后重试）")
    if (got_start, got_end) == (exp_start, exp_end):
        return None
    return (f"（时间窗与问题不一致：问题说的是「{exp_cn}」＝{exp_label}（{exp_period}）；"
            f"你填的 {period}＝{range_label(got_start, got_end)}。请用 {exp_period} 重试）")


def _quoted_spans(question: str, left: str, right: str) -> list[str]:
    out: list[str] = []
    rest = question
    while True:
        i = rest.find(left)
        if i < 0:
            break
        j = rest.find(right, i + len(left))
        if j < 0:
            break
        out.append(rest[i + len(left): j])
        rest = rest[j + len(right):]
    return out


def quoted_entities(question: str) -> list[str]:
    """引号内的实体候选（去重、去空白、长度 ≥2）。"""
    found: list[str] = []
    for left, right in _QUOTE_PAIRS:
        found.extend(_quoted_spans(question, left, right))
    seen: set[str] = set()
    out: list[str] = []
    for raw in found:
        entity = raw.strip()
        if len(entity) >= 2 and entity not in seen:
            seen.add(entity)
            out.append(entity)
    return out


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _offending_entity(question: str, value: str | None) -> str | None:
    """`value` 像是"把问题里的实体名改写了" → 返回问题里的那个实体名；否则 `None`。

    判定：不含实体 E，但与 E **公共前缀 ≥2 且长度差 ≤2**（近似改写）；
    合法的缩写（长度差大）与无关值一律放行。**非字符串一律放行**（数值筛选很常见）。
    """
    if not value or not isinstance(value, str):
        return None
    for entity in quoted_entities(question):
        if entity in value:
            continue
        if _common_prefix_len(entity, value) >= 2 and abs(len(entity) - len(value)) <= 2:
            return entity
    return None


def entity_mismatch(question: str, value: str | None) -> str | None:
    """打回消息（首次不符时用）；无问题返回 `None`。"""
    entity = _offending_entity(question, value)
    if entity is None:
        return None
    return (f"（筛选值与问题里的实体不一致：问题说的是「{entity}」，你用的是「{value}」。"
            f"请严格原样使用问题里的名称）")


def entity_fix(question: str, value: str | None) -> str | None:
    """第二次仍不符时**该替换成什么**（问题里的实体名）；无需替换返回 `None`。"""
    return _offending_entity(question, value)


def with_stats_caliber(answer: str, calls: list[dict]) -> str:
    """本轮统计调用的口径，凡答案里**没出现**的，追加为独立末行。"""
    missing = [c["label"] for c in calls if c.get("probe") and c["probe"] not in answer]
    if not missing:
        return answer
    return answer + "\n统计口径：" + "；".join(missing)
