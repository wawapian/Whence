"""表格条件统计：LLM 只抽条件（JSON）→ 代码执行筛选/聚合 → 代码确定性渲染（sum 除外）。

与 tables.py 解耦：调用方负责把表头/行数据摊平传入，本模块不感知表格来源。
"""
import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

OPS = ("<", "<=", ">", ">=", "==", "contains")
NUMERIC_OPS = ("<", "<=", ">", ">=")
AGGREGATES = ("none", "count", "sum")  # legacy 条件抽取用（不含 avg）
QUERY_AGGREGATES = ("none", "count", "sum", "avg")  # 通用引擎（工具）用
MAX_SAMPLE_ROWS = 3
MAX_RESULT_ROWS = 30  # 组答提示词里最多列出多少行结果
GROUPS_MAX = 30       # 分组渲染上限（超出给「…共 N 组」）

_EXTRACT_PROMPT = (
    "你是表格条件抽取器：根据用户问题，从候选表格中选出一张表，抽取出一个筛选条件与聚合方式。\n"
    "只输出一个 JSON 对象，不要输出解释、标注或多余文字。\n\n"
    "用户问题：{question}\n\n"
    "候选表格：\n{tables}\n\n"
    "输出 JSON 的六个字段：\n"
    '- "table_id"：所选表格的 id（必须是上面列出的 id 之一）\n'
    '- "column"：筛选条件所在的列名（必须是该表表头之一，允许与表头有空格差异）；无筛选条件时为 null\n'
    '- "op"：比较操作符，取值 {{"<", "<=", ">", ">=", "==", "contains"}}；无筛选条件时为 null（此时 aggregate 为 count 或 sum）\n'
    '- "value"：比较值（字符串；contains 为子串匹配，== 为整体相等）；无筛选条件时为 null（此时 aggregate 为 count 或 sum）\n'
    '- "aggregate"：聚合方式，取值 {{"none", "count", "sum"}}'
    "（none=列出结果行；count=统计结果行数；sum=对 sum_column 列数值求和）\n"
    '- "sum_column"：aggregate 为 sum 时，要合计的数值列（有筛选条件时可与 column 不同；无筛选条件时必填）\n'
    '无筛选条件且要统计数量时输出："column": null, "op": null, "value": null, "aggregate": "count"\n'
)

_COMPOSE_SYSTEM = (
    "根据给定的筛选结果回答用户问题。只依据给定的筛选结果作答，不得新增、合并或改写任何条目，不要编造。"
    "若给出了聚合结果，答案必须原样采用该数值。"
    "若聚合结果标注为无法计算，如实说明无法计算，不要编造数值。"
    "若注明了无法解析为数值的单元格，请在答案中提及。"
)


@dataclass
class TableCandidate:
    id: str
    headers: list[str]
    sample_rows: list[list[str]]


@dataclass
class Filter:
    table_id: str
    column: str
    op: str
    value: str
    aggregate: str = "none"
    sum_column: str | None = None  # aggregate 为 sum 时的求和列（None = 用 column）


@dataclass
class FilterOutcome:
    rows: list[list[str]]
    aggregate_value: float | int | None
    skipped: int = 0
    aggregate: str = "none"  # apply_filter 原样带回，供渲染分支判定
    sum_column: str | None = None
    filter_skipped: int = 0  # 数值 op 下筛选列单元格不可解析的行数（字符串 op 恒 0）


def extract_filter(client, model: str, question: str, tables: list[TableCandidate]) -> Filter | None:
    """LLM 抽取筛选条件；调用失败 / 输出非法（越界或非 JSON）→ None（调用方降级普通问答）。"""
    prompt = _EXTRACT_PROMPT.format(question=question, tables=_format_tables(tables))
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content or ""
    except Exception:
        logger.warning("条件抽取 LLM 调用失败（降级普通问答）", exc_info=True)
        return None
    data = _parse_json(content)
    if data is None:
        logger.warning("条件抽取输出无法解析为 JSON（降级普通问答）：%r", content[:200])
        return None
    return _validate_filter(data, tables)


def build_filter(table_id: str, headers: list[str], *, column: str | None, op: str | None,
                 value: str | None, aggregate: str = "none",
                 sum_column: str | None = None) -> Filter | None:
    """工具参数 → 可执行 Filter：校验规则与 extract_filter 完全同一套（复用 _validate_filter）。

    供 table_stats 工具使用（模型直接给参数）；任何非法（列不在表头、op/aggregate 越界、
    数值 op 比较值不可解析等）→ None，由调用方向模型说明参数不合法。
    """
    data = {
        "table_id": table_id, "column": column, "op": op, "value": value,
        "aggregate": aggregate, "sum_column": sum_column,
    }
    return _validate_filter(data, [TableCandidate(id=table_id, headers=headers, sample_rows=[])])


def _format_tables(tables: list[TableCandidate]) -> str:
    lines: list[str] = []
    for t in tables:
        lines.append(f"- id: {t.id}")
        lines.append(f"  表头: {' | '.join(t.headers)}")
        for row in t.sample_rows[:MAX_SAMPLE_ROWS]:
            lines.append(f"  行: {' | '.join(row)}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict | None:
    """整体 json.loads；失败则取首个 {...} 再试（LLM 常夹带说明文字）。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.S)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _stored_header(name, headers: list[str]) -> str | None:
    """在表头中找与 name 空白无关相等的原表头字符串（docling 会输出「课 程 名 称」）。

    返回存储的原表头（保证 apply_filter 用 headers.index 能定位）；找不到返回 None。
    """
    if not isinstance(name, str) or not name.strip():
        return None
    key = re.sub(r"\s+", "", name)
    for header in headers:
        if re.sub(r"\s+", "", header) == key:
            return header
    return None


def _validate_filter(data: dict, tables: list[TableCandidate]) -> Filter | None:
    table = next((t for t in tables if t.id == data.get("table_id")), None)
    if table is None:
        logger.warning("条件抽取 table_id 不在候选表格内：%r", data.get("table_id"))
        return None
    aggregate = data.get("aggregate", "none")
    if aggregate not in AGGREGATES:
        logger.warning("条件抽取 aggregate 越界：%r", aggregate)
        return None
    column_raw = data.get("column")
    # 无筛选条件（column/op/value 全空）：count=统计全部行数；sum=对 sum_column 全列求和
    if column_raw is None or column_raw == "":
        if data.get("op") not in (None, "") or data.get("value") not in (None, ""):
            logger.warning("条件抽取缺 column 但给了 op/value：%r", data)
            return None
        if aggregate == "count":
            return Filter(table_id=table.id, column="", op="", value="", aggregate="count")
        if aggregate == "sum":
            sum_column_raw = data.get("sum_column")
            if sum_column_raw in (None, ""):
                logger.warning("无条件下 sum 缺 sum_column：%r", data)
                return None
            sum_column = _stored_header(sum_column_raw, table.headers)
            if sum_column is None:
                logger.warning("条件抽取 sum_column 不在表头内：%r", sum_column_raw)
                return None
            return Filter(
                table_id=table.id, column="", op="", value="", aggregate="sum", sum_column=sum_column
            )
        logger.warning("条件抽取缺 column 且非无条件下 count/sum：%r", data)
        return None
    column = _stored_header(column_raw, table.headers)
    if column is None:
        logger.warning("条件抽取 column 不在表头内：%r", column_raw)
        return None
    op = data.get("op")
    if op not in OPS:
        logger.warning("条件抽取 op 越界：%r", op)
        return None
    value = data.get("value")
    if value is None:
        logger.warning("条件抽取缺 value")
        return None
    if op in ("<", "<=", ">", ">=") and _to_float(value) is None:
        # 数值 op 的比较值不可解析（如「小于1分」）→ 条件根本执行不了：拒绝走降级（如实说明）。
        # 不得放行——放行会让所有行都不匹配，把「条件没执行」答成「没有符合条件的记录。」
        logger.warning("条件抽取数值 op 的比较值不可解析为数值：%r", value)
        return None
    sum_column = None
    sum_column_raw = data.get("sum_column")
    if sum_column_raw not in (None, ""):
        sum_column = _stored_header(sum_column_raw, table.headers)
        if sum_column is None:
            logger.warning("条件抽取 sum_column 不在表头内：%r", sum_column_raw)
            return None
    return Filter(
        table_id=table.id,
        column=column,
        op=op,
        value=str(value),
        aggregate=aggregate,
        sum_column=sum_column,
    )


def apply_filter(headers: list[str], rows: list[list[str]], filt: Filter) -> FilterOutcome:
    """程序化执行：先筛出结果行（column 为空 = 不过滤，统计全部行），聚合在结果集上执行。"""
    filter_skipped = 0
    if filt.column == "":
        matched = list(rows)
    elif filt.column not in headers:
        return FilterOutcome(rows=[], aggregate_value=None, aggregate=filt.aggregate)
    else:
        idx = headers.index(filt.column)
        value = str(filt.value).strip()
        numeric_op = filt.op in ("<", "<=", ">", ">=")
        matched = []
        for row in rows:
            if _match(row[idx], filt.op, value):
                matched.append(row)
            elif numeric_op and _to_float(row[idx]) is None:
                filter_skipped += 1  # 数值 op 下该行筛选列不可解析：不匹配，但如实计数

    if filt.aggregate == "count":
        return FilterOutcome(
            rows=matched, aggregate_value=len(matched), aggregate="count",
            filter_skipped=filter_skipped,
        )
    if filt.aggregate == "sum":
        sum_column = filt.sum_column or filt.column
        sum_idx = headers.index(sum_column) if sum_column in headers else None
        total = 0.0
        parsed = 0
        skipped = 0
        for row in matched:
            num = _to_float(row[sum_idx]) if sum_idx is not None else None
            if num is None:
                skipped += 1
            else:
                total += num
                parsed += 1
        return FilterOutcome(
            rows=matched,
            aggregate_value=total if parsed else None,  # 无可解析数值 → 不编造 0
            skipped=skipped,
            aggregate="sum",
            sum_column=sum_column,
            filter_skipped=filter_skipped,
        )
    return FilterOutcome(
        rows=matched, aggregate_value=None, aggregate="none", filter_skipped=filter_skipped
    )


@dataclass
class Cond:
    column: str
    op: str
    value: str


@dataclass
class Query:
    """通用统计查询（工具用；legacy 的 Filter 不动）：多条件 AND / 分组 / sum/avg / 排序取前 N。"""

    table_id: str
    conds: list[Cond] = field(default_factory=list)
    aggregate: str = "none"
    value_column: str | None = None  # sum/avg 的数值列
    group_by: str | None = None
    order_by: str | None = None
    desc: bool = False
    limit: int | None = None  # ≤ MAX_QUERY_LIMIT；仅作用于展示行/分组


@dataclass
class QueryOutcome:
    rows: list[list[str]] = field(default_factory=list)      # 展示行（none：排序+limit 后；其他：匹配行）
    rows_total: int = 0                                      # 匹配总行数（渲染「N 条/N 行/…共」用）
    groups: list[tuple[str, float | None, int]] = field(default_factory=list)
    groups_total: int = 0
    aggregate_value: float | int | None = None
    skipped: int = 0
    filter_skipped: int = 0
    aggregate: str = "none"
    group_by: str | None = None
    value_column: str | None = None
    rows_truncated: bool = False
    order_skipped: int = 0  # 排序列不可解析、被恒置末尾的行数（如实披露用）


MAX_QUERY_LIMIT = 50


def build_query(table_id: str, headers: list[str], *, filters: list[dict] | None = None,
                column: str | None = None, op: str | None = None, value: str | None = None,
                aggregate: str = "none", sum_column: str | None = None,
                group_by: str | None = None, order_by: str | None = None,
                desc: bool = False, limit: int | None = None) -> Query | None:
    """工具参数 → 可执行 Query；任何非法（列不在表头、op 越界、filters 与单条件同给等）→ None。

    兼容旧单条件参数（column/op/value）；filters（多条件 AND）与单条件**二选一**。
    `group_by` 未给聚合时按 `count`；sum/avg 无 sum_column 时默认取唯一筛选列。
    """
    if aggregate not in QUERY_AGGREGATES:
        logger.warning("统计查询 aggregate 越界：%r", aggregate)
        return None
    conds: list[Cond] = []
    if filters:
        if any(x not in (None, "") for x in (column, op, value)):
            logger.warning("filters 与单条件（column/op/value）同时给出（二选一）")
            return None
        for item in filters:
            if not isinstance(item, dict):
                logger.warning("filters 元素不是对象：%r", item)
                return None
            cond = _build_cond(item.get("column"), item.get("op"), item.get("value"), headers)
            if cond is None:
                return None
            conds.append(cond)
    elif column not in (None, ""):
        cond = _build_cond(column, op, value, headers)
        if cond is None:
            return None
        conds.append(cond)
    elif op not in (None, "") or value not in (None, ""):
        logger.warning("缺 column 但给了 op/value：%r", (op, value))
        return None

    group = None
    if group_by not in (None, ""):
        group = _stored_header(group_by, headers)
        if group is None:
            logger.warning("group_by 不在表头内：%r", group_by)
            return None
        if aggregate == "none":
            aggregate = "count"  # 分组未给聚合 → 按条数（宽松口径）
    order = None
    if order_by not in (None, ""):
        order = _stored_header(order_by, headers)
        if order is None:
            logger.warning("order_by 不在表头内：%r", order_by)
            return None
    value_col = None
    if aggregate in ("sum", "avg"):
        raw_col = sum_column if sum_column not in (None, "") else (
            conds[0].column if len(conds) == 1 else None)
        if raw_col is None:
            logger.warning("sum/avg 需要 sum_column（多条件或无筛选时必填）")
            return None
        value_col = _stored_header(raw_col, headers)
        if value_col is None:
            logger.warning("sum_column 不在表头内：%r", raw_col)
            return None
    limit_value = None
    if limit is not None:
        try:
            limit_value = max(1, min(int(limit), MAX_QUERY_LIMIT))
        except (TypeError, ValueError):
            logger.warning("limit 非法：%r", limit)
            return None
    return Query(table_id=table_id, conds=conds, aggregate=aggregate, value_column=value_col,
                 group_by=group, order_by=order, desc=bool(desc), limit=limit_value)


def _build_cond(column, op, value, headers: list[str]) -> Cond | None:
    col = _stored_header(column, headers)
    if col is None:
        logger.warning("筛选列不在表头内：%r", column)
        return None
    if op not in OPS:
        logger.warning("筛选 op 越界：%r", op)
        return None
    if value is None:
        logger.warning("筛选缺 value")
        return None
    if op in NUMERIC_OPS and _to_float(value) is None and not _is_iso_date(value):
        logger.warning("数值 op 的比较值不可解析为数值或 ISO 日期：%r", value)
        return None
    return Cond(column=col, op=op, value=str(value))


def apply_query(headers: list[str], rows: list[list[str]], query: Query) -> QueryOutcome:
    """执行：AND 过滤 → 聚合（可分组）→ 排序 → limit（展示行/分组）。"""
    index = {h: i for i, h in enumerate(headers)}
    matched: list[list[str]] = []
    filter_skipped = 0
    for row in rows:
        ok = True
        unparsable = False
        for cond in query.conds:
            idx = index.get(cond.column)
            if idx is None or not _match(row[idx], cond.op, cond.value):
                ok = False
                if (idx is not None and cond.op in NUMERIC_OPS
                        and _to_float(row[idx]) is None and not _is_iso_date(row[idx])):
                    unparsable = True
        if ok:
            matched.append(row)
        elif unparsable:
            filter_skipped += 1

    def _aggregate(subset: list[list[str]]) -> tuple[float | int | None, int]:
        """(聚合值, 跳过的单元格数)；count → 行数。"""
        if query.aggregate == "count":
            return len(subset), 0
        if query.aggregate in ("sum", "avg"):
            idx = index.get(query.value_column) if query.value_column else None
            total, parsed, skipped = 0.0, 0, 0
            for row in subset:
                num = _to_float(row[idx]) if idx is not None else None
                if num is None:
                    skipped += 1
                else:
                    total += num
                    parsed += 1
            if parsed == 0:
                return None, skipped  # 无可解析数值 → 不编造 0
            return (total if query.aggregate == "sum" else total / parsed), skipped
        return None, 0

    if query.group_by:
        gidx = index[query.group_by]
        buckets: dict[str, list[list[str]]] = {}
        for row in matched:
            buckets.setdefault(str(row[gidx]).strip(), []).append(row)
        skipped_total = 0
        groups: list[tuple[str, float | None, int]] = []
        for key in sorted(buckets):
            value, skipped = _aggregate(buckets[key])
            skipped_total += skipped
            groups.append((key, value, len(buckets[key])))
        if query.order_by:  # 分组时按聚合值排序（order_by 列名只作"要排序"信号）
            # 聚合值不可计算的组**恒置末尾**（不随 desc 翻转——否则"无法计算"组会冒充最高组进前 N；终审 C1）
            valued = [g for g in groups if g[1] is not None]
            unvalued = [g for g in groups if g[1] is None]
            valued.sort(key=lambda g: g[1], reverse=query.desc)
            groups = valued + unvalued
        groups_total = len(groups)
        truncated = False
        if query.limit is not None:
            truncated = groups_total > query.limit
            groups = groups[:query.limit]
        return QueryOutcome(rows=matched, rows_total=len(matched), groups=groups,
                            groups_total=groups_total, aggregate_value=None,
                            skipped=skipped_total, filter_skipped=filter_skipped,
                            aggregate=query.aggregate, group_by=query.group_by,
                            value_column=query.value_column, rows_truncated=truncated)

    display = matched
    order_skipped = 0
    if query.order_by:
        oidx = index[query.order_by]
        numeric = [r for r in display if _to_float(r[oidx]) is not None]
        if numeric and len(numeric) < len(display):
            # 混合列：能解析的按数值排，不能解析的**恒置末尾**（与 desc 无关）——终审 C1：
            # 否则 desc 下字符串序会把「面议」排到最前、把 9000 排在 12000 前，top-N 成员与次序同时错
            texty = [r for r in display if _to_float(r[oidx]) is None]
            order_skipped = len(texty)
            display = sorted(numeric, key=lambda r: _to_float(r[oidx]), reverse=query.desc) + texty
        elif numeric:
            display = sorted(display, key=lambda r: _to_float(r[oidx]), reverse=query.desc)
        else:  # 全文本列：字符串序（本就非数值，不算"跳过"）
            display = sorted(display, key=lambda r: str(r[oidx]), reverse=query.desc)
    truncated = False
    if query.aggregate == "none" and query.limit is not None:
        truncated = len(display) > query.limit
        display = display[:query.limit]
    value, skipped = _aggregate(matched)
    return QueryOutcome(rows=display, rows_total=len(matched), aggregate_value=value,
                        skipped=skipped, filter_skipped=filter_skipped,
                        aggregate=query.aggregate, value_column=query.value_column,
                        rows_truncated=truncated, order_skipped=order_skipped)


def render_query(outcome: QueryOutcome, suspect: bool) -> str:
    """确定性渲染（单条件文案与旧 `toolbox._render_outcome` 逐字一致，保证工具测试零改动）。"""
    disclosure = ""
    if outcome.filter_skipped > 0:
        disclosure = f"另有 {outcome.filter_skipped} 行该列无法解析为数值，未纳入筛选。"
    order_disclosure = (f"另有 {outcome.order_skipped} 行该列无法解析为数值，未参与排序。"
                        if outcome.order_skipped else "")

    if outcome.group_by:
        if not outcome.groups:
            text = "没有符合条件的记录。" + disclosure
        else:
            lines = [f"分组统计（按「{outcome.group_by}」，{outcome.groups_total} 组）："]
            for name, value, count in outcome.groups[:GROUPS_MAX]:
                if outcome.aggregate == "count":
                    lines.append(f"- {name}：{int(value)} 条")
                else:
                    label = "合计" if outcome.aggregate == "sum" else "均值"
                    rendered = "无法计算" if value is None else f"{value:g}"
                    lines.append(f"- {name}：{label} {rendered}（{count} 行）")
            if outcome.groups_total > GROUPS_MAX:
                lines.append(f"…共 {outcome.groups_total} 组")
            text = "\n".join(lines)
            if outcome.skipped:
                text += f"\n另有 {outcome.skipped} 个单元格无法解析为数值，未计入。"
            if disclosure:
                text += "\n" + disclosure
    elif outcome.rows_total == 0:
        text = "没有符合条件的记录。" + disclosure
    elif outcome.aggregate == "count":
        text = f"符合条件的有 {outcome.aggregate_value} 条。" + disclosure
    elif outcome.aggregate in ("sum", "avg"):
        label = "求和结果" if outcome.aggregate == "sum" else "均值结果"
        if outcome.aggregate_value is None:
            text = (f"{label}：无法计算（列「{outcome.value_column}」无可解析为数值的单元格）。"
                    + disclosure)
        else:
            text = (f"{label}：{outcome.aggregate_value:g}"
                    f"（列「{outcome.value_column}」，{outcome.rows_total} 行）")
            if outcome.skipped > 0:
                text += f"；另有 {outcome.skipped} 个单元格无法解析为数值，未计入。"
            text += disclosure
    else:  # none：列出结果行
        shown = outcome.rows[:MAX_RESULT_ROWS]
        lines = [
            f"{i}. " + " | ".join(str(c) for c in row if str(c).strip())
            for i, row in enumerate(shown, start=1)
        ]
        text = f"筛选结果（{outcome.rows_total} 条）：\n" + "\n".join(lines)
        if outcome.rows_total > len(shown):
            text += f"\n…共 {outcome.rows_total} 行"
        if disclosure:
            text += "\n" + disclosure
        if order_disclosure:
            text += "\n" + order_disclosure
    if suspect:
        text += "（该表格解析可能不完整）"
    return text


def _match(cell: str, op: str, value: str) -> bool:
    """数值 op：双边数字按数值比较；双边都是 ISO 日期（YYYY-MM-DD）按日期序比较
    （ISO 字典序即日期序——时间列筛选凭此解锁）；==/contains 按 strip 后字符串。"""
    text = str(cell).strip()
    if op == "==":
        return text == value
    if op == "contains":
        return value in text
    if op not in ("<", "<=", ">", ">="):
        raise ValueError(f"未知操作符: {op}")
    cell_num = _to_float(text)
    value_num = _to_float(value)
    if cell_num is not None and value_num is not None:
        left, right = cell_num, value_num
    elif _is_iso_date(text) and _is_iso_date(value):
        left, right = text, value
    else:
        return False
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left >= right


def _to_float(text) -> float | None:
    try:
        return float(str(text).strip())
    except ValueError:
        return None


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_iso_date(text) -> bool:
    return bool(_ISO_DATE_RE.match(str(text).strip()))


def compose_answer(client, model: str, question: str, headers: list[str], outcome: FilterOutcome) -> str:
    """组织答案：空结果/count/none 由代码确定性渲染（不经 LLM）；sum 走 LLM 组答。"""
    disclosure = ""
    if outcome.filter_skipped > 0:
        disclosure = f"另有 {outcome.filter_skipped} 行该列无法解析为数值，未纳入筛选。"
    if not outcome.rows:
        return "没有符合条件的记录。" + disclosure
    if outcome.aggregate == "count":
        return f"符合条件的有 {len(outcome.rows)} 条。" + disclosure
    if outcome.aggregate == "none":
        lines = []
        for i, row in enumerate(outcome.rows, start=1):
            cells = [str(c) for c in row if str(c).strip()]
            lines.append(f"{i}. " + " | ".join(cells))
        if disclosure:
            lines.append(disclosure)
        return f"筛选结果（{len(outcome.rows)} 条）：\n" + "\n".join(lines)

    # sum：LLM 仅组织语言，提示词给全量约束 + 求和列 + 聚合值 + 跳过计数
    parts = [f"问题：{question}", f"表头：{' | '.join(headers)}"]
    shown = outcome.rows[:MAX_RESULT_ROWS]
    lines = [" | ".join(str(c) for c in row) for row in shown]
    if len(outcome.rows) > MAX_RESULT_ROWS:
        lines.append(f"…共 {len(outcome.rows)} 行")
    parts.append("筛选结果：\n" + "\n".join(lines))
    parts.append(f"求和列：{outcome.sum_column}")
    if outcome.aggregate_value is None:
        parts.append("聚合结果：无法计算（求和列无可解析为数值的单元格）")
    else:
        parts.append(f"聚合结果：{outcome.aggregate_value:g}")
    if outcome.skipped > 0:
        parts.append(f"另有 {outcome.skipped} 个单元格无法解析为数值，未计入求和")
    if disclosure:
        parts.append(disclosure)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _COMPOSE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    return resp.choices[0].message.content or ""
