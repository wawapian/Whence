"""工具箱 v1（spec §5.2）：六件工具，全部复用既有实现，不重造。

两条纪律：
- **数字只从工具出**（与「LLM 不接触表格做算术」同一原则）：字数/计数/求和由代码精确计算。
- **工具输出出库前过隐私闸**（spec §5.5）：文档 allow_cloud=0、记录侧内容一律 → 本地打码版；
  勾选「可理解」的文档原文可完整给模型。
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import cloud, param_guard, stats
from app.core import memory_stats as memory_stats_core
from app.core.masking import mask_outbound
from app.memory.documents import get_document
from app.models.db import Chunk, DocTable, Document, Image

logger = logging.getLogger(__name__)

READ_DOC_MAX = 20000  # read_document 单次返回上限（超长截断并注明）
LIST_FILES_MAX = 50   # list_files 条数硬上限


@dataclass
class ToolContext:
    """一次问答内工具执行的共享上下文 + 溯源累计（实际被工具读过的对象）。"""

    session: Session
    settings: Settings
    # 参数守卫的基准（spec 2026-10-16-stats-param-guard §3.1）：本轮独立问题原文
    question: str = ""
    # 本轮统计调用的口径留痕（每项 {"label","probe"}）——供终答口径落地
    stats_calls: list[dict] = field(default_factory=list)
    # 每类校验的打回计数（键：time / entity）——第二次走兜底，不再打回
    rejections: dict[str, int] = field(default_factory=dict)
    used_doc_ids: list[str] = field(default_factory=list)
    used_image_ids: list[str] = field(default_factory=list)
    used_table_ids: list[str] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)


def gate(text: str, allowed: bool) -> str:
    """隐私闸（spec §5.5）：可上云的原文原样给模型；否则只给本地打码版。"""
    return text if allowed else mask_outbound(text)


def _full_text(session: Session, doc: Document) -> str:
    """文档全文：优先解析正文，缺失时回落分块拼接。"""
    text = doc.parsed_text or ""
    if not text:
        rows = session.scalars(
            select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.seq)
        ).all()
        text = "".join(c.text for c in rows)
    return text


def _visible_doc(ctx: ToolContext, doc_id: str) -> Document | str:
    """取未删除文档；不存在/已删除返回可读文案（str）。"""
    doc = get_document(ctx.session, doc_id)
    if doc is None or doc.deleted_at is not None:
        return "（文档不存在或已删除）"
    return doc


def _mark_doc(ctx: ToolContext, doc_id: str) -> None:
    if doc_id not in ctx.used_doc_ids:
        ctx.used_doc_ids.append(doc_id)


def read_document(ctx: ToolContext, doc_id: str, keyword: str | None = None) -> str:
    """取全文（可选只取含关键词的行）；出库按 allow_cloud 过闸。"""
    doc = _visible_doc(ctx, doc_id)
    if isinstance(doc, str):
        return doc
    _mark_doc(ctx, doc.id)
    text = _full_text(ctx.session, doc)
    if keyword:
        lines = [line for line in text.splitlines() if keyword in line]
        if not lines:
            return f"（全文里没有含「{keyword}」的段落）"
        text = "\n".join(lines)
    note = ""
    if len(text) > READ_DOC_MAX:
        text = text[:READ_DOC_MAX]
        note = f"\n（全文过长，已截断到 {READ_DOC_MAX} 字）"
    return gate(text + note, bool(doc.allow_cloud))


def measure_text(ctx: ToolContext, doc_id: str, metric: str = "chars") -> str:
    """精确计数（v1 仅字数）：纯代码 len(解析全文)，不经模型。"""
    if metric != "chars":
        return f"（暂不支持度量「{metric}」，v1 只支持 chars = 字数）"
    doc = _visible_doc(ctx, doc_id)
    if isinstance(doc, str):
        return doc
    _mark_doc(ctx, doc.id)
    count = len(_full_text(ctx.session, doc))
    return f"字数：{count} 个字符（按解析全文由代码精确统计）"


def _filter_values(value: str | None, filters: list[dict] | None) -> list[str]:
    """本次调用用到的筛选值（单条件 + filters 多条件）。

    一律转成字符串：模型做数值筛选（「90 分以上」）时传的是**数字**，
    下游守卫按字符串比对，不转型会 TypeError 把整次工具调用打掉。
    """
    out = [str(value)] if value not in (None, "") else []
    for item in filters or []:
        if isinstance(item, dict) and item.get("value") not in (None, ""):
            out.append(str(item["value"]))
    return out


def _filter_parts(column: str | None, op: str | None, value: str | None,
                  filters: list[dict] | None) -> list[str]:
    """筛选条件的可读片段（单条件 + filters 多条件）。"""
    parts: list[str] = []
    if column and op:
        parts.append(f"{column} {op} {value}")
    for item in filters or []:
        if isinstance(item, dict):
            parts.append(f"{item.get('column')} {item.get('op')} {item.get('value')}")
    return parts


def _filter_line(parts: list[str]) -> str:
    """工具输出里的筛选条件行（给模型看；无筛选返回空串）。"""
    return "筛选条件：" + " 且 ".join(parts) if parts else ""


def _filter_caliber(parts: list[str]) -> str:
    """终答口径（给用户看）。

    它同时也是「答案是否已复述口径」的**探针**——探针取**整条口径**而非筛选值：
    筛选值（如「1」）太容易在答案里偶然出现，会漏判，而漏判 = 用户看不到口径。
    """
    return "按「" + " 且 ".join(parts) + "」筛选" if parts else ""


def table_stats(ctx: ToolContext, table_id: str, filters: list[dict] | None = None,
                column: str | None = None, op: str | None = None, value: str | None = None,
                aggregate: str = "none", sum_column: str | None = None,
                group_by: str | None = None, order_by: str | None = None,
                desc: bool = False, limit: int | None = None) -> str:
    """筛选/计数/求和/均值/分组/排序取前 N：通用引擎执行，代码确定性渲染（数字不经模型）。"""
    row = ctx.session.get(DocTable, table_id)
    if row is None:
        return "（表格不存在）"
    doc = get_document(ctx.session, row.document_id)
    if doc is None or doc.deleted_at is not None:
        return "（表格所属文档已删除）"
    try:
        headers = json.loads(row.headers)
        rows = json.loads(row.rows)
    except Exception:
        return "（表格数据损坏，无法统计）"
    # 参数守卫（spec 2026-10-16-stats-param-guard §3.3）：问题里带引号的实体被**改写** → 先打回一次；
    # 第二次仍不符则**强制替换**成问题里的实体名（2026-10-16 修订：软兜底会让错答落地——
    # 实测模型被打回后仍坚持用「样例科技有限公司」，还把问题里的「样例信息」说成"另一家公司"）。
    # 替换是安全的：问题里的名字是用户亲口说的；表里若没有，会如实返回 0 条，不会编造。
    if ctx.question:
        fixes = {
            candidate: fix
            for candidate in _filter_values(value, filters)
            if (fix := param_guard.entity_fix(ctx.question, candidate))
        }
        if fixes:
            if ctx.rejections.get("entity", 0) == 0:
                ctx.rejections["entity"] = 1
                return param_guard.entity_mismatch(ctx.question, next(iter(fixes)))
            if value in fixes:
                value = fixes[value]
            if filters:
                filters = [
                    {**item, "value": fixes.get(item.get("value"), item.get("value"))}
                    if isinstance(item, dict) else item
                    for item in filters
                ]
    query = stats.build_query(
        table_id, headers, filters=filters, column=column, op=op, value=value,
        aggregate=aggregate, sum_column=sum_column, group_by=group_by,
        order_by=order_by, desc=desc, limit=limit,
    )
    if query is None:
        return (
            "（统计参数不合法——表头为：" + " | ".join(headers)
            + "；单条件用 column/op/value，多条件用 filters（[{column,op,value}] 须全部满足），"
            "计数/求和/均值用 aggregate=count/sum/avg，分组用 group_by，排序取前 N 用 order_by/limit）"
        )
    outcome = stats.apply_query(headers, rows, query)
    if row.id not in ctx.used_table_ids:
        ctx.used_table_ids.append(row.id)
    _mark_doc(ctx, row.document_id)
    parts = _filter_parts(column, op, value, filters)
    caliber = _filter_caliber(parts)
    if caliber:
        ctx.stats_calls.append({"label": caliber, "probe": caliber})
    # 筛选条件随工具输出一并给模型（口径可见：模型才能在答案里复述"按什么筛的"）
    line = _filter_line(parts)
    rendered = gate(stats.render_query(outcome, bool(row.suspect)), bool(doc.allow_cloud))
    return rendered + ("\n" + line if line else "")


def memory_stats(ctx: ToolContext, period: str, kind: str = "spending",
                 start: str | None = None, end: str | None = None,
                 keyword: str | None = None, type: str | None = None) -> str:
    """记忆类统计（花费/次数）：收集与渲染全部由代码完成（数字不经模型）。

    出库一律走记录侧闸（`gate(..., False)` → 本地打码）；命中的明细 fact 记入溯源。

    参数守卫（spec 2026-10-16-stats-param-guard §3.2）：问题里出现无歧义的时间说法而
    `period` 的**区间**与之不等 → 先打回一次（附正确值）；第二次仍不符则**强制覆盖**。
    """
    reject = (param_guard.period_mismatch(ctx.question, period, start, end)
              if ctx.question else None)
    force_note = ""
    if reject:
        if ctx.rejections.get("time", 0) == 0:
            ctx.rejections["time"] = 1
            return reject                      # 打回：不统计、不留痕
        forced = param_guard.expected_window(ctx.question)
        if forced is not None:
            period, start, end = forced[0], None, None
            cn = param_guard.PERIOD_CN.get(forced[0], forced[0])
            force_note = f"（问题说的是「{cn}」，已按此统计）"
    try:
        outcome = memory_stats_core.collect(
            ctx.session, period=period, start=start, end=end, kind=kind,
            keyword=keyword, type_filter=type,
        )
    except ValueError:
        return ("（统计参数不合法——period 取值：this_week/last_7_days/this_month/last_month/"
                "last_30_days/this_year/all/custom（custom 需 start/end，YYYY-MM-DD）；"
                "kind 取值：spending/count）")
    # 只并入**渲染明细里出现过**的 fact（上限=STATS_ROWS_MAX）——period=all 大库不爆 sources（终审）
    for row in outcome.rows[:memory_stats_core.STATS_ROWS_MAX]:
        if row.fact_id not in ctx.used_fact_ids:
            ctx.used_fact_ids.append(row.fact_id)
    label = param_guard.range_label(outcome.start, outcome.end)
    ctx.stats_calls.append({"label": f"按记录时间 {label}{force_note}", "probe": label})
    return gate(memory_stats_core.render(outcome), False)


def view_image(ctx: ToolContext, image_id: str, question: str) -> str:
    """看图回答：allow_cloud 且有 VLM → 云端看图；否则只给本地 OCR 文字（过闸）。"""
    img = ctx.session.get(Image, image_id)
    if img is None:
        return "（图片不存在）"
    doc = get_document(ctx.session, img.document_id)
    if doc is None or doc.deleted_at is not None:
        return "（图片所属文档已删除）"
    if img.id not in ctx.used_image_ids:
        ctx.used_image_ids.append(img.id)
    _mark_doc(ctx, doc.id)
    allowed = bool(doc.allow_cloud)
    reason = "附件未授权本次云看图" if not allowed else "未配置看图模型" if not ctx.settings.vlm_model else "图片路径缺失"
    if allowed and ctx.settings.vlm_model and img.file_path:
        try:
            blob = (Path(ctx.settings.data_dir) / img.file_path).read_bytes()
            desc = cloud.describe_image(
                cloud.build_vlm_client(ctx.settings), ctx.settings.vlm_model, blob,
                question or "请描述这张图片的内容",
            )
            if desc:
                return f"图片内容（VLM 识别）：{desc}"
        except Exception:
            reason = "本次看图服务调用失败"
            logger.warning("view_image VLM 失败（回落本地 OCR）", exc_info=True)
    if img.caption:
        return gate(f"{reason}。以下是此前已保存的图片描述，不是本次重新看图：\n{img.caption}", allowed)
    if img.ocr_text:
        return gate(
            f"{reason}，以下是本地 OCR 文字：\n{img.ocr_text[:2000]}",
            allowed,
        )
    return f"该图片没有可用的内容（{reason}，且无已存描述或本地 OCR 文字）。"


def list_files(ctx: ToolContext, keyword: str | None = None, limit: int = 20) -> str:
    """按文件名关键词列出未删除文件（文件标题按各自 allow_cloud 过闸）。"""
    from app.memory.documents import find_files
    try:
        n = max(1, min(int(limit or 20), LIST_FILES_MAX))
    except (TypeError, ValueError):
        n = 20
    docs = find_files(ctx.session, keyword, limit=n)
    if not docs:
        return "（没有匹配的文件）"
    lines = []
    for i, d in enumerate(docs, start=1):
        title = gate(d.title or "（无标题）", bool(d.allow_cloud))
        date = d.created_at.date().isoformat() if d.created_at else ""
        lines.append(f"{i}. {title}（{d.kind}，{date}）id={d.id}")
    return "\n".join(lines)


_HANDLERS = {
    "read_document": read_document,
    "measure_text": measure_text,
    "table_stats": table_stats,
    "memory_stats": memory_stats,
    "view_image": view_image,
    "list_files": list_files,
}


def run_tool(ctx: ToolContext, name: str, args: dict) -> str:
    """执行一次工具调用；任何异常不外抛——返回可读消息让模型继续（不打断主循环）。"""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"（未知工具：{name}）"
    try:
        result = handler(ctx, **args)
        ctx.tool_trace.append({"tool": name, "arguments": args, "status": "returned",
                               "result_chars": len(result), "result_note": result[:250]})
        return result
    except TypeError as e:
        ctx.tool_trace.append({"tool": name, "arguments": args, "status": "invalid_arguments"})
        return f"（工具 {name} 参数不合法：{e}）"
    except Exception:
        ctx.tool_trace.append({"tool": name, "arguments": args, "status": "failed"})
        logger.warning("工具 %s 执行失败", name, exc_info=True)
        return f"（工具 {name} 执行失败）"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "读取一份文档的全文，可选只取包含关键词的行。doc_id 见材料清单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档 id"},
                    "keyword": {"type": "string", "description": "可选：只返回包含该关键词的行"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_text",
            "description": "对文档做精确计数（v1 仅字数）。字数必须用本工具，不要自己数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "metric": {"type": "string", "enum": ["chars"], "description": "chars = 字数（字符数）"},
                },
                "required": ["doc_id", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "table_stats",
            "description": (
                "对一张已解析表格做筛选/计数/求和/均值/分组/排序取前 N（表格 id 与表头见「可用表格」清单）。"
                "数字由代码精确计算。多个条件同时满足用 filters 传多个；分组统计用 group_by；"
                "「最多的前 3 个」这类用 order_by + limit。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "filters": {
                        "type": "array",
                        "description": "多条件（AND，须全部满足）；与 column/op/value 二选一",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "op": {"type": "string",
                                       "enum": ["<", "<=", ">", ">=", "==", "contains"]},
                                "value": {"type": "string"},
                            },
                            "required": ["column", "op", "value"],
                        },
                    },
                    "column": {"type": "string", "description": "单条件筛选列名（表头之一）；不筛选则不传"},
                    "op": {
                        "type": "string",
                        "enum": ["<", "<=", ">", ">=", "==", "contains"],
                        "description": "比较符；数值比较（< <= > >=）时 value 必须为数字或 ISO 日期（YYYY-MM-DD）",
                    },
                    "value": {"type": "string", "description": "比较值"},
                    "aggregate": {
                        "type": "string",
                        "enum": ["none", "count", "sum", "avg"],
                        "description": "none=列出结果行；count=统计条数；sum=求和；avg=求均值",
                    },
                    "sum_column": {"type": "string",
                                   "description": "aggregate=sum/avg 的数值列（多条件或无筛选时必填）"},
                    "group_by": {"type": "string", "description": "分组列（分组输出各组 count/sum/avg）"},
                    "order_by": {"type": "string", "description": "排序取前 N 的排序列"},
                    "desc": {"type": "boolean", "description": "order_by 是否降序（默认升序）"},
                    "limit": {"type": "integer", "description": "取前 N 条/组（≤50）"},
                },
                "required": ["table_id", "aggregate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_stats",
            "description": (
                "统计用户记忆中的花费/次数（如「这个月花了多少钱」「上周记了几笔消费」）。"
                "金额合计与条数由代码精确计算，必须用本工具、不要自己数。"
                "问「几笔消费/花销/支出」→ kind=count 且 type=consumption"
                "（不给 type 会把所有事件类记录都算上）；问「花了多少钱」→ kind=spending。"
                "时间说法映射：这个月=this_month、上个月=last_month、本周=this_week、"
                "最近七天=last_7_days、最近30天=last_30_days、今年=this_year、不限=all。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["this_week", "last_7_days", "this_month", "last_month",
                                 "last_30_days", "this_year", "all", "custom"],
                        "description": "时间窗；custom 时必须给 start/end（YYYY-MM-DD）",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["spending", "count"],
                        "description": "spending=金额合计（默认）；count=记录条数",
                    },
                    "start": {"type": "string", "description": "period=custom 的起始日（含）"},
                    "end": {"type": "string", "description": "period=custom 的结束日（含）"},
                    "keyword": {"type": "string", "description": "可选：只统计包含该词的记录"},
                    "type": {"type": "string", "description": "可选：限定记忆类型（如 consumption）"},
                },
                "required": ["period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "查看一张图片并回答问题（图片 id 见「可用图片」清单）。未勾选可理解的图片只会返回本地 OCR 文字。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "question": {"type": "string", "description": "针对这张图的问题"},
                },
                "required": ["image_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "按文件名关键词列出用户上传的文件（文档/图片）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "文件名包含的关键词；不传 = 全部"},
                    "limit": {"type": "integer", "description": "最多返回条数（默认 20）"},
                },
                "required": [],
            },
        },
    },
]
