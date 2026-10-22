"""问答层主循环（spec §5）：模型主导，五工具（toolbox）；旧 4 管线经 qa_engine=legacy 回退。

流程：材料层先选 top-K 材料 → 只在命中材料内检索细节（spec §4 两级检索）→
「材料清单 + 明细 + 表格/图片清单 + 工具定义」交给模型，模型自己决定
「直接答 / 调工具（≤ qa_max_rounds 轮）」→ 终答 + 来源（复用 sources/documents 结构）。

隐私闸（spec §5.5）：一切出库文本过 toolbox.gate——文档按 allow_cloud、记录侧内容一律本地打码。
兜底（spec §5.3）：无材料 → 固定文案；模型答不出 → 可读文案；任何情况下不编造。
"""
import json
import logging
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import llm, param_guard, retrieval, rewrite, toolbox
from app.memory.documents import (
    get_document,
    image_source,
    visible_doc_tables,
    visible_images,
)
from app.memory.store import to_local, utcnow
from app.models.db import Chunk, Document, Fact, Material, RawEvidence
from app.models.schemas import DocumentSource, QueryResponse, Source

logger = logging.getLogger(__name__)

INSUFFICIENT_MARKER = "[信息不足]"
INSUFFICIENT_ANSWER = "根据已有内容无法回答该问题。"
NO_RESULT_ANSWER = "没有找到相关记录"

_BLANK_RUN_RE = re.compile(r"\n{3,}")


def finalize_answer(text: str | None) -> str:
    """终答收尾（qa 与 answer 两条路径共用）：剥掉 [信息不足] 标记；无实质内容 → 兜底文案。

    模型按提示词答不出时会输出标记，但实际常写成「解释 + 标记」——旧逻辑一经命中就整段
    替换成兜底文案，把解释一并丢掉（会话 26 评测排查：8 道失败题里 5 道源于此）。现在只有
    「剥掉标记后没有实质内容」的答案才换兜底文案；带解释的详答保留，标记本身不外泄。
    """
    if not text:
        return INSUFFICIENT_ANSWER
    remainder = _BLANK_RUN_RE.sub("\n\n", text.replace(INSUFFICIENT_MARKER, "")).strip()
    if any(ch.isalnum() for ch in remainder):  # 只剩标点/空白（如整段就是标记）→ 无实质内容
        return remainder
    return INSUFFICIENT_ANSWER

_MATERIALS_TOP_K = 8   # 两级检索第一级：先选多少份材料（spec §4）
_DETAILS_TOP_K = 10    # 第二级：命中材料内检索多少条细节
_TOOL_DOC_CHUNKS = 3   # 工具读过的文档，溯源补前 N 个片段
_IMAGES_LIST_MAX = 15  # 上下文里最多列几张图（图多的论文常见十几张图，10 会漏掉"图 8"这类点名）

_SYSTEM_PROMPT = (
    "你是「Whence · 来处」的问答助手。\n\n"
    "回答规则：\n"
    "- 只依据给定材料与工具返回的内容回答，不得编造。\n"
    "- 涉及精确数字（字数、计数、求和）必须调用工具计算，不要心算；工具返回的数字可原样使用。\n"
    "- 中文数量表述：「N 分以上/N 以上」含 N 本身（筛选用 >=），「超过 N」不含（用 >）；"
    "不要自行缩窄用户给的范围。\n"
    "- 材料不够时先考虑调用工具（读全文 / 查看表格 / 看图 / 找文件）；仍不足以回答时："
    "先说明材料里有什么、缺什么（如「只找到一条记录，没写……」，给出线索，不要凭空推断），"
    f"最后另起一行输出：{INSUFFICIENT_MARKER}\n"
    "- 问题问到图片内容（\"图 N\"、照片、截图里是什么/长什么样）→ 先在「可用图片」清单里按页/标题"
    "找到对应图，调 view_image 看过后再回答；该图因未勾选「可理解」无法上云时，如实告诉用户这个原因。\n"
    "- 问题要在表格里查具体记录（\"XX 公司有哪些岗位\"这类列举）→ 用 table_stats 筛选（op=contains），"
    "以工具返回逐条作答，不要只凭检索片段下结论；**材料常含多张同构表，一张表里没有就先换另一张再试**"
    "（「可用表格」清单里逐张排查，别查一张就下结论）。\n"
    "- 统计类问题：花费/金额（\"这个月花了多少钱\"）→ 调 memory_stats（kind=spending）；"
    "「几笔消费/花销」→ memory_stats（kind=count 且 type=consumption，否则会把普通事件也算上）；"
    "「几笔**有金额**的消费」→ 同样用 kind=spending（返回里的「N 笔」或跨币种的「共 N 笔」即答案，"
    "别用 count 自己数、更不要把各币种的笔数相加）"
    "（时间说法映射：问题怎么说就怎么对应——这个月=this_month、上个月=last_month、"
    "本周=this_week、最近七天=last_7_days、今年=this_year；填错会被工具按问题原文打回重试）；"
    "表格里的筛选/计数/求和数字必须来自 table_stats 结果，不要自己算。\n"
    "- 工具参数里的实体名（公司名、文件名、人名、项目名等）必须**原样使用问题里的名称**，"
    "不得替换、缩写或改写——写错会被工具打回重试。\n"
    "- 指代模糊（\"我最喜欢的/上次说的那个\"）且材料里有多个候选 → 列出候选并用工具逐一核对"
    "（如各份字数），请用户确认；不要硬选一份作答、也不要笼统说无法回答。\n"
    "- 材料与工具结果中的 [[SECRET]] 是已本地打码的敏感值，原样保留在答案里，不要猜测或还原其内容。"
)


def run(session: Session, question: str, settings: Settings) -> QueryResponse:
    """工具循环问答（qa_engine=tool_loop 路径）。"""
    client = llm.build_client(settings)
    queries = rewrite.retrieval_queries(question, settings, client)

    # 第一级：材料层选 top-K 材料
    material_ids = retrieval.search_materials_multi(
        session, queries, top_k=_MATERIALS_TOP_K, model=settings.embedding_model
    )
    if not material_ids:
        return QueryResponse(answer=NO_RESULT_ANSWER, sources=[], documents=[])
    mats = _ordered(session, Material, material_ids)
    record_raw_ids = [m.source_id for m in mats if m.source_type == "record"]
    doc_ids = [m.source_id for m in mats if m.source_type == "document"]
    # 组关系传递：命中「记录」时，其同组文档（同一次上传的附件）一并纳入范围——
    # 记录里的说法常指向它自己那份文件/照片（"我微信头像长什么样"、"最适合我的一篇论文"），
    # 不传递的话模型拿不到文档片段、也看不到这些图片
    if record_raw_ids:
        for doc_id in session.scalars(
            select(Document.id).where(
                Document.raw_id.in_(record_raw_ids), Document.deleted_at.is_(None)
            )
        ):
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
    raws = (
        {r.id: r for r in session.scalars(select(RawEvidence).where(RawEvidence.id.in_(record_raw_ids))).all()}
        if record_raw_ids else {}
    )
    docs = (
        {d.id: d for d in session.scalars(select(Document).where(Document.id.in_(doc_ids))).all()}
        if doc_ids else {}
    )

    # 第二级：只在命中材料内检索细节
    fact_ids, chunk_ids = retrieval.search_details_in_materials(
        session, queries, record_raw_ids=record_raw_ids, doc_ids=doc_ids,
        top_k=_DETAILS_TOP_K, model=settings.embedding_model,
        rerank_engine=settings.rerank_engine, rerank_model=settings.rerank_model,
        client=client, llm_model=settings.llm_model,
    )
    facts = _ordered(session, Fact, fact_ids)
    chunks = _ordered(session, Chunk, chunk_ids)

    ctx = toolbox.ToolContext(session=session, settings=settings, question=question)
    user_content = _build_user_content(
        session, question, mats, raws, docs, facts, chunks, doc_ids=doc_ids
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    final = finalize_answer(_loop(client, settings, messages, ctx))
    # 口径落地（spec 2026-10-16-stats-param-guard §3.4）：本轮统计调用过的口径，
    # 答案里没写就由**代码**补上——不靠提示词（模型可以不听），保证用户一定看得见。
    final = param_guard.with_stats_caliber(final, ctx.stats_calls)

    sources, documents = _provenance(session, facts, chunks, ctx)
    execution = {"trace": ctx.tool_trace} if ctx.tool_trace else {}
    if ctx.used_image_ids:
        from app.models.db import Image
        execution["images"] = [{"image_id": image.id, "document_id": image.document_id,
                                 "title": docs.get(image.document_id).title if docs.get(image.document_id) else "文档图片",
                                 "page": image.page} for image in session.scalars(
            select(Image).join(Document, Document.id == Image.document_id).where(
                Image.id.in_(ctx.used_image_ids), Document.deleted_at.is_(None)))]
    return QueryResponse(answer=final, sources=sources, documents=documents, execution=execution)


def _ordered(session: Session, model_cls, ids: list[str]) -> list:
    """按给定 id 顺序取行（缺失的跳过）。"""
    rows = session.scalars(select(model_cls).where(model_cls.id.in_(ids))).all()
    by_id = {r.id: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def _loop(client, settings: Settings, messages: list, ctx: toolbox.ToolContext) -> str:
    """主循环：模型直接答 或 调工具（≤ qa_max_rounds 轮）；轮数耗尽后强制终答。"""
    rounds = max(1, int(settings.qa_max_rounds or 1))
    for _ in range(rounds):
        resp = client.chat.completions.create(
            model=settings.llm_model, messages=messages, tools=toolbox.TOOL_SCHEMAS
        )
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            return (msg.content or "").strip()
        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in calls
            ],
        }
        # Thinking APIs require reasoning fields/signatures in subsequent tool turns.
        if hasattr(msg, "model_dump"):
            assistant_message = msg.model_dump(exclude_none=True)
        messages.append(assistant_message)
        for tc in calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments not a JSON object")
            except Exception:
                result = "（工具参数不是合法 JSON）"
            else:
                result = toolbox.run_tool(ctx, tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # 轮数耗尽：禁用工具，基于已回填的内容拿终答（spec §5.3）
    resp = client.chat.completions.create(
        model=settings.llm_model, messages=messages,
        tools=toolbox.TOOL_SCHEMAS, tool_choice="none",
    )
    return (resp.choices[0].message.content or "").strip()


_COVERED_NOTE = "（已被更晚记录覆盖）"  # 记录材料的事实全部失效时的行尾标注（读侧标注，不删旧记录）


def _covered_record_raw_ids(session: Session, raw_ids: list[str]) -> set[str]:
    """「事实全部已失效」的记录 raw id 集合（材料清单行尾加覆盖标注用；一次批量查询，防 N+1）。

    口径：该 raw 有 ≥1 条事实、且全部已失效（`valid_to` 非空或 `archived != 0`）→ 标。
    raw 无事实（raw_only）或事实有活有死（混合）→ 不标（v1 局限）。
    """
    if not raw_ids:
        return set()
    alive: dict[str, list[bool]] = {}
    rows = session.execute(
        select(Fact.raw_id, Fact.valid_to, Fact.archived).where(Fact.raw_id.in_(raw_ids))
    ).all()
    for raw_id, valid_to, archived in rows:
        alive.setdefault(raw_id, []).append(valid_to is None and archived == 0)
    return {raw_id for raw_id, flags in alive.items() if not any(flags)}


def _build_user_content(session: Session, question: str, mats: list[Material],
                        raws: dict, docs: dict, facts: list[Fact],
                        chunks: list[Chunk], *, doc_ids: list[str]) -> str:
    """组装给模型的材料：清单 + 命中细节 + 表格/图片清单（全部过隐私闸）。

    表格/图片清单用**扩展后**的 doc_ids（含命中记录的同组文档）——组关系传递不能在这里被丢掉。
    记录材料的事实若全部失效（已被更晚记录取代）→ 行尾标注「已被更晚记录覆盖」：
    账本语义下旧记录仍展示，但模型不被它误导（不删、不改写侧）。
    """
    today = to_local(utcnow())
    weekday = "周一周二周三周四周五周六周日"[today.weekday() * 2: today.weekday() * 2 + 2]
    sections = [f"今天：{today.strftime('%Y-%m-%d')}（{weekday}）",
                f"问题：{question}", "## 材料清单"]
    covered_raws = _covered_record_raw_ids(
        session, [m.source_id for m in mats if m.source_type == "record"]
    )

    for m in mats:
        if m.source_type == "document":
            doc = docs.get(m.source_id)
            allowed = bool(doc and doc.allow_cloud)
            date = to_local(doc.created_at).date().isoformat() if doc and doc.created_at else ""
            title = toolbox.gate(m.title or "（无标题）", allowed)
            summary = toolbox.gate(m.summary or "", allowed)
            sections.append(f"- [文档] {title}（{date}）id={m.source_id}：{summary}")
        else:
            raw = raws.get(m.source_id)
            date = to_local(raw.created_at).date().isoformat() if raw and raw.created_at else ""
            title = toolbox.gate(m.title or "（无标题）", False)
            summary = toolbox.gate(m.summary or "", False)
            note = _COVERED_NOTE if m.source_id in covered_raws else ""
            sections.append(f"- [记录] {title}（{date}）：{summary}{note}")

    if facts or chunks:
        sections.append("## 相关内容（检索命中）")
        for f in facts:
            raw = session.get(RawEvidence, f.raw_id) if f.raw_id else None
            text = f.summary or ""
            if raw and raw.content:
                text += "\n" + raw.content[:3000]
            sections.append("- 记忆：" + toolbox.gate(text, False))
        for c in chunks:
            doc = get_document(session, c.document_id)
            if doc is None or doc.deleted_at is not None:
                continue
            allowed = bool(doc.allow_cloud)
            title = f"《{doc.title}》" if doc.title else ""
            text = c.text or ""
            if c.parent_text:
                text += "\n" + c.parent_text[:3000]
            sections.append(f"- 文档片段：{title}{toolbox.gate(text, allowed)}")

    tables = visible_doc_tables(session, doc_ids)
    if tables:
        sections.append("## 可用表格（统计问题调用 table_stats）")
        for t in tables:
            doc = docs.get(t.document_id)
            allowed = bool(doc and doc.allow_cloud)
            try:
                headers = " | ".join(json.loads(t.headers))
            except Exception:
                headers = ""
            note = "（该表格解析可能不完整）" if t.suspect else ""
            title = f"《{doc.title}》" if doc and doc.title else ""
            sections.append(
                f"- 表 {t.id}：{title}表头：{toolbox.gate(headers, allowed)}{note}"
            )
    images = visible_images(session, doc_ids)[:_IMAGES_LIST_MAX]
    if images:
        sections.append("## 可用图片（看图调用 view_image）")
        for img in images:
            doc = docs.get(img.document_id)
            allowed = bool(doc and doc.allow_cloud)
            desc = img.caption or (img.ocr_text or "")[:100] or "无描述"
            sections.append(
                f"- 图 {img.id}：{image_source(doc, img)}（{toolbox.gate(desc, allowed)}）"
            )

    return "\n".join(sections)


def _provenance(session: Session, facts: list[Fact], chunks: list[Chunk],
                ctx: toolbox.ToolContext) -> tuple[list[Source], list[DocumentSource]]:
    """溯源：命中记忆 + **工具读过的明细 fact**（如 memory_stats 的数字来源）+ 工具读过的文档片段。"""
    ordered = list(facts)
    seen_facts = {f.id for f in ordered}
    for fact_id in ctx.used_fact_ids:  # 工具读过的明细也要能点回记录
        if fact_id in seen_facts:
            continue
        fact = session.get(Fact, fact_id)
        if fact is not None:
            seen_facts.add(fact_id)
            ordered.append(fact)
    # 附件映射：来源要能点开原件（前端拼 /api/files/{id}/raw 需要 doc id）——一次查询防 N+1
    raw_ids = [f.raw_id for f in ordered if f.raw_id]
    docs_by_raw: dict[str, list[str]] = {}
    if raw_ids:
        for doc in session.scalars(
            select(Document).where(
                Document.raw_id.in_(raw_ids), Document.deleted_at.is_(None)
            ).order_by(Document.created_at)
        ):
            docs_by_raw.setdefault(doc.raw_id, []).append(doc.id)

    sources = []
    for f in ordered:
        raw = session.get(RawEvidence, f.raw_id) if f.raw_id else None
        sources.append(
            Source(
                fact_id=f.id,
                summary=f.summary,
                raw=raw.content if raw else None,
                file=raw.file_path if raw else None,
                raw_id=f.raw_id,
                doc_ids=docs_by_raw.get(f.raw_id, []),
            )
        )

    documents: list[DocumentSource] = []
    seen: set[str] = set()

    def add_chunk(chunk: Chunk) -> None:
        if chunk.id in seen:
            return
        seen.add(chunk.id)
        doc = get_document(session, chunk.document_id)
        documents.append(
            DocumentSource(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                title=doc.title if doc else None,
                heading_path=chunk.heading_path,
                page=chunk.page,
                text=chunk.text,
            )
        )

    for c in chunks:
        add_chunk(c)
    for doc_id in ctx.used_doc_ids:
        rows = session.scalars(
            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.seq).limit(_TOOL_DOC_CHUNKS)
        ).all()
        for c in rows:
            add_chunk(c)
    return sources, documents
