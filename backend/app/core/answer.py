"""问答合成 + 溯源（记忆 fact + 文档 chunk 统一检索）。"""
import json
import logging
from collections import Counter
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import cloud, llm, qa, retrieval, rewrite, router, stats, summarize
from app.core.qa import finalize_answer, INSUFFICIENT_MARKER
from app.core.router import QueryKind
from app.memory.documents import (
    get_chunk,
    get_document,
    image_source,
    visible_doc_tables,
    visible_images,
)
from app.models.db import Document, Fact, RawEvidence
from app.models.schemas import DocumentSource, QueryResponse, Source

logger = logging.getLogger(__name__)

# VLM 提示词：先中性描述全图（不预设问题相关），再针对用户问题作答——
# 避免「不相关的图 + 具体问题」被 VLM 误解（如拿代码截图问颜色）。
_DESCRIBE_PROMPT = (
    "用户的问题是：{question}\n\n"
    "请先完整描述这张图片的内容（图片类型、关键信息、图中出现的文字与数字），再针对用户的问题给出回答。"
)

# 「答不出」协议：与工具循环共用同一套收尾（qa.finalize_answer）——合成判不足输出标记，
# 收尾剥标记：整段就是标记 → 换可读文案；带解释的详答保留
_INSUFFICIENT = INSUFFICIENT_MARKER

# 统计路径降级原因（_answer_analysis 的 reason）→ 如实说明文案
_ANALYSIS_DEGRADE_NOTES = {
    "no_chunks": "未检索到相关文档",
    "no_tables": "未找到可统计的表格",
    "extract_failed": "未能确定统计条件",
    "compose_failed": "未能组织统计结果",
    "error": "统计路径异常",
}


def synthesize(client, model: str, question: str, contexts: list[str],
               images: list[str] | None = None) -> str:
    """合成回答：图片描述单独成段并置前（检索片段常有噪声，图片内容是强信号）。"""
    parts = [f"问题：{question}"]
    if images:
        parts.append("文档中图片的内容（由图片理解模型识别）：\n" + "\n".join(f"- {c}" for c in images))
    if contexts:
        parts.append("文本相关内容：\n" + "\n".join(f"- {c}" for c in contexts))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "根据给定的记忆和文档内容回答用户问题，只依据给定内容，不要编造。"
                f"若给定内容不足以回答问题，只输出：{_INSUFFICIENT}，不要编造。"
            )},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    return resp.choices[0].message.content


def _describe_retrieved_images(
    session: Session, chunk_ids: list[str], question: str, settings: Settings, force: bool = False
) -> list[str]:
    """对命中 chunk 所属文档的图片调用 VLM 描述，返回带前缀的描述列表。

    - vlm_model 为空（VLM 未配置）或无可定位文档 → 空列表。
    - force=True 且 chunk 未命中时，回退到所有允许上云的文档（限少量图片）。
    - 单张失败（缺文件 / VLM 报错）一律跳过，不影响回答主链路。
    """
    if not settings.vlm_model:
        return []
    doc_ids: list[str] = []
    if chunk_ids:
        for cid in chunk_ids:
            chunk = get_chunk(session, cid)
            if chunk is not None and chunk.document_id not in doc_ids:
                doc_ids.append(chunk.document_id)
    elif force:
        doc_ids = list(
            session.scalars(select(Document.id).where(Document.allow_cloud != 0)).all()
        )
    if not doc_ids:
        return []

    docs = session.scalars(select(Document).where(Document.id.in_(doc_ids))).all()
    allowed_ids = [d.id for d in docs if d.allow_cloud]
    if not allowed_ids:
        return []

    images = visible_images(session, allowed_ids)

    vlm_client = cloud.build_vlm_client(settings)
    doc_by_id = {d.id: d for d in docs}
    descriptions: list[str] = []
    for image in images[:3]:
        if not image.file_path:
            continue
        try:
            image_bytes = (Path(settings.data_dir) / image.file_path).read_bytes()
        except Exception:
            continue
        try:
            desc = cloud.describe_image(
                vlm_client, settings.vlm_model, image_bytes,
                _DESCRIBE_PROMPT.format(question=question),
            )
        except Exception:
            continue
        if desc:
            descriptions.append(f"图片描述（{image_source(doc_by_id.get(image.document_id), image)}）：{desc}")
    return descriptions


def _doc_fact_summaries(session: Session, doc: Document) -> list[str]:
    """文档关联的记忆摘要（经 RawEvidence.file_path 关联的 fact，排除已进回收站的记录）。

    已删记录不仅不可见，也不再影响「总结选哪份文档」的身份打分材料。
    """
    if not doc.file_path:
        return []
    rows = session.execute(
        select(Fact.summary)
        .outerjoin(RawEvidence, RawEvidence.id == Fact.raw_id)
        .where(
            RawEvidence.file_path == doc.file_path,
            or_(RawEvidence.id.is_(None), RawEvidence.deleted_at.is_(None)),
        )
    ).scalars().all()
    return [r for r in rows if r]


def _identify_target_document(
    session: Session, question: str, hit_chunks: list, settings: Settings
) -> str | None:
    """总结目标文档选择：身份匹配（标题+关联记忆 的语义相似度）为主、chunk 命中数为辅。

    「用户自己对这份资料的记忆」（如「机器人课程的大作业是一篇论文」）是
    「我上传的那个XX」类指代问题的正确线索；纯 chunk 命中常被通用词（总结/论文/课程）稀释。
    """
    doc_counts = Counter(c.document_id for c in hit_chunks)
    if not doc_counts:
        return None
    docs = session.scalars(select(Document).where(Document.id.in_(doc_counts))).all()
    qv = llm.embed(question, model=settings.embedding_model)
    best_id, best_score = None, float("-inf")
    for doc in docs:
        identity = " ".join([doc.title or ""] + _doc_fact_summaries(session, doc)).strip()
        similarity = 0.0
        if identity:
            dv = llm.embed(identity, model=settings.embedding_model)
            similarity = sum(a * b for a, b in zip(qv, dv))  # bge-m3 已归一化，点积即余弦
        score = similarity + 0.01 * doc_counts[doc.id]  # 命中数仅作同分时的细微倾向
        if score > best_score:
            best_id, best_score = doc.id, score
    return best_id


def _answer_analysis(
    session: Session, question: str, settings: Settings, client, chunk_ids: list[str]
) -> tuple[QueryResponse | None, str | None]:
    """表格条件统计路径：命中 chunk 所属文档的表格 → LLM 抽条件 → 代码筛选/聚合 → 渲染/组答。

    返回 (答案, None) 或 (None, 降级原因)：任一步不成立（无命中/无表/抽取失败/组答失败/
    表格数据损坏）→ 失败并给出原因，调用方降级普通问答时如实说明；
    「json 解析 → QueryResponse 构造」整段密封，任何异常都不外泄（reason=error）。
    """
    if not chunk_ids:
        return None, "no_chunks"

    doc_ids: list[str] = []
    for cid in chunk_ids:
        chunk = get_chunk(session, cid)
        if chunk is not None and chunk.document_id not in doc_ids:
            doc_ids.append(chunk.document_id)
    if not doc_ids:
        return None, "no_chunks"

    tables = visible_doc_tables(session, doc_ids)
    if not tables:
        return None, "no_tables"

    try:
        # 每个候选表只解析一次 JSON：样本行与后续执行共用同一份数据
        parsed: dict[str, tuple[list[str], list[list[str]]]] = {}
        candidates = []
        for t in tables:
            headers = json.loads(t.headers)
            rows = json.loads(t.rows)
            parsed[t.id] = (headers, rows)
            candidates.append(
                stats.TableCandidate(id=t.id, headers=headers, sample_rows=rows[:3])
            )
        filt = stats.extract_filter(client, settings.llm_model, question, candidates)
        if filt is None:
            return None, "extract_failed"

        headers, rows = parsed[filt.table_id]  # extract_filter 已保证 table_id ∈ 候选表
        outcome = stats.apply_filter(headers, rows, filt)
        answer_text = stats.compose_answer(client, settings.llm_model, question, headers, outcome)
        if not answer_text:
            logger.warning("统计组答返回空答案（降级普通问答）")
            return None, "compose_failed"
        # 作答所用的表被标 suspect（解析可能不完整；落库为 0/1 int）→ 答案末尾如实提示
        used_table = next(t for t in tables if t.id == filt.table_id)
        if used_table.suspect:
            answer_text += "（该表格解析可能不完整）"

        # 溯源：命中 chunk 的文档片段（与普通问答路径同款构造）
        documents = []
        for cid in chunk_ids:
            chunk = get_chunk(session, cid)
            if chunk is None:
                continue
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
        return QueryResponse(answer=answer_text, sources=[], documents=documents), None
    except Exception:
        logger.warning("统计分析路径失败（降级普通问答）", exc_info=True)
        return None, "error"


def answer(session: Session, question: str, settings: Settings) -> QueryResponse:
    """问答入口：按 `qa_engine` 分发（spec §5）；工具循环异常 → 回退旧管线（不允许不可用期）。"""
    from app.core import document_actions
    required = document_actions.try_handle(session, question, settings)
    if required is not None:
        return required
    if settings.qa_engine == "legacy":
        return answer_legacy(session, question, settings)
    try:
        return qa.run(session, question, settings)
    except Exception:
        logger.warning("工具循环问答异常（回退旧管线）", exc_info=True)
        return answer_legacy(session, question, settings)


def answer_legacy(session: Session, question: str, settings: Settings) -> QueryResponse:
    client = llm.build_client(settings)
    kind = router.classify(question, client, settings.llm_model)
    # 检索用查询列表（分级触发多查询扩展，queries[0] 恒为原问题）；路由/合成/图片判定一律用原问题
    queries = rewrite.retrieval_queries(question, settings, client)
    if kind == QueryKind.summarize:
        chunk_ids = retrieval.search_documents_multi(
            session, queries, top_k=10, model=settings.embedding_model,
            rerank_engine=settings.rerank_engine, rerank_model=settings.rerank_model,
            client=client, llm_model=settings.llm_model,
        )
        if chunk_ids:
            # 定位目标文档：身份匹配（标题+关联记忆）为主、chunk 命中数为辅
            hit_chunks = [c for c in (get_chunk(session, cid) for cid in chunk_ids) if c is not None]
            target_doc_id = _identify_target_document(session, question, hit_chunks, settings)
            if target_doc_id:
                summary_text = summarize.summarize_document(
                    session, target_doc_id, client=client, model=settings.llm_model
                )
                # 溯源：返回目标文档的命中 chunk 片段
                documents = []
                for chunk in hit_chunks:
                    if chunk.document_id != target_doc_id:
                        continue
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
                return QueryResponse(answer=summary_text, sources=[], documents=documents)
        # 无命中文档 → 落到原有混合检索兜底（不 return，继续往下走）

    # 文档检索一次：analysis 分支与普通路径共用（summarize 分支自算不动）
    chunk_ids = retrieval.search_documents_multi(
        session, queries, top_k=10, model=settings.embedding_model,
        rerank_engine=settings.rerank_engine, rerank_model=settings.rerank_model,
        client=client, llm_model=settings.llm_model,
    )
    degrade_note = ""
    if kind == QueryKind.analysis_query:
        resp, reason = _answer_analysis(session, question, settings, client, chunk_ids)
        if resp is not None:
            return resp
        # 无命中/无表/抽取失败/组答失败 → 落到原有混合检索（降级普通问答），如实说明原因
        degrade_note = (
            f"（{_ANALYSIS_DEGRADE_NOTES.get(reason, '未能完成表格统计')}，以下为基于原文的回答）\n"
        )

    fact_ids = retrieval.hybrid_retrieve_multi(
        session, queries, top_k=10, model=settings.embedding_model,
        rerank_engine=settings.rerank_engine, rerank_model=settings.rerank_model,
        client=client, llm_model=settings.llm_model,
    )
    if not fact_ids and not chunk_ids:
        # 无命中兜底小路径：问题指向图片（关键词命中或 LLM 路由）才 force 描述上云文档的图
        if kind == QueryKind.image_query or router.mentions_images(question):
            image_descriptions = _describe_retrieved_images(
                session, chunk_ids, question, settings, force=True
            )
            if image_descriptions:
                try:
                    answer_text = synthesize(client, settings.llm_model, question, [], images=image_descriptions)
                except Exception:
                    answer_text = "（回答合成失败，以下为检索到的相关记录）"
                answer_text = finalize_answer(answer_text)
                return QueryResponse(answer=degrade_note + answer_text, sources=[], documents=[])
        return QueryResponse(answer=degrade_note + "没有找到相关记录", sources=[])

    facts = session.scalars(select(Fact).where(Fact.id.in_(fact_ids))).all()
    by_id = {f.id: f for f in facts}
    ordered = [by_id[i] for i in fact_ids if i in by_id]

    # 上下文 = 记忆（摘要 + 原文）+ 文档片段（小块 + 父节），供 LLM 总结/回答
    contexts = []
    for f in ordered:
        raw = session.get(RawEvidence, f.raw_id) if f.raw_id else None
        ctx = f.summary or ""
        if raw and raw.content:
            ctx += "\n" + raw.content[:3000]
        contexts.append(ctx)

    documents = []
    for cid in chunk_ids:
        chunk = get_chunk(session, cid)
        if chunk is None:
            continue
        doc = get_document(session, chunk.document_id)
        ctx = chunk.text or ""
        if chunk.parent_text:
            ctx += "\n" + chunk.parent_text[:3000]
        contexts.append(ctx)
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

    response_status = "ok"
    try:
        answer_text = synthesize(client, settings.llm_model, question, contexts)
    except Exception:
        # LLM 不可用：降级返回候选原文，不做综合
        answer_text = "（回答合成失败，以下为检索到的相关记录）"
        response_status = "degraded"

    # 兜底：合成标记「信息不足」且命中文档有图可看 → 看图（带问题）后重答一次
    if _INSUFFICIENT in answer_text:
        descriptions = _describe_retrieved_images(session, chunk_ids, question, settings)
        if descriptions:
            try:
                answer_text = synthesize(client, settings.llm_model, question, contexts, images=descriptions)
            except Exception:
                logger.warning("看图兜底合成失败（保留首轮输出）", exc_info=True)
    answer_text = finalize_answer(answer_text)

    sources = []
    for f in ordered:
        raw = session.get(RawEvidence, f.raw_id) if f.raw_id else None
        sources.append(
            Source(
                fact_id=f.id,
                summary=f.summary,
                raw=raw.content if raw else None,
                file=raw.file_path if raw else None,
            )
        )

    return QueryResponse(answer=degrade_note + answer_text, sources=sources, documents=documents,
                         status=response_status)
