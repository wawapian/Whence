"""问答接口。"""
import json
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.config import Settings
from app.core import answer, llm, summarize, document_actions
from app.deps import get_session, get_settings
from app.memory import qa_history
from app.models.db import QaHistory, Fact, RawEvidence, Document
from app.models.schemas import DocumentSource, QueryResponse, Source

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)

# 消解提示词：只补全「听懂」所缺的指代/省略，不回答、不解释；拿不准一律原样返回（保守回退）。
# 硬约束「严禁改名」（发现#2：实测把「耳机盒子呢」消解成「外包装盒放在哪里」——替用户换了实体名）。
_RESOLVE_PROMPT = (
    "你是追问消解助手：结合近期对话，把用户的当前问题补全成不依赖上文的独立问题。"
    "只做两件事——补全省略、消解指代；"
    "严禁改写、替换或重新命名用户给出的任何名词（人名、物品名、机构名等一律原样保留）；"
    "只输出补全后的问题本身——不要回答问题、不要解释、不要加引号；"
    "如果当前问题已经独立成立，或你拿不准该怎么补全，就原样返回当前问题。"
)


class QueryTurn(BaseModel):
    """一组历史问答（问题与答案都要——指代对象常出现在答案里）。"""

    question: str
    answer: str


class QueryIn(BaseModel):
    question: str
    history: list[QueryTurn] | None = None


def _resolve_question(question: str, history: list[QueryTurn] | None, settings: Settings) -> str:
    """指代消解：最近 `qa_history_window` 组 + 当前句 → 独立问题；异常/空输出 → 原句。

    历史只用于「听懂」（补全指代与省略），不喂回答环节；事实仍只来自记忆库。
    不阻断主流程：云调用任何异常、空输出都降级用原句（warning）。
    """
    if not history:
        return question
    if settings.qa_history_window <= 0:
        # 护栏（终审 Minor#2）：window<=0 时 history[-window:] 语义反转（0 → 全窗口，
        # 负值 → 反向裁剪）——直接原句返回，不发云调用（等同「不开窗口」的保守回退）。
        return question
    window = history[-settings.qa_history_window :]  # 服务端截取，防超量入参
    turns = "\n".join(f"用户：{t.question}\n回答：{t.answer}" for t in window)
    try:
        client = llm.build_client(settings)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _RESOLVE_PROMPT},
                {"role": "user", "content": f"近期对话：\n{turns}\n\n当前问题：{question}"},
            ],
        )
        resolved = (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("指代消解失败（用原句继续）", exc_info=True)
        return question
    if not resolved:
        logger.warning("指代消解返回空（用原句继续）")
        return question
    return resolved


def _find_cached(session: Session, question: str, settings: Settings) -> QaHistory | None:
    """查答案缓存（`log`/`on` 共用）：本地嵌入 + `qa_history.lookup`。

    无候选 / 低于阈值 / 版本不符 → None（调用方静默走管线，用户无感）。`log` 模式
    把「若开会否命中」的结果记 info（问题/匹配行 id/相似度/阈值）——攒真实命中数据；
    `on` 模式命中时打同款审计 info（spec §4），miss 不记（防噪音，miss 仅 log 档记）。
    """
    full_summary = summarize.is_summary_request(question)
    target_id = None
    if full_summary:
        from app.memory.documents import named_documents
        targets = named_documents(session, question)
        if len(targets) != 1:
            return None  # An ambiguous document must not be silently selected by cache similarity.
        target_id = targets[0].id
    vec = llm.embed(question, model=settings.embedding_model)
    row = qa_history.lookup(
        session,
        vec,
        threshold=settings.answer_cache_threshold,
        top_k=settings.answer_cache_top_k,
        ttl_hours=settings.qa_history_ttl_hours,
        require_full_summary=full_summary,
        required_document_id=target_id,
    )
    # 审计日志：log 档命中/miss 都记；on 档仅命中记
    if settings.answer_cache == "log" or row is not None:
        similarity = None
        if row is not None:
            similarity = sum(a * b for a, b in zip(qa_history.unpack_vec(row.question_vec), vec))
        logger.info(
            "答案缓存(%s)：问题=%s 匹配行=%s 相似度=%s 阈值=%s",
            settings.answer_cache,
            question,
            row.id if row is not None else None,
            f"{similarity:.4f}" if similarity is not None else None,
            settings.answer_cache_threshold,
        )
    return row


@router.post("/query")
def query(
    payload: QueryIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    # 顺序：消解（带窗口时）→ 查缓存（log/on 共用独立问题作键）→ 命中直答 / 未命中走管线 → 恒定落行
    resolved = _resolve_question(payload.question, payload.history, settings)
    # 查缓存块整体窄护栏：任何异常（如换 embedding 模型后旧行向量维度不符）→ warning +
    # 按未命中继续走管线。off 档不达此块；log 正常路径不受影响（护栏只在异常时介入）；
    # 管线调用与落行块各自的 try 均不在此范围内（红线不破）。
    resp = None
    # Original-file lookups are local and cheap; always refresh visibility and matching files.
    if settings.answer_cache in ("log", "on") and not document_actions.is_find_request(resolved):
        try:
            cached = _find_cached(session, resolved, settings)
            if cached is not None and settings.answer_cache == "on":
                resp = QueryResponse(
                    answer=cached.answer,
                    sources=[Source(**item) for item in json.loads(cached.sources)],
                    documents=[DocumentSource(**item) for item in json.loads(cached.documents)],
                    from_cache=True,
                    execution={k: v for k, v in json.loads(cached.execution or "{}").items() if k != "status"},
                )
        except Exception:
            logger.warning("答案缓存查询失败（按未命中走管线）", exc_info=True)
    if resp is None:
        resp = answer.answer(session, resolved, settings)
    # The legacy/error fallback also needs working local evidence links.
    # Read current visibility rather than relying on cached attachment ids.
    fact_ids = [source.fact_id for source in resp.sources]
    raws_by_fact = dict(session.execute(select(Fact.id, Fact.raw_id).join(
        RawEvidence, RawEvidence.id == Fact.raw_id).where(
        Fact.id.in_(fact_ids), RawEvidence.deleted_at.is_(None))).all()) if fact_ids else {}
    docs_by_raw = {}
    if raws_by_fact:
        for doc in session.scalars(select(Document).where(
            Document.raw_id.in_(raws_by_fact.values()), Document.deleted_at.is_(None))):
            docs_by_raw.setdefault(doc.raw_id, []).append(doc.id)
    for source in resp.sources:
        if source.fact_id in raws_by_fact:
            source.raw_id = raws_by_fact[source.fact_id]
            source.doc_ids = docs_by_raw.get(source.raw_id, [])
    resp = resp.model_copy(update={"resolved_question": resolved if payload.history else None})
    # 每次提问恒定落一行历史（resolved_question 写实际使用值；向量=独立问题，与缓存键同源）；
    # 落行失败只告警，不阻断回答（红线）。
    try:
        qa_history.record(
            session,
            question=payload.question,
            answer=resp.answer,
            sources=[s.model_dump() for s in resp.sources],
            documents=[d.model_dump() for d in resp.documents],
            question_vec=llm.embed(resolved, model=settings.embedding_model),
            resolved_question=resp.resolved_question,
            from_cache=resp.from_cache,
            execution={**resp.execution, "status": resp.status},
        )
        session.commit()
    except Exception:
        logger.warning("提问历史落行失败（不阻断回答）", exc_info=True)
    return resp
