"""混合检索：向量 + 关键词，RRF 融合。"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import llm, rerank
from app.memory import chunks, fts, materials, vector
from app.memory.store import fact_search_text
from app.models.db import Chunk, Fact

# 细节检索（_fuse_and_rerank）的重排候选池上限：默认 top_k=10 时原为 top_k*3=30，实测本地
# 重排 30 对 6.65s、15 对 3.40s（默认 512 档）——池减半直接省一半耗时。代价：RRF 尾部第
# 16-30 名不再进重排（它们本就更难进最终 top_k）；max(top_k, ...) 保底不低于 top_k。
_RERANK_CANDIDATES = 15


def reciprocal_rank_fusion(rankings, k: int = 60):
    scores: dict = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _inactive_fact_ids(session: Session, cand_ids: list[str] | set[str]) -> set[str]:
    """候选中已失效的 fact id：已被取代（valid_to 非空）或已归档——读侧不再作为当前内容出现。

    双时态取代只置 `valid_to`/`superseded_by`（行留档、索引随留档保留）：检索召回后必须
    在这里过滤，否则旧值原样进问答上下文（用户实测：同一 subject_key 的两条矛盾记录）。
    过滤口径同 reminders.py：保留 `valid_to IS NULL AND archived == 0`。
    无 fact 行的孤儿索引 id 不在此列（照旧保留，与 fts/vector 的既有可见性口径一致）。
    """
    if not cand_ids:
        return set()
    rows = session.execute(
        select(Fact.id, Fact.valid_to, Fact.archived).where(Fact.id.in_(list(cand_ids)))
    ).all()
    return {fid for fid, valid_to, archived in rows if not (valid_to is None and archived == 0)}


def hybrid_retrieve_multi(session: Session, queries: list[str], top_k: int = 10,
                          model: str = llm.DEFAULT_EMBEDDING_MODEL,
                          rerank_engine: str = "off", rerank_model: str = rerank.DEFAULT_RERANK_MODEL,
                          client=None, llm_model: str = "") -> list[str]:
    """多查询混合检索（记忆）：每条查询各跑向量+关键词，榜单一并 RRF 融合后重排。

    queries[0] 应为用户原问题——重排以它为准衡量与真实意图的相关性。
    候选按 top_k*3 放宽（「先取候选再重排」的既有模式）：读侧双时态过滤剔除已取代
    旧值后，仍能从池中补足 top_k 条当前有效的记忆（而不是截断到 top_k 后再筛）。
    """
    queries = [q for q in queries if q and q.strip()]
    candidate_k = max(top_k * 3, top_k)
    rankings = []
    for q in queries:
        rankings.append(vector.search(session, llm.embed(q, model=model), candidate_k))
        rankings.append(fts.search(session, q, candidate_k))
    fused = reciprocal_rank_fusion(rankings)
    cand_ids = [fid for fid, _ in fused[:candidate_k]]
    inactive = _inactive_fact_ids(session, cand_ids)
    cand_ids = [fid for fid in cand_ids if fid not in inactive]
    if rerank_engine == "off" or len(cand_ids) <= 1:
        return cand_ids[:top_k]
    texts = [fact_search_text(session, fid) for fid in cand_ids]
    order = rerank.rerank(queries[0], texts, engine=rerank_engine, model=rerank_model,
                          client=client, llm_model=llm_model)
    return [cand_ids[i] for i in order][:top_k]


def hybrid_retrieve(session: Session, q: str, top_k: int = 10, model: str = llm.DEFAULT_EMBEDDING_MODEL,
                    rerank_engine: str = "off", rerank_model: str = rerank.DEFAULT_RERANK_MODEL,
                    client=None, llm_model: str = "") -> list[str]:
    """单查询混合检索（hybrid_retrieve_multi 的薄封装）。"""
    return hybrid_retrieve_multi(
        session, [q], top_k=top_k, model=model, rerank_engine=rerank_engine,
        rerank_model=rerank_model, client=client, llm_model=llm_model,
    )


def search_documents_multi(session, queries: list[str], top_k=10,
                           model=llm.DEFAULT_EMBEDDING_MODEL,
                           rerank_engine="off", rerank_model=rerank.DEFAULT_RERANK_MODEL,
                           client=None, llm_model="") -> list[str]:
    """多查询文档分块检索：每条查询各跑向量+关键词，榜单一并 RRF 融合后重排（queries[0] 为原问题）。"""
    queries = [q for q in queries if q and q.strip()]
    rankings = []
    for q in queries:
        rankings.append(chunks.vector_search(session, llm.embed(q, model=model), top_k))
        rankings.append(chunks.fts_search(session, q, top_k))
    fused = reciprocal_rank_fusion(rankings)
    candidate_k = max(top_k * 3, top_k)
    cand_ids = [cid for cid, _ in fused[:candidate_k]]
    if rerank_engine == "off" or len(cand_ids) <= 1:
        return cand_ids[:top_k]
    texts = [chunks.chunk_text(session, cid) for cid in cand_ids]
    order = rerank.rerank(queries[0], texts, engine=rerank_engine, model=rerank_model,
                          client=client, llm_model=llm_model)
    return [cand_ids[i] for i in order][:top_k]


def search_documents(session, q, top_k=10, model=llm.DEFAULT_EMBEDDING_MODEL,
                     rerank_engine="off", rerank_model=rerank.DEFAULT_RERANK_MODEL,
                     client=None, llm_model="") -> list[str]:
    """单查询文档分块检索（search_documents_multi 的薄封装）。"""
    return search_documents_multi(
        session, [q], top_k=top_k, model=model, rerank_engine=rerank_engine,
        rerank_model=rerank_model, client=client, llm_model=llm_model,
    )


def search_materials_multi(session: Session, queries: list[str], top_k: int = 10,
                           model: str = llm.DEFAULT_EMBEDDING_MODEL) -> list[str]:
    """材料层多查询检索：每条查询各跑向量+关键词，榜单一并 RRF 融合取 top_k（返回材料 id）。

    v1 不做 rerank（材料摘要短、RRF 融合已够用；后续需要再按 search_documents_multi 的模式加）。
    可见性（源对象软删/不存在 → 材料查不到）由 materials.fts_search/vector_search 内置过滤。
    """
    queries = [q for q in queries if q and q.strip()]
    rankings = []
    for q in queries:
        rankings.append(materials.vector_search(session, llm.embed(q, model=model), top_k))
        rankings.append(materials.fts_search(session, q, top_k))
    fused = reciprocal_rank_fusion(rankings)
    return [mid for mid, _ in fused[:top_k]]


def search_details_in_materials(
    session: Session, queries: list[str], *, record_raw_ids: list[str], doc_ids: list[str],
    top_k: int = 10, model: str = llm.DEFAULT_EMBEDDING_MODEL,
    rerank_engine: str = "off", rerank_model: str = rerank.DEFAULT_RERANK_MODEL,
    client=None, llm_model: str = "",
) -> tuple[list[str], list[str]]:
    """只在命中材料内检索细节（spec §4 两级检索的第二级）：facts 限命中记录、chunks 限命中文档。

    候选放宽到 top_k*5，按「属于命中材料」过滤后再 RRF 融合 + 重排（queries[0] 为准）；
    空集合的一侧直接返回空列表（不检索）。
    """
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return [], []
    raw_set, doc_set = set(record_raw_ids), set(doc_ids)
    pool = max(top_k * 5, top_k)
    fact_rankings: list[list[str]] = []
    chunk_rankings: list[list[str]] = []
    for q in queries:
        qv = llm.embed(q, model=model)
        if raw_set:
            for cand in (vector.search(session, qv, pool), fts.search(session, q, pool)):
                keep = _fact_ids_in_raws(session, cand, raw_set)
                fact_rankings.append([i for i in cand if i in keep])
        if doc_set:
            for cand in (chunks.vector_search(session, qv, pool), chunks.fts_search(session, q, pool)):
                keep = _chunk_ids_in_docs(session, cand, doc_set)
                chunk_rankings.append([i for i in cand if i in keep])
    fact_ids = _fuse_and_rerank(
        session, fact_rankings, queries[0], top_k, rerank_engine, rerank_model,
        client, llm_model, lambda fid: fact_search_text(session, fid),
    )
    chunk_ids = _fuse_and_rerank(
        session, chunk_rankings, queries[0], top_k, rerank_engine, rerank_model,
        client, llm_model, lambda cid: chunks.chunk_text(session, cid),
    )
    return fact_ids, chunk_ids


def _fact_ids_in_raws(session: Session, cand_ids: list[str], raw_set: set[str]) -> set[str]:
    """候选 fact id 中，raw_id 属于命中记录集合、且仍为当前有效值（未取代/未归档）的那些。"""
    if not cand_ids:
        return set()
    rows = session.execute(select(Fact.id, Fact.raw_id).where(Fact.id.in_(cand_ids))).all()
    in_raws = {fid for fid, rid in rows if rid in raw_set}
    return in_raws - _inactive_fact_ids(session, in_raws)


def _chunk_ids_in_docs(session: Session, cand_ids: list[str], doc_set: set[str]) -> set[str]:
    """候选 chunk id 中，document_id 属于命中文档集合的那些。"""
    if not cand_ids:
        return set()
    rows = session.execute(
        select(Chunk.id, Chunk.document_id).where(Chunk.id.in_(cand_ids))
    ).all()
    return {cid for cid, did in rows if did in doc_set}


def _fuse_and_rerank(session: Session, rankings: list[list[str]], primary_query: str, top_k: int,
                     rerank_engine: str, rerank_model: str, client, llm_model: str,
                     text_of) -> list[str]:
    """RRF 融合 → 重排（候选池上限 _RERANK_CANDIDATES，见常量注释；保底不低于 top_k）。"""
    fused = reciprocal_rank_fusion(rankings)
    candidate_k = max(top_k, min(top_k * 3, _RERANK_CANDIDATES))
    cand_ids = [cid for cid, _ in fused[:candidate_k]]
    if not cand_ids:
        return []
    if rerank_engine == "off" or len(cand_ids) <= 1:
        return cand_ids[:top_k]
    texts = [text_of(cid) for cid in cand_ids]
    order = rerank.rerank(primary_query, texts, engine=rerank_engine, model=rerank_model,
                          client=client, llm_model=llm_model)
    return [cand_ids[i] for i in order][:top_k]
