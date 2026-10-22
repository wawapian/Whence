from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core import llm, retrieval
from app.memory import fts, vector
from app.models.db import init_db


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def test_rrf_fusion():
    fused = retrieval.reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    top = [doc for doc, _ in fused[:2]]
    assert "b" in top  # b 在双路都出现


def test_fts_index_and_search(tmp_path):
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "身份证放在书房抽屉")
    session.commit()
    assert fts.search(session, "身份证", k=5) == ["f1"]


def test_vector_index_and_search(tmp_path):
    session = _session(tmp_path)
    vec = [0.1] * 512
    vector.index_fact(session, "f1", vec)
    session.commit()
    assert vector.search(session, vec, k=5) == ["f1"]


def test_hybrid_returns_ids(tmp_path, monkeypatch):
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "身份证放在书房抽屉")
    vector.index_fact(session, "f1", [0.1] * 512)
    fts.index_fact(session, "f2", "今天买了鸡蛋")
    vector.index_fact(session, "f2", [0.9] * 512)
    session.commit()

    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)  # 查询向量接近 f1
    ids = retrieval.hybrid_retrieve(session, "身份证在哪", top_k=5)
    assert "f1" in ids


def test_hybrid_retrieve_forwards_model(tmp_path, monkeypatch):
    session = _session(tmp_path)
    captured = {}

    def fake_embed(q, model=None):
        captured["model"] = model
        return [0.1] * 512

    monkeypatch.setattr(llm, "embed", fake_embed)
    retrieval.hybrid_retrieve(session, "身份证", top_k=5, model="BAAI/bge-small-zh-v1.5")
    assert captured["model"] == "BAAI/bge-small-zh-v1.5"


def test_fact_search_text_mirrors_index(tmp_path):
    from app.memory import store
    from app.models.db import Fact, RawEvidence
    from datetime import datetime, timezone
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    raw = store.insert_raw(session, kind="text", content="正文内容" * 500)
    session.flush()
    f = Fact(id="f1", type="other", summary="摘要", valid_from=datetime.now(timezone.utc), raw_id=raw.id)
    session.add(f)
    session.commit()
    text = store.fact_search_text(session, "f1")
    assert text.startswith("摘要 ")
    assert "正文内容" in text


def test_hybrid_retrieve_honors_rerank_order(tmp_path, monkeypatch):
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "身份证放在书房抽屉")
    vector.index_fact(session, "f1", [0.1] * 512)
    fts.index_fact(session, "f2", "今天买了鸡蛋")
    vector.index_fact(session, "f2", [0.9] * 512)
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)

    # rerank 返回逆序，证明最终顺序由 rerank 决定
    def fake_rerank(query, candidates, *, engine="local", model=None, client=None, llm_model=""):
        return list(range(len(candidates)))[::-1]

    monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
    ids = retrieval.hybrid_retrieve(session, "身份证", top_k=2, rerank_engine="local")
    # query 向量近 f1，RRF 默认 f1 靠前；rerank 逆序后变 f2 靠前
    assert ids == ["f2", "f1"]


def test_search_documents_returns_chunk_ids(tmp_path, monkeypatch):
    session = _session(tmp_path)
    from app.memory import chunks, documents
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    c1 = documents.insert_chunk(session, document_id=doc.id, seq=0, text="学习率", parent_text="x")
    c2 = documents.insert_chunk(session, document_id=doc.id, seq=1, text="今天买了鸡蛋", parent_text="x")
    session.flush()
    chunks.index_chunk(session, c1.id, [0.1] * 512, "学习率")
    chunks.index_chunk(session, c2.id, [0.9] * 512, "今天买了鸡蛋")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    ids = retrieval.search_documents(session, "学习率", top_k=5)
    assert c1.id in ids


def test_search_documents_empty_query_returns_empty(tmp_path, monkeypatch):
    session = _session(tmp_path)
    from app.memory import chunks, documents
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    c1 = documents.insert_chunk(session, document_id=doc.id, seq=0, text="学习率", parent_text="x")
    session.flush()
    chunks.index_chunk(session, c1.id, [0.1] * 512, "学习率")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    assert retrieval.search_documents(session, "  ", top_k=5) == []
    assert retrieval.search_documents(session, "", top_k=5) == []


def test_search_documents_honors_rerank(tmp_path, monkeypatch):
    session = _session(tmp_path)
    from app.memory import chunks, documents
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    c1 = documents.insert_chunk(session, document_id=doc.id, seq=0, text="学习率", parent_text="x")
    c2 = documents.insert_chunk(session, document_id=doc.id, seq=1, text="今天买了鸡蛋", parent_text="x")
    session.flush()
    chunks.index_chunk(session, c1.id, [0.1] * 512, "学习率")
    chunks.index_chunk(session, c2.id, [0.9] * 512, "今天买了鸡蛋")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    def fake_rerank(query, candidates, *, engine="local", model=None, client=None, llm_model=""):
        return list(range(len(candidates)))[::-1]
    monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
    ids = retrieval.search_documents(session, "学习率", top_k=2, rerank_engine="local")
    assert ids == [c2.id, c1.id]


def test_fts_search_handles_punctuation_question(tmp_path):
    """用户自然语言问题带逗号等 FTS5 语法字符，不应报错且能命中关键词。"""
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "实验四 ROS catkin_package message_runtime 配置")
    session.commit()
    q = "官网没有在 catkin_package 中配置 message_runtime,配置还行不行"
    assert "f1" in fts.search(session, q, k=5)


def test_fts_search_punctuation_only_returns_empty(tmp_path):
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "身份证放在书房抽屉")
    session.commit()
    assert fts.search(session, "!!!,，。？！", k=5) == []


def test_fts_search_keeps_symbol_keywords(tmp_path):
    """带符号关键词（C++、node.js）整体加引号原样可命中，不因清洗丢失。"""
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "用 C++ 写的 node.js 服务")
    session.commit()
    assert "f1" in fts.search(session, "C++", k=5)
    assert "f1" in fts.search(session, "node.js", k=5)


def test_hybrid_retrieve_filters_superseded_in_candidate_pool(tmp_path, monkeypatch):
    """双时态读侧：已取代的旧值从候选池剔除，top_k 由当前有效值补足（不是截断到 top_k 后再筛）。"""
    from app.memory import bitemporal, store
    from app.models.db import Fact
    from app.models.schemas import ExtractedFact

    session = _session(tmp_path)

    def _record(content, summary, vec):
        raw = store.insert_raw(session, kind="text", content=content)
        session.flush()
        f = bitemporal.upsert(
            session,
            ExtractedFact(type="item_location", subject_key="身份证:位置",
                          summary=summary, confidence=0.9),
            raw.id,
        )
        session.flush()
        fts.index_fact(session, f.id, f"{summary} {content}")
        vector.index_fact(session, f.id, vec)
        return f

    f_old = _record("身份证我放在书房第一柜了", "身份证放在书房第一柜", [0.1] * 512)
    f_new = _record("身份证改放背包夹层了", "身份证放在背包夹层", [0.9] * 512)
    session.commit()

    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)  # 查询向量最近旧值
    assert retrieval.hybrid_retrieve(session, "身份证在哪", top_k=1) == [f_new.id]
    assert session.get(Fact, f_old.id).valid_to is not None  # 旧值留档（账本语义不变）


def test_hybrid_retrieve_multi_fuses_all_queries(tmp_path, monkeypatch):
    """多查询扩展：任一变体命中的记忆都进入融合结果（只建 FTS 索引，隔离向量路）。"""
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "第三组学习率 0.001")
    fts.index_fact(session, "f2", "今天买了鸡蛋")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)

    assert retrieval.hybrid_retrieve(session, "买了鸡蛋", top_k=5) == ["f2"]  # 单查询只命中 f2
    fused = retrieval.hybrid_retrieve_multi(session, ["买了鸡蛋", "学习率"], top_k=5)
    assert set(fused) == {"f1", "f2"}  # 变体把 f1 也带进来


def test_search_documents_multi_fuses_all_queries(tmp_path, monkeypatch):
    session = _session(tmp_path)
    from app.memory import chunks, documents
    doc = documents.insert_document(session, kind="word", status="done")
    session.flush()
    c1 = documents.insert_chunk(session, document_id=doc.id, seq=0, text="学习率", parent_text="x")
    c2 = documents.insert_chunk(session, document_id=doc.id, seq=1, text="今天买了鸡蛋", parent_text="x")
    session.flush()
    chunks.index_chunk(session, c1.id, [0.1] * 512, "学习率")
    chunks.index_chunk(session, c2.id, [0.9] * 512, "今天买了鸡蛋")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    monkeypatch.setattr(chunks, "vector_search", lambda *a, **k: [])  # 隔离向量路，只测 FTS 融合

    assert retrieval.search_documents(session, "买了鸡蛋", top_k=5) == [c2.id]
    fused = retrieval.search_documents_multi(session, ["买了鸡蛋", "学习率"], top_k=5)
    assert set(fused) == {c1.id, c2.id}


def test_hybrid_retrieve_multi_reranks_with_first_query(tmp_path, monkeypatch):
    """重排基准是 queries[0]（原问题），不是变体。"""
    session = _session(tmp_path)
    fts.index_fact(session, "f1", "学习率设置")
    fts.index_fact(session, "f2", "另一个记忆")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    captured = {}

    def fake_rerank(query, candidates, *, engine="local", model=None, client=None, llm_model=""):
        captured["query"] = query
        return list(range(len(candidates)))

    monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
    retrieval.hybrid_retrieve_multi(session, ["原问题", "学习率", "另一个"], top_k=5, rerank_engine="local")
    assert captured["query"] == "原问题"


def test_fuse_and_rerank_caps_candidate_pool(monkeypatch):
    """细节检索重排候选池上限 15（30 对 6.65s vs 15 对 3.40s）：尾部名次不再进重排，且保底 ≥top_k。"""
    captured = {}

    def fake_rerank(query, candidates, *, engine="local", model=None, client=None, llm_model=""):
        captured["candidates"] = list(candidates)
        return list(range(len(candidates)))

    monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
    rankings = [[f"c{i}" for i in range(25)]]  # 单路 RRF：25 条候选

    ids = retrieval._fuse_and_rerank(
        None, rankings, "q", 10, "local", "m", None, "", lambda cid: f"text-{cid}",
    )

    n = len(captured["candidates"])
    assert n <= 15  # 上限：第 16-25 名不进重排
    assert n >= 10  # 保底：不少于 top_k
    assert captured["candidates"] == [f"text-c{i}" for i in range(15)]  # 取 RRF 头部 15 条
    assert ids == [f"c{i}" for i in range(10)]  # 身份重排 → 前 top_k 原序返回


def test_fuse_and_rerank_keeps_all_when_fewer_than_cap(monkeypatch):
    """候选不足 15 条时全量进重排（不为凑上限补造候选）。"""
    captured = {}

    def fake_rerank(query, candidates, *, engine="local", model=None, client=None, llm_model=""):
        captured["candidates"] = list(candidates)
        return list(range(len(candidates)))

    monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
    rankings = [["c0", "c1", "c2"]]

    ids = retrieval._fuse_and_rerank(
        None, rankings, "q", 2, "local", "m", None, "", lambda cid: f"text-{cid}",
    )

    assert captured["candidates"] == ["text-c0", "text-c1", "text-c2"]
    assert ids == ["c0", "c1"]
