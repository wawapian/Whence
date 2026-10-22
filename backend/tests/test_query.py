import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.core import answer, llm, rewrite, router, stats
from app.main import create_app
from app.memory import fts, vector
from app.models.db import Fact, init_db


def test_no_result_fallback(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    session = Session(engine)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    resp = answer.answer(session, "不存在的记忆", Settings(qa_engine="legacy",data_dir=tmp_path))
    assert resp.answer == "没有找到相关记录"
    assert resp.sources == []


def test_legacy_context_excludes_superseded_facts(tmp_path, monkeypatch):
    """legacy 管线读侧双时态：已被取代的旧值不进合成上下文、也不进来源（账本语义不受影响）。"""
    from app.memory import bitemporal, store
    from app.models.schemas import ExtractedFact

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)

    def _record(content, summary):
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
        vector.index_fact(session, f.id, [0.1] * 512)
        session.commit()
        return f

    _record("身份证我放在书房第一柜了", "身份证放在书房第一柜")
    f_new = _record("身份证改放背包夹层了", "身份证放在背包夹层")

    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    captured = {}

    def fake_synthesize(client, model, q, contexts, images=None):
        captured["contexts"] = list(contexts)
        return "身份证放在背包夹层"

    monkeypatch.setattr(answer, "synthesize", fake_synthesize)
    resp = answer.answer(
        session, "身份证在哪",
        Settings(qa_engine="legacy", data_dir=tmp_path, rerank_engine="off"),
    )

    assert resp.answer == "身份证放在背包夹层"
    assert [s.fact_id for s in resp.sources] == [f_new.id]  # 旧值不进来源
    assert all("身份证放在书房第一柜" not in c for c in captured["contexts"])  # 旧值不进合成上下文


def test_query_returns_sources(tmp_path, monkeypatch):
    # 预置：Fact 行 + 索引
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    session.add(
        Fact(id="f1", type="item_location", summary="身份证放在书房抽屉",
             valid_from=datetime.now(timezone.utc))
    )
    session.flush()
    fts.index_fact(session, "f1", "身份证放在书房抽屉")
    vector.index_fact(session, "f1", [0.1] * 1024)
    session.commit()

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(qa_engine="legacy",data_dir=tmp_path)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize", lambda client, model, q, contexts, images=None: "身份证放在书房抽屉")

    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "身份证在哪"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "身份证放在书房抽屉"
    assert body["sources"][0]["fact_id"] == "f1"


def test_query_returns_documents(tmp_path, monkeypatch):
    from app.memory import chunks, documents

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", status="done", title="实验报告")
    session.flush()
    c = documents.insert_chunk(session, document_id=doc.id, seq=0,
                               text="第三组学习率 0.001", parent_text="第三组实验设置全文")
    session.flush()
    chunks.index_chunk(session, c.id, [0.1] * 1024, "第三组学习率 0.001")
    session.commit()

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize", lambda client, model, q, contexts, images=None: "学习率 0.001")

    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "学习率多少"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] and body["documents"][0]["text"] == "第三组学习率 0.001"
    assert body["documents"][0]["title"] == "实验报告"
    assert body["documents"][0]["document_id"] == doc.id
    assert body["sources"] == []


def test_query_summarize_intent(tmp_path, monkeypatch):
    from app.core import summarize
    from app.memory import chunks, documents

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", status="done", title="实验报告")
    session.flush()
    c = documents.insert_chunk(session, document_id=doc.id, seq=0,
                               text="第三组学习率 0.001", parent_text="第三组实验设置全文")
    session.flush()
    chunks.index_chunk(session, c.id, [0.1] * 1024, "第三组学习率 0.001")
    session.commit()

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(summarize, "summarize_with_coverage",
                        lambda session, doc_id, *, client, model, max_chars=8000: summarize.SummaryOutcome("这是全文总结", "ok", {"complete": True}))

    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "总结这篇论文"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "这是全文总结"
    assert body["documents"] and body["documents"][0]["text"] == "第三组学习率 0.001"
    assert body["documents"][0]["document_id"] == doc.id
    assert body["sources"] == []


def _seed_doc_with_image(tmp_path, allow_cloud=1, with_chunk=True):
    """预置一个文档 + 可选 chunk 索引 + 落盘的图片行，返回 (session, doc_id)。"""
    from app.memory import chunks, documents
    from app.models.db import Image

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    doc = documents.insert_document(
        session, kind="word", status="done", title="实验报告", allow_cloud=allow_cloud
    )
    session.flush()
    if with_chunk:
        c = documents.insert_chunk(session, document_id=doc.id, seq=0,
                                   text="第三组学习率 0.001", parent_text="第三组实验设置全文")
        session.flush()
        chunks.index_chunk(session, c.id, [0.1] * 1024, "第三组学习率 0.001")
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    png = img_dir / "img1.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfakepngdata")
    session.add(Image(id="img1", document_id=doc.id, file_path="images/img1.png"))
    session.commit()
    return session, doc.id


def test_query_plain_question_triggers_no_vlm(tmp_path, monkeypatch):
    """普通问题（非图片类）不再无条件看图——回归「每次提问烧 1~3 次千问」。"""
    from app.core import cloud
    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1)
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda c, m, blob, q: calls.append(q) or "图片描述X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")
    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="qwen3-vl-plus", rerank_engine="off")
    resp = answer.answer(session, "学习率多少", settings)
    assert calls == []  # 0 次 VLM
    assert resp.answer == "普通答案"


def test_query_falls_back_to_images_when_insufficient(tmp_path, monkeypatch):
    """合成输出 [信息不足] 且文档有图 → 看图重答（二次合成带 images）。"""
    from app.core import cloud
    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1)
    described = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda c, m, blob, q: described.append(q) or "图片描述X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    answers = iter(["[信息不足]", "带图的答案"])
    synth_calls = []
    def fake_synth(client, model, q, contexts, images=None):
        synth_calls.append(images)
        return next(answers)
    monkeypatch.setattr(answer, "synthesize", fake_synth)
    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="qwen3-vl-plus", rerank_engine="off")
    resp = answer.answer(session, "准确率是多少", settings)
    assert len(described) == 1
    assert len(synth_calls) == 2 and synth_calls[1]  # 第二轮带 images
    assert synth_calls[0] is None  # 首轮不带图片
    assert resp.answer == "带图的答案"


def test_query_insufficient_without_images_returns_readable_answer(tmp_path, monkeypatch):
    """[信息不足] 但无图可看 → 不暴露内部标记，返回可读文案。"""
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=0)  # 未允许上云 → 兜底不可用
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda c, m, blob, q: calls.append(q) or "X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "[信息不足]")
    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="qwen3-vl-plus", rerank_engine="off")
    resp = answer.answer(session, "准确率是多少", settings)
    assert calls == []  # 门控拦住的是 VLM 调用本身，而非 VLM 异常被吞
    assert resp.answer == "根据已有内容无法回答该问题。"


def test_query_explained_insufficient_keeps_explanation(tmp_path, monkeypatch):
    """legacy 路径同口径：「解释 + 标记」保留解释、剥掉标记（整段就是标记才换兜底文案）。"""
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=0)
    monkeypatch.setattr(cloud, "describe_image",
                        lambda c, m, blob, q: "X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None:
                        "材料里只有一条「存了体检报告」，没有写存放位置。\n\n[信息不足]")
    settings = Settings(qa_engine="legacy", data_dir=tmp_path,
                        vlm_model="qwen3-vl-plus", rerank_engine="off")
    resp = answer.answer(session, "体检报告在哪", settings)
    assert "[信息不足]" not in resp.answer
    assert "存了体检报告" in resp.answer


def test_query_skips_images_when_allow_cloud_off(tmp_path, monkeypatch):
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=0)
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda client, model, image_bytes, question: calls.append(question) or "X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    # 合成出 [信息不足] 才会真正走到兜底取图的判定处，否则测不到门控
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "[信息不足]")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="qwen-vl-plus", rerank_engine="off")
    answer.answer(session, "学习率多少", settings)

    assert calls == []


def test_query_skips_images_when_vlm_off(tmp_path, monkeypatch):
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1)
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda client, model, image_bytes, question: calls.append(question) or "X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    # 同上：先进入兜底判定（[信息不足]），再证明 vlm_model 未配置时 VLM 不被调用
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "[信息不足]")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="", rerank_engine="off")
    answer.answer(session, "学习率多少", settings)

    assert calls == []


def test_query_image_keyword_forces_describe_without_chunks(tmp_path, monkeypatch):
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1, with_chunk=False)
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda client, model, image_bytes, question: calls.append(question) or "图片描述X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    captured = {}
    def fake_synthesize(client, model, q, contexts, images=None):
        captured["contexts"] = contexts
        captured["images"] = images
        return "|".join(contexts + (images or []))
    monkeypatch.setattr(answer, "synthesize", fake_synthesize)

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="qwen-vl-plus", rerank_engine="off")
    resp = answer.answer(session, "图片里有什么", settings)

    assert len(calls) == 1 and "图片里有什么" in calls[0]
    assert captured["contexts"] == []  # 图片描述不再混进文本上下文，而是单独成段传给合成
    assert captured["images"] == ["图片描述（《实验报告》正文中）：图片描述X"]
    assert "图片描述X" in resp.answer


def test_query_image_intent_without_keywords_triggers_describe(tmp_path, monkeypatch):
    """LLM 路由判为图片问题（问题里没有「图」字）→ 同样触发 VLM 看图。"""
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1, with_chunk=False)
    calls = []
    monkeypatch.setattr(cloud, "describe_image",
                        lambda client, model, image_bytes, question: calls.append(question) or "图片描述X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize", lambda client, model, q, contexts, images=None: "ok")
    monkeypatch.setattr(router, "classify", lambda q, client, model: router.QueryKind.image_query)

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, vlm_model="qwen3-vl-plus", rerank_engine="off")
    answer.answer(session, "这两种标注分别代表什么含义", settings)

    assert len(calls) == 1


def _capture_vlm_client(monkeypatch):
    from app.core import cloud

    captured = {}
    def fake_openai(*args, **kwargs):
        from types import SimpleNamespace
        captured["kwargs"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None)))
    monkeypatch.setattr(cloud, "OpenAI", fake_openai)
    return captured


def test_vlm_key_not_leaked_to_non_qwen_llm(tmp_path, monkeypatch):
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1)
    monkeypatch.setattr(cloud, "describe_image",
                        lambda client, model, image_bytes, question: "X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    # 普通问题不再无条件看图；用 [信息不足] 触发兜底路径，让 VLM 客户端构造走到（本测试关注的是 key 选择）
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "[信息不足]")
    captured = _capture_vlm_client(monkeypatch)

    settings = Settings(qa_engine="legacy",
        data_dir=tmp_path, vlm_model="qwen-vl-plus", rerank_engine="off",
        llm_provider="deepseek", llm_api_key="test-key-deepseek", vlm_api_key="",
    )
    answer.answer(session, "学习率多少", settings)

    assert captured["kwargs"]["api_key"] == "dummy"


def test_vlm_key_falls_back_to_llm_key_when_provider_qwen(tmp_path, monkeypatch):
    from app.core import cloud

    session, _ = _seed_doc_with_image(tmp_path, allow_cloud=1)
    monkeypatch.setattr(cloud, "describe_image",
                        lambda client, model, image_bytes, question: "X")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    # 同上：走兜底路径触发 VLM 客户端构造，验证 provider=qwen 时的 key 回退
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "[信息不足]")
    captured = _capture_vlm_client(monkeypatch)

    settings = Settings(qa_engine="legacy",
        data_dir=tmp_path, vlm_model="qwen-vl-plus", rerank_engine="off",
        llm_provider="qwen", llm_api_key="test-key-qwen", vlm_api_key="",
    )
    answer.answer(session, "学习率多少", settings)

    assert captured["kwargs"]["api_key"] == "test-key-qwen"


def test_query_rewrite_expands_long_question(tmp_path, monkeypatch):
    """auto 模式：长问题先扩展，只有变体能命中的记忆也被检索到（无向量索引，隔离向量路）。"""
    from app.core import rewrite

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    session.add(
        Fact(id="f1", type="other", summary="学习率设置 0.001",
             valid_from=datetime.now(timezone.utc))
    )
    session.flush()
    fts.index_fact(session, "f1", "学习率设置 0.001")
    session.commit()

    calls = []
    monkeypatch.setattr(rewrite, "expand_queries",
                        lambda client, model, q: calls.append(q) or ["学习率"])
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(answer, "synthesize", lambda client, model, q, contexts, images=None: "答案")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, query_rewrite="auto", rerank_engine="off")
    resp = answer.answer(session, "帮我看看之前那个实验里参数怎么设置的", settings)

    assert calls == ["帮我看看之前那个实验里参数怎么设置的"]
    assert [s.fact_id for s in resp.sources] == ["f1"]


def test_query_rewrite_skipped_for_short_question(tmp_path, monkeypatch):
    from app.core import rewrite

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    session = Session(engine)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    calls = []
    monkeypatch.setattr(rewrite, "expand_queries", lambda *a, **k: calls.append(a) or ["x"])

    answer.answer(session, "查重报告", Settings(qa_engine="legacy",data_dir=tmp_path, query_rewrite="auto"))
    assert calls == []


def test_query_rewrite_skipped_when_off(tmp_path, monkeypatch):
    from app.core import rewrite

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    session = Session(engine)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    calls = []
    monkeypatch.setattr(rewrite, "expand_queries", lambda *a, **k: calls.append(a) or ["x"])

    long_q = "帮我看看之前那个实验里参数怎么设置的"
    answer.answer(session, long_q, Settings(qa_engine="legacy",data_dir=tmp_path, query_rewrite="off"))
    assert calls == []


def _seed_two_docs_for_summarize(tmp_path):
    """论文（身份有「机器人」、chunk 命中少）vs 任务书（chunk 命中多、身份没有「机器人」）。"""
    from app.memory import chunks, documents, store

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)

    paper = documents.insert_document(
        session, kind="word", status="done",
        title="封闭性条件下AI不存在技术失控风险", file_path="files/paper.docx",
    )
    session.flush()
    raw = store.insert_raw(session, kind="document", content="", file_path="files/paper.docx")
    session.flush()
    session.add(Fact(id="f1", type="other", summary="机器人课程的大作业是一篇论文",
                     valid_from=datetime.now(timezone.utc), raw_id=raw.id))
    for i, t in enumerate(["本文是一篇论文", "技术失控风险分析"]):
        c = documents.insert_chunk(session, document_id=paper.id, seq=i, text=t, parent_text=t)
        session.flush()
        chunks.index_chunk(session, c.id, [0.1] * 512, t)

    book = documents.insert_document(
        session, kind="word", status="done",
        title="模式识别课程设计任务书", file_path="files/book.pdf",
    )
    session.flush()
    for i, t in enumerate(["报告要有分析总结", "论文格式要求", "课程设计任务", "总结与参考文献",
                           "课程实验列表", "论文撰写规范"]):
        c = documents.insert_chunk(session, document_id=book.id, seq=i, text=t, parent_text=t)
        session.flush()
        chunks.index_chunk(session, c.id, [0.1] * 512, t)
    session.commit()
    return session, paper, book


def test_summarize_target_prefers_fact_identity_over_chunk_count(tmp_path, monkeypatch):
    """总结选文档：身份匹配（标题+关联记忆）胜过 chunk 命中数——「机器人课程的论文」回归。"""
    from app.core import summarize

    session, paper, book = _seed_two_docs_for_summarize(tmp_path)

    def fake_embed(text, model=None):
        v = [0.0] * 512
        v[0 if "机器人" in text else 1] = 1.0
        return v

    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(rewrite, "expand_queries", lambda client, model, q: ["机器人 论文 总结"])
    picked = []
    monkeypatch.setattr(summarize, "summarize_with_coverage",
                        lambda session, doc_id, *, client, model, max_chars=8000: picked.append(doc_id) or summarize.SummaryOutcome("总结X", "ok", {"complete": True}))

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, query_rewrite="auto", rerank_engine="off")
    resp = answer.answer(session, "总结一下我上传的那个机器人课程的论文", settings)

    assert picked == [paper.id]  # 尽管 book 的 chunk 命中更多，身份匹配应选中 paper
    assert resp.answer == "总结X"


def test_summarize_target_falls_back_to_chunk_count(tmp_path, monkeypatch):
    """无标题/无关联记忆时，退回「chunk 命中最多的文档」原逻辑。"""
    from app.core import summarize
    from app.memory import chunks, documents

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    session = Session(engine)
    doc_a = documents.insert_document(session, kind="word", status="done")
    doc_b = documents.insert_document(session, kind="word", status="done")
    session.flush()
    for doc, texts in ((doc_a, ["学习率"]), (doc_b, ["总结报告", "论文要求", "总结要点"])):
        for i, t in enumerate(texts):
            c = documents.insert_chunk(session, document_id=doc.id, seq=i, text=t, parent_text=t)
            session.flush()
            chunks.index_chunk(session, c.id, [0.1] * 512, t)
    session.commit()

    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    monkeypatch.setattr(rewrite, "expand_queries", lambda client, model, q: ["总结 论文"])
    picked = []
    monkeypatch.setattr(summarize, "summarize_with_coverage",
                        lambda session, doc_id, *, client, model, max_chars=8000: picked.append(doc_id) or summarize.SummaryOutcome("总结X", "ok", {"complete": True}))

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, query_rewrite="auto", rerank_engine="off")
    answer.answer(session, "总结一下我上传的论文内容", settings)

    assert picked == [doc_b.id]


def _seed_doc_with_table(tmp_path, with_table=True, suspect=False):
    """预置：文档 + chunk（可命中「学分」）+ 可选 doc_tables 行；返回 (session, doc_id)。"""
    from app.memory import chunks, documents

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    session = Session(engine)
    doc = documents.insert_document(session, kind="word", status="done", title="成绩总表")
    session.flush()
    c = documents.insert_chunk(session, document_id=doc.id, seq=0,
                               text="课程学分列表", parent_text="课程学分列表全文")
    session.flush()
    chunks.index_chunk(session, c.id, [0.1] * 1024, "课程学分列表")
    if with_table:
        documents.insert_table(
            session, document_id=doc.id, suspect=suspect,
            headers=json.dumps(["课程", "学分"], ensure_ascii=False),
            rows=json.dumps([["体育", "0.5"], ["高数", "3.0"], ["就业指导", "0.5"]],
                            ensure_ascii=False),
        )
    session.commit()
    return session, doc.id


def _no_synth(*a, **k):
    raise AssertionError("不应走普通问答")


def test_query_analysis_filters_table(tmp_path, monkeypatch):
    """analysis_query：命中表格文档 → 抽条件（打桩）→ 真实 apply_filter → 组答（打桩）。"""
    session, doc_id = _seed_doc_with_table(tmp_path)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(stats, "extract_filter",
                        lambda c, m, q, tables: stats.Filter(tables[0].id, "学分", "<", "1"))
    captured = {}
    def fake_compose(client, model, q, headers, outcome):
        captured["outcome"] = outcome
        return "共 2 门"
    monkeypatch.setattr(stats, "compose_answer", fake_compose)
    monkeypatch.setattr(answer, "synthesize", _no_synth)

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert resp.answer == "共 2 门"
    assert [r[0] for r in captured["outcome"].rows] == ["体育", "就业指导"]  # 真实 apply_filter 执行
    assert resp.sources == []
    assert [d.document_id for d in resp.documents] == [doc_id]  # 溯源 = 命中 chunk 的文档


def test_query_analysis_notes_suspect_table(tmp_path, monkeypatch):
    """作答所用表格被标 suspect（解析可能不完整）→ 答案末尾如实追加提示。"""
    session, _ = _seed_doc_with_table(tmp_path, suspect=True)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(stats, "extract_filter",
                        lambda c, m, q, tables: stats.Filter(tables[0].id, "学分", "<", "1"))
    monkeypatch.setattr(stats, "compose_answer", lambda *a, **k: "共 2 门")
    monkeypatch.setattr(answer, "synthesize", _no_synth)

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert resp.answer == "共 2 门（该表格解析可能不完整）"


def test_query_analysis_falls_back_without_tables(tmp_path, monkeypatch):
    """analysis_query 但命中文档无 doc_tables → 不报错，降级普通问答并前置如实说明。"""
    session, _ = _seed_doc_with_table(tmp_path, with_table=False)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert resp.answer == "（未找到可统计的表格，以下为基于原文的回答）\n普通答案"


def test_query_analysis_falls_back_when_extract_fails(tmp_path, monkeypatch):
    """有表但条件抽取失败（None）→ 降级普通问答，并如实说明降级原因。"""
    session, _ = _seed_doc_with_table(tmp_path)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(stats, "extract_filter", lambda c, m, q, tables: None)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "这些课学分怎么算", settings)

    assert resp.answer == "（未能确定统计条件，以下为基于原文的回答）\n普通答案"


def test_query_analysis_degrades_when_numeric_value_unparseable(tmp_path, monkeypatch):
    """终审 I2：数值 op 但比较值不可解析（「小于1分」）→ 条件根本没被执行：如实降级普通问答
    （走真实 extract_filter 校验），不得答「没有符合条件的记录。」（那是把「没执行」说成「没有」）。"""
    from sqlalchemy import select

    from app.models.db import DocTable

    session, _ = _seed_doc_with_table(tmp_path)
    table_id = session.scalars(select(DocTable.id)).first()
    bad_json = (
        f'{{"table_id": "{table_id}", "column": "学分", "op": "<", "value": "小于1分", "aggregate": "none"}}'
    )

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    message = type("M", (), {"content": bad_json})()
                    return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(llm, "build_client", lambda settings: FakeClient())
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分小于1分的课有哪些", settings)

    assert resp.answer == "（未能确定统计条件，以下为基于原文的回答）\n普通答案"


def test_query_analysis_falls_back_when_compose_returns_empty(tmp_path, monkeypatch):
    """I3：组答返回空串视为失败 → 降级普通问答（而非返回空答案），并如实说明降级原因。"""
    session, _ = _seed_doc_with_table(tmp_path)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(stats, "extract_filter",
                        lambda c, m, q, tables: stats.Filter(tables[0].id, "学分", "<", "1"))
    monkeypatch.setattr(stats, "compose_answer", lambda *a, **k: "")
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert resp.answer == "（未能组织统计结果，以下为基于原文的回答）\n普通答案"


def test_query_analysis_falls_back_on_corrupt_table_json(tmp_path, monkeypatch):
    """I3：doc_tables JSON 损坏 → 整段 try 兜底，不抛 500，降级普通问答并如实说明。"""
    from app.memory import documents

    session, doc_id = _seed_doc_with_table(tmp_path, with_table=False)
    documents.insert_table(session, document_id=doc_id, headers="not-json", rows="not-json")
    session.commit()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert resp.answer == "（统计路径异常，以下为基于原文的回答）\n普通答案"


def test_query_analysis_success_has_no_prefix(tmp_path, monkeypatch):
    """正常出统计结果 → 不加降级说明（前缀只属于降级路径）。"""
    session, _ = _seed_doc_with_table(tmp_path)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(stats, "extract_filter",
                        lambda c, m, q, tables: stats.Filter(tables[0].id, "学分", "<", "1"))
    monkeypatch.setattr(stats, "compose_answer", lambda *a, **k: "共 2 门")
    monkeypatch.setattr(answer, "synthesize", _no_synth)

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert "基于原文的回答" not in resp.answer


def test_query_analysis_fallback_reuses_single_retrieval(tmp_path, monkeypatch):
    """M4：文档检索只跑一次，analysis 降级与普通路径复用同一批 chunk_ids。"""
    from app.core import retrieval

    session, _ = _seed_doc_with_table(tmp_path, with_table=False)
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 1024)
    monkeypatch.setattr(router, "classify", lambda q, c, m: router.QueryKind.analysis_query)
    monkeypatch.setattr(answer, "synthesize",
                        lambda client, model, q, contexts, images=None: "普通答案")
    calls = []
    original = retrieval.search_documents_multi
    monkeypatch.setattr(retrieval, "search_documents_multi",
                        lambda *a, **k: calls.append(a) or original(*a, **k))

    settings = Settings(qa_engine="legacy",data_dir=tmp_path, rerank_engine="off")
    resp = answer.answer(session, "学分少于1的课程", settings)

    assert resp.answer == "（未找到可统计的表格，以下为基于原文的回答）\n普通答案"
    assert len(calls) == 1  # 修复前：分析路径 + 降级普通路径各检索一次 = 2
