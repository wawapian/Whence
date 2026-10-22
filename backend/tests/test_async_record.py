"""记录异步化：consolidate_record 单元行为（跳过已删 / 兜底 / 正文回填）+ 启动重投。"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.api.memories import consolidate_record
from app.config import Settings
from app.core import extraction, llm, queue
from app.core.recovery import requeue_pending
from app.memory import documents, store
from app.models.db import Document, Fact, Material, RawEvidence, init_db
from app.models.schemas import ExtractedFact

MODEL = "BAAI/bge-small-zh-v1.5"


def _settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, embedding_model=MODEL)


def _mk_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return engine


def _patch(monkeypatch, extract=None):
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    monkeypatch.setattr(
        extraction, "extract",
        extract or (lambda c, m, t: [ExtractedFact(type="other", summary="摘要", confidence=0.9)]),
    )


def test_consolidate_skips_deleted_raw(tmp_path, monkeypatch):
    _patch(monkeypatch)
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="text", content="身份证放在书房")
        raw.consolidated = 0
        raw.deleted_at = store.utcnow()
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        assert s.query(Fact).count() == 0  # 已删记录不整理（防「删了还冒出来」）


def test_consolidate_falls_back_to_pending_on_extract_failure(tmp_path, monkeypatch):
    def _boom(c, m, t):
        raise RuntimeError("llm down")

    _patch(monkeypatch, extract=_boom)
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="text", content="今天买了鸡蛋")
        raw.consolidated = 0
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        fact = s.query(Fact).one()
        assert fact.state == "pending"
        assert fact.summary == "今天买了鸡蛋"[:50]
        assert s.get(RawEvidence, raw_id).consolidated == 1
        assert s.query(Material).count() == 1  # 兜底事实同样带材料（时间线与材料层都不丢）


def test_consolidate_persists_and_indexes_every_fact(tmp_path, monkeypatch):
    """发现#2-① 回归：一条记录多个陈述 → 全部 fact 落库（状态类不遗漏）且逐条建 FTS+向量索引。"""
    facts = [
        ExtractedFact(type="consumption", summary="五月三号买了一个耳机", amount=199.0, confidence=0.9),
        ExtractedFact(type="item_location", subject_key="耳机:位置",
                      summary="耳机放到书柜里了", confidence=0.9),
        ExtractedFact(type="item_location", subject_key="耳机盒子:位置",
                      summary="耳机盒子放到进门左边立柜里了", confidence=0.9),
    ]
    _patch(monkeypatch, extract=lambda c, m, t: facts)
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(
            s, kind="text",
            content="我五月三号买了一个耳机，耳机放到书柜里了，耳机盒子放到进门左边立柜里了",
        )
        raw.consolidated = 0
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        rows = s.query(Fact).all()
        assert {f.summary for f in rows} == {f.summary for f in facts}  # 三个陈述一个不落
        assert all(f.state == "confirmed" for f in rows)
        for f in rows:
            indexed = s.execute(
                text("SELECT content FROM memory_fts WHERE fact_id = :f"), {"f": f.id}
            ).scalar()
            assert indexed and f.summary in indexed  # 逐条建 FTS 索引
        assert s.execute(
            text("SELECT count(*) FROM vec_map WHERE fact_id IN "
                 "(SELECT id FROM facts WHERE raw_id = :r)"),
            {"r": raw_id},
        ).scalar() == 3  # 逐条建向量索引
        assert s.query(Material).one().title == facts[0].summary  # 材料 title = 首条 fact 摘要


def test_consolidate_same_subject_key_later_fact_supersedes_earlier(tmp_path, monkeypatch):
    """同记录内同 subject_key 两条 → 后说的覆盖先说的（前条 valid_to 被置、后条有效）。"""
    earlier = ExtractedFact(type="item_location", subject_key="耳机:位置",
                            summary="耳机放到书柜里了", confidence=0.9)
    later = ExtractedFact(type="item_location", subject_key="耳机:位置",
                          summary="耳机改放书桌上了", confidence=0.9)
    _patch(monkeypatch, extract=lambda c, m, t: [earlier, later])
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="text", content="耳机放书柜了，哦不对，改放书桌上了")
        raw.consolidated = 0
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        rows = s.query(Fact).filter(Fact.subject_key == "耳机:位置").all()
        assert len(rows) == 2  # 旧值留档，不删
        old = next(f for f in rows if f.summary == earlier.summary)
        new = next(f for f in rows if f.summary == later.summary)
        assert old.valid_to is not None and old.superseded_by == new.id
        assert new.valid_to is None and new.superseded_by is None


def test_consolidate_single_statement_behavior_unchanged(tmp_path, monkeypatch):
    """单陈述文本：与修复前单事实行为一致——1 条 confirmed、索引同源正文、材料 title=摘要。"""
    _patch(monkeypatch, extract=lambda c, m, t: [
        ExtractedFact(type="item_location", subject_key="身份证:位置",
                      summary="身份证放书房", confidence=0.9),
    ])
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="text", content="身份证放在书房左边第二个抽屉")
        raw.consolidated = 0
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        f = s.query(Fact).one()
        assert (f.state, f.subject_key, f.summary) == ("confirmed", "身份证:位置", "身份证放书房")
        assert s.execute(
            text("SELECT content FROM memory_fts WHERE fact_id = :f"), {"f": f.id}
        ).scalar() == "身份证放书房 身份证放在书房左边第二个抽屉"
        material = s.query(Material).one()
        assert (material.title, material.summary) == ("身份证放书房", "身份证放在书房左边第二个抽屉")
        assert s.get(RawEvidence, raw_id).consolidated == 1


def test_consolidate_empty_fact_list_falls_back_to_pending(tmp_path, monkeypatch):
    """抽取成功但零事实（空列表）→ 沿用 pending 兜底单事实（时间线不丢记录）。"""
    _patch(monkeypatch, extract=lambda c, m, t: [])
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="text", content="今天买了鸡蛋")
        raw.consolidated = 0
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        fact = s.query(Fact).one()
        assert fact.state == "pending"
        assert fact.summary == "今天买了鸡蛋"[:50]
        assert s.get(RawEvidence, raw_id).consolidated == 1
        assert s.query(Material).count() == 1


def test_consolidate_backfills_raw_content_and_desc_from_attachment(tmp_path, monkeypatch):
    from app.core import parsing

    seen = {}

    def fake_extract(c, m, t):
        seen["desc"] = t
        return [ExtractedFact(type="other", summary="查重报告", confidence=0.9)]

    _patch(monkeypatch, extract=fake_extract)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="document", file_path="files/a.docx")  # 无原话
        raw.consolidated = 0
        s.flush()
        documents.insert_document(
            s, kind="document", title="a.docx", file_path="files/a.docx",
            status="done", raw_id=raw.id,
        )
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        assert s.get(RawEvidence, raw_id).content == "查重报告 相似度 15%"  # 原话优先：只填空
        assert s.query(Fact).count() == 1
    assert seen["desc"] == "上传了文档：a.docx"  # 描述与旧同步版逐字一致


def test_consolidate_raw_only_marks_done_without_fact(tmp_path, monkeypatch):
    _patch(monkeypatch)
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        raw = store.insert_raw(s, kind="photo")  # 纯照片无文字：解析也出不了正文
        raw.consolidated = 0
        s.commit()
        raw_id = raw.id

    consolidate_record(_settings(tmp_path), raw_id)

    with Session(engine) as s:
        assert s.query(Fact).count() == 0
        assert s.get(RawEvidence, raw_id).consolidated == 1


def test_requeue_pending_resubmits_stuck_work(tmp_path, monkeypatch):
    """启动重投：consolidated=0 未删的 raw → record 车道；processing 未删文档 → doc 车道。"""
    engine = _mk_engine(tmp_path)
    with Session(engine) as s:
        pending = store.insert_raw(s, kind="text", content="待整理")
        pending.consolidated = 0
        done = store.insert_raw(s, kind="text", content="已整理")
        done.consolidated = 1
        legacy = store.insert_raw(s, kind="text", content="存量旧数据")
        legacy.consolidated = None
        deleted = store.insert_raw(s, kind="text", content="已删")
        deleted.consolidated = 0
        deleted.deleted_at = store.utcnow()
        documents.insert_document(
            s, kind="pdf", title="滞留", file_path="files/x.pdf", status="processing"
        )
        documents.insert_document(
            s, kind="pdf", title="已完成", file_path="files/y.pdf", status="done"
        )
        s.commit()

    submitted = []

    def _record(fn, *args, **kwargs):
        submitted.append((fn.__name__, kwargs.get("lane", "doc")))
        return "test"

    monkeypatch.setattr(queue, "submit", _record)

    with Session(engine) as s:
        n_records, n_docs = requeue_pending(s, _settings(tmp_path))

    assert (n_records, n_docs) == (1, 1)
    assert ("consolidate_record", "record") in submitted
    assert ("process_document", "doc") in submitted
