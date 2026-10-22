"""存量材料回填（F1）：M3 前置——不跑回填，存量档案在材料层不可见。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core import llm, retrieval
from app.memory import documents, materials, store
from app.models.db import Material, init_db
from app.models.schemas import ExtractedFact
from scripts.materials_backfill import backfill

MODEL = "BAAI/bge-small-zh-v1.5"


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def test_backfill_covers_records_and_documents_idempotently(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    s = _session(tmp_path)

    # 有 fact 的记录 → 回填；raw_only（无 fact）→ 不回填（与记录接口语义一致）
    raw1 = store.insert_raw(s, kind="text", content="身份证放在书房")
    s.flush()
    store.insert_fact(
        s,
        ExtractedFact(type="item_location", subject_key="身份证:位置",
                      summary="身份证放书房", confidence=0.9),
        raw1.id,
    )
    raw2 = store.insert_raw(s, kind="text", content="随手一句话")
    s.flush()

    # 未删文档 → 回填；已软删文档 → 不回填
    doc1 = documents.insert_document(s, kind="word", status="done", title="论文", parsed_text="正文")
    documents.insert_document(s, kind="word", status="done", title="已删",
                              parsed_text="x", deleted_at=store.utcnow())
    s.commit()

    n_records, n_docs = backfill(s, MODEL)
    assert (n_records, n_docs) == (1, 1)

    rows = s.query(Material).all()
    assert len(rows) == 2
    by_source = {(m.source_type, m.source_id) for m in rows}
    assert ("record", raw1.id) in by_source
    assert ("document", doc1.id) in by_source
    assert all(m.source_id != raw2.id for m in rows)

    # 幂等：重复跑不新增行、计数不变
    n_records2, n_docs2 = backfill(s, MODEL)
    assert (n_records2, n_docs2) == (1, 1)
    assert s.query(Material).count() == 2

    # 回填后的记录材料可被材料层检索到（FTS 路，「身份证」3 字可命中 trigram）
    hits = materials.fts_search(s, "身份证")
    assert any(m.source_id == raw1.id for m in rows if m.id in hits)
    # 第一级检索可用（向量路，embed 已打桩）
    ids = retrieval.search_materials_multi(s, ["身份证"], top_k=5, model=MODEL)
    assert set(ids) == {m.id for m in rows}
