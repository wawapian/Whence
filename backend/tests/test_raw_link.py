"""documents.raw_id：迁移（列缺失才补列 + 同块回填，同 file_path 取最早 raw）与回收站「组」联动。

组语义：一次上传 = 1 条记录 + N 份附件——`raw_id` 认组、`file_path` 兜底（历史数据）；
软删/彻底删除按组联动，恢复保持逐条。
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core import trash
from app.memory import chunks as chunk_store, documents, fts, store, vector
from app.models.db import Chunk, Document, Fact, RawEvidence, init_db
from app.models.schemas import ExtractedFact


def test_init_db_adds_raw_id_and_backfills_by_file_path(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "title TEXT, file_path TEXT, status TEXT NOT NULL, allow_cloud INTEGER NOT NULL DEFAULT 0, "
            "parsed_text TEXT, summary TEXT, created_at TEXT NOT NULL)"
        ))
        conn.execute(text(
            "CREATE TABLE raw_evidence (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "content TEXT, file_path TEXT, created_at TEXT NOT NULL)"
        ))
        # 一对同 file_path 的 raw + doc（同一次上传）
        conn.execute(text(
            "INSERT INTO documents (id, user_id, kind, title, file_path, status, created_at) "
            "VALUES ('doc1', 'me', 'pdf', '成绩总表', 'files/a.pdf', 'done', '2026-01-01 00:00:00')"
        ))
        conn.execute(text(
            "INSERT INTO raw_evidence (id, user_id, kind, file_path, created_at) "
            "VALUES ('raw1', 'me', 'document', 'files/a.pdf', '2026-01-01 00:00:01')"
        ))
        # 一份无匹配 raw 的独立 doc
        conn.execute(text(
            "INSERT INTO documents (id, user_id, kind, title, file_path, status, created_at) "
            "VALUES ('doc2', 'me', 'pdf', '没有配对的文档', 'files/orphan.pdf', 'done', '2026-01-01 00:00:00')"
        ))
    engine.dispose()
    init_db(engine, dim=512)
    with engine.begin() as conn:
        doc_cols = {r[1] for r in conn.execute(text("PRAGMA table_info(documents)"))}
        got = dict(conn.execute(text("SELECT id, raw_id FROM documents")).all())
    assert "raw_id" in doc_cols
    assert got["doc1"] == "raw1"   # 回填：同 file_path 配对
    assert got["doc2"] is None     # 无匹配 raw 的独立 doc 保持 NULL


def test_add_raw_id_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "title TEXT, file_path TEXT, status TEXT NOT NULL, allow_cloud INTEGER NOT NULL DEFAULT 0, "
            "parsed_text TEXT, summary TEXT, created_at TEXT NOT NULL)"
        ))
        conn.execute(text(
            "CREATE TABLE raw_evidence (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "content TEXT, file_path TEXT, created_at TEXT NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO documents (id, user_id, kind, title, file_path, status, created_at) "
            "VALUES ('doc1', 'me', 'pdf', '成绩总表', 'files/a.pdf', 'done', '2026-01-01 00:00:00')"
        ))
        conn.execute(text(
            "INSERT INTO raw_evidence (id, user_id, kind, file_path, created_at) "
            "VALUES ('raw1', 'me', 'document', 'files/a.pdf', '2026-01-01 00:00:01')"
        ))
    engine.dispose()
    init_db(engine, dim=512)
    with engine.begin() as conn:
        first = conn.execute(text("SELECT raw_id FROM documents WHERE id = 'doc1'")).scalar()
    assert first == "raw1"
    init_db(engine, dim=512)  # 幂等：连跑两遍不抛
    with engine.begin() as conn:
        second = conn.execute(text("SELECT raw_id FROM documents WHERE id = 'doc1'")).scalar()
    assert second == "raw1"  # 第二次走「列已存在」分支，不重复回填，raw_id 保持原值


# ── 回收站「组」联动：一次上传 = 1 条记录 + N 份附件，删任一侧全组进站 ──

def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def _group(s, tmp_path, n=3):
    """一次上传的组：1 条记录 + n 份附件（`raw_id` 均指向该记录、各自独立 `file_path`）；文件落盘。"""
    files_dir = tmp_path / "files"
    files_dir.mkdir(exist_ok=True)
    raw = store.insert_raw(s, kind="document", content=f"上传了 {n} 个文件", file_path="files/a1.pdf")
    s.flush()
    docs = []
    for i in range(1, n + 1):
        (files_dir / f"a{i}.pdf").write_bytes(b"x")
        docs.append(documents.insert_document(
            s, kind="pdf", title=f"附件{i}", file_path=f"files/a{i}.pdf",
            raw_id=raw.id, status="done",
        ))
    s.commit()
    return raw, docs


def test_soft_delete_any_side_trashes_whole_group(tmp_path):
    """仅删其中 1 份附件 → 记录与 3 份附件全部进站；live 的同一时间戳，已在站内的保留原时间戳。"""
    s = _session(tmp_path)
    raw, docs = _group(s, tmp_path)
    # 混合情形：docs[2] 已单独在站内（旧时间戳），联动时只处理 live 的
    s.execute(text("UPDATE documents SET deleted_at = '2026-01-01 00:00:00' WHERE id = :id"),
              {"id": docs[2].id})
    s.commit()

    assert trash.soft_delete_document(s, docs[1].id) is True
    s.commit()
    s.expire_all()
    assert s.get(RawEvidence, raw.id).deleted_at is not None
    assert all(s.get(Document, d.id).deleted_at is not None for d in docs)  # 3 份附件全进站
    live_stamps = {
        s.get(RawEvidence, raw.id).deleted_at,
        s.get(Document, docs[0].id).deleted_at,
        s.get(Document, docs[1].id).deleted_at,
    }
    assert None not in live_stamps and len(live_stamps) == 1  # 组内 live 对象同一时间戳
    assert str(s.get(Document, docs[2].id).deleted_at).startswith("2026-01-01")  # 已在站内的不动


def test_soft_delete_record_trashes_all_attachments(tmp_path):
    """删记录 → 3 份附件全进站。"""
    s = _session(tmp_path)
    raw, docs = _group(s, tmp_path)

    assert trash.soft_delete_record(s, raw.id) is True
    s.commit()
    s.expire_all()
    assert s.get(RawEvidence, raw.id).deleted_at is not None
    assert all(s.get(Document, d.id).deleted_at is not None for d in docs)


def test_purge_group_removes_everything_incl_material_indexes(tmp_path):
    """组全在站内 → purge 任一侧 → raw/3 docs/派生行与索引/各文件全清（复用现有行清逻辑）。"""
    s = _session(tmp_path)
    raw, docs = _group(s, tmp_path)
    for i, doc in enumerate(docs):
        chunk = documents.insert_chunk(
            s, document_id=doc.id, seq=0, text=f"第{i}份正文", parent_text=f"第{i}份正文"
        )
        s.flush()
        chunk_store.index_chunk(s, chunk.id, [0.1] * 512, f"第{i}份正文")
    fact = store.insert_fact(
        s, ExtractedFact(type="other", summary="组的总结", confidence=0.9), raw.id
    )
    s.flush()
    fts.index_fact(s, fact.id, "组的总结")
    vector.index_fact(s, fact.id, [0.2] * 512)
    s.commit()

    assert trash.soft_delete_record(s, raw.id) is True  # 全组进站
    s.commit()

    assert trash.purge_document(s, docs[0].id) is True  # 从附件侧触发联动
    s.commit()
    assert s.get(RawEvidence, raw.id) is None and s.get(Document, docs[0].id) is None
    assert s.query(Document).count() == 0 and s.query(Chunk).count() == 0 and s.query(Fact).count() == 0
    assert s.execute(text("SELECT count(*) FROM chunk_fts")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM chunk_vec")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM memory_fts")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM memory_vec")).scalar() == 0
    assert trash.purge_record(s, raw.id) is False  # 记录已随联动消失
    for i in range(1, 4):
        assert not (tmp_path / "files" / f"a{i}.pdf").exists()


def test_legacy_file_path_fallback_still_links(tmp_path):
    """不带 raw_id、仅同 file_path 的旧 pair：删除/恢复/彻底删除与今天完全一致。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "old.pdf").write_bytes(b"x")
    doc = documents.insert_document(
        s, kind="pdf", title="旧配对", file_path="files/old.pdf", status="done"
    )
    raw = store.insert_raw(s, kind="document", file_path="files/old.pdf")
    s.commit()
    assert s.get(Document, doc.id).raw_id is None

    # 软删联动（兜底键）→ 恢复逐条
    assert trash.soft_delete_document(s, doc.id) is True
    s.commit()
    s.expire_all()
    assert s.get(RawEvidence, raw.id).deleted_at is not None
    assert trash.restore_record(s, raw.id) is True
    assert trash.restore_document(s, doc.id) is True
    s.commit()
    s.expire_all()
    assert s.get(RawEvidence, raw.id).deleted_at is None
    assert s.get(Document, doc.id).deleted_at is None

    # 彻底删除联动：一对都在站内 → 删任一，两条行与文件全清
    assert trash.soft_delete_record(s, raw.id) is True
    s.commit()
    assert trash.purge_record(s, raw.id) is True
    s.commit()
    assert s.query(Document).count() == 0 and s.query(RawEvidence).count() == 0
    assert not (files_dir / "old.pdf").exists()


def test_group_purge_leaves_live_sides_untouched(tmp_path):
    """半边状态：raw 在站内、某份附件活着 → purge 只清站内成员，活附件与其文件不碰。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    raw, docs = _group(s, tmp_path, n=2)
    assert trash.soft_delete_record(s, raw.id) is True    # 全组进站
    assert trash.restore_document(s, docs[0].id) is True  # 只恢复一份 → 半边状态
    s.commit()

    assert trash.purge_record(s, raw.id) is True
    s.commit()
    s.expire_all()
    assert s.get(RawEvidence, raw.id) is None
    assert s.get(Document, docs[0].id) is not None      # 活附件不动
    assert s.get(Document, docs[1].id) is None          # 站内成员随组清理
    assert (files_dir / "a1.pdf").exists()              # 活附件的文件不删
    assert not (files_dir / "a2.pdf").exists()          # 站内成员的文件同清
