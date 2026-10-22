from datetime import timedelta

from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core import llm, trash
from app.core.masking import link_secrets_to_fact, mask_text
from app.memory import documents, fts, materials, store, vector
from app.memory.store import utcnow
from app.models.db import (
    Chunk,
    DocTable,
    Document,
    Fact,
    Image,
    Material,
    RawEvidence,
    Secret,
    init_db,
)
from app.models.schemas import ExtractedFact


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def test_init_db_adds_deleted_at_columns_idempotently(tmp_path):
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
    engine.dispose()
    init_db(engine, dim=512)
    with engine.begin() as conn:
        doc_cols = {r[1] for r in conn.execute(text("PRAGMA table_info(documents)"))}
        raw_cols = {r[1] for r in conn.execute(text("PRAGMA table_info(raw_evidence)"))}
    assert "deleted_at" in doc_cols and "deleted_at" in raw_cols
    init_db(engine, dim=512)  # 幂等


def test_init_db_migration_preserves_existing_rows(tmp_path):
    """存量库补列不动既有行：旧行原样保留，deleted_at 为 NULL。"""
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "title TEXT, file_path TEXT, status TEXT NOT NULL, allow_cloud INTEGER NOT NULL DEFAULT 0, "
            "parsed_text TEXT, summary TEXT, created_at TEXT NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO documents (id, user_id, kind, title, status, created_at) "
            "VALUES ('old1', 'me', 'pdf', '旧文档', 'done', '2026-01-01 00:00:00')"
        ))
    engine.dispose()
    init_db(engine, dim=512)
    with engine.begin() as conn:
        title, deleted_at = conn.execute(
            text("SELECT title, deleted_at FROM documents WHERE id = 'old1'")
        ).one()
    assert title == "旧文档" and deleted_at is None


def test_soft_delete_and_restore_document(tmp_path):
    s = _session(tmp_path)
    doc = documents.insert_document(s, kind="document", title="t", status="done")
    s.commit()
    assert trash.soft_delete_document(s, doc.id) is True
    assert trash.soft_delete_document(s, doc.id) is False       # 已在回收站
    assert trash.restore_document(s, doc.id) is True
    assert trash.restore_document(s, doc.id) is False           # 不在回收站
    assert trash.soft_delete_document(s, "不存在") is False


def test_soft_delete_and_restore_record(tmp_path):
    s = _session(tmp_path)
    raw = store.insert_raw(s, kind="text", content="身份证放书房")
    s.commit()
    assert trash.soft_delete_record(s, raw.id) is True
    assert trash.restore_record(s, raw.id) is True
    assert trash.restore_record(s, raw.id) is False


def _pair(s, files_dir=None, fname="a.pdf"):
    """同一次上传产生的一对：文档 + 记录（共用 file_path）；可选顺带落一个真文件。"""
    if files_dir is not None:
        files_dir.mkdir(exist_ok=True)
        (files_dir / fname).write_bytes(b"x")
    doc = documents.insert_document(
        s, kind="document", title="成绩总表", file_path=f"files/{fname}", status="done"
    )
    s.commit()
    raw = store.insert_raw(s, kind="document", file_path=f"files/{fname}")
    s.commit()
    return doc, raw


def test_soft_delete_record_also_trashes_linked_document(tmp_path):
    """联动软删：删记录 → 同一次上传的文档一并进回收站；无关对象不受影响。"""
    s = _session(tmp_path)
    doc, raw = _pair(s)
    other = store.insert_raw(s, kind="text", content="另一条没有文件的记录")
    s.commit()

    assert trash.soft_delete_record(s, raw.id) is True
    s.commit()
    assert s.get(RawEvidence, raw.id).deleted_at is not None
    assert s.get(Document, doc.id).deleted_at is not None     # 联动
    assert s.get(RawEvidence, other.id).deleted_at is None    # 无文件/无关的不动


def test_soft_delete_document_also_trashes_linked_record(tmp_path):
    """联动软删对称：删文档 → 关联记录一并进回收站。"""
    s = _session(tmp_path)
    doc, raw = _pair(s)
    s.commit()

    assert trash.soft_delete_document(s, doc.id) is True
    s.commit()
    assert s.get(Document, doc.id).deleted_at is not None
    assert s.get(RawEvidence, raw.id).deleted_at is not None


def test_restore_is_per_object(tmp_path):
    """恢复不联动：找回记录后文档仍留在回收站（想只留一边就只恢复一边）。"""
    s = _session(tmp_path)
    doc, raw = _pair(s)
    s.commit()
    assert trash.soft_delete_record(s, raw.id)
    s.commit()

    assert trash.restore_record(s, raw.id) is True
    s.commit()
    assert s.get(RawEvidence, raw.id).deleted_at is None
    assert s.get(Document, doc.id).deleted_at is not None      # 文档仍在站内等你自己点

    assert trash.restore_document(s, doc.id) is True           # 也可单独恢复文档
    s.commit()
    assert s.get(Document, doc.id).deleted_at is None


def test_list_trash_shape_and_days_left(tmp_path):
    s = _session(tmp_path)
    doc = documents.insert_document(s, kind="document", title="成绩总表", status="done")
    raw = store.insert_raw(s, kind="text", content="测试记录")
    s.commit()
    assert trash.soft_delete_document(s, doc.id) and trash.soft_delete_record(s, raw.id)
    got = trash.list_trash(s)
    assert [d["title"] for d in got["documents"]] == ["成绩总表"]
    assert got["documents"][0]["days_left"] == trash.TRASH_TTL_DAYS
    assert [r["text"] for r in got["records"]] == ["测试记录"]
    assert trash.TRASH_TTL_DAYS == 30


def test_list_trash_record_shows_user_text_and_file_name(tmp_path):
    """记录条目只暴露用户原始输入：原话（截断 100 字）+ 附件原始文件名；无附件 → None。"""
    s = _session(tmp_path)
    documents.insert_document(
        s, kind="photo", title="发票照片.jpg", file_path="files/a.jpg", status="done"
    )
    raw = store.insert_raw(s, kind="photo", content="x" * 120, file_path="files/a.jpg")
    nofile = store.insert_raw(s, kind="text", content="身份证放书房")
    s.commit()
    assert trash.soft_delete_record(s, raw.id) and trash.soft_delete_record(s, nofile.id)

    got = {r["id"]: r for r in trash.list_trash(s)["records"]}
    assert set(got[raw.id]) == {"id", "text", "file_name", "deleted_at", "days_left"}
    assert got[raw.id]["text"] == "x" * 100        # 截断到 100 字
    assert got[raw.id]["file_name"] == "发票照片.jpg"  # 附件名取同 file_path 文档行的原始文件名
    assert got[nofile.id]["text"] == "身份证放书房"
    assert got[nofile.id]["file_name"] is None


def test_list_trash_days_left_after_reload_from_db(tmp_path):
    """deleted_at 从 SQLite 读回为无时区：新会话（API 的真实路径）list_trash 不抛且天数正确。"""
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    s = Session(engine)
    doc = documents.insert_document(s, kind="document", title="t", status="done")
    s.commit()
    assert trash.soft_delete_document(s, doc.id)
    s.commit()
    s.close()
    s2 = Session(engine)
    got = trash.list_trash(s2)
    assert got["documents"][0]["days_left"] == trash.TRASH_TTL_DAYS


def test_list_trash_orders_by_deleted_at_desc(tmp_path):
    s = _session(tmp_path)
    a = documents.insert_document(s, kind="document", title="先删", status="done")
    b = documents.insert_document(s, kind="document", title="后删", status="done")
    s.commit()
    assert trash.soft_delete_document(s, a.id) and trash.soft_delete_document(s, b.id)
    s.execute(text("UPDATE documents SET deleted_at = '2026-10-01 00:00:00' WHERE id = :id"),
              {"id": a.id})
    s.execute(text("UPDATE documents SET deleted_at = '2026-10-11 00:00:00' WHERE id = :id"),
              {"id": b.id})
    s.commit()
    s.expire_all()
    assert [d["title"] for d in trash.list_trash(s)["documents"]] == ["后删", "先删"]


def test_purge_document_removes_rows_indexes_and_files(tmp_path):
    s = _session(tmp_path)
    files_dir = tmp_path / "files"; files_dir.mkdir()
    images_dir = tmp_path / "images"; images_dir.mkdir()
    (files_dir / "a.pdf").write_bytes(b"x")
    (images_dir / "i.png").write_bytes(b"y")
    doc = documents.insert_document(s, kind="document", title="t", file_path="files/a.pdf", status="done")
    s.commit()
    chunk = documents.insert_chunk(s, document_id=doc.id, seq=0, text="正文", parent_text="正文")
    s.commit()
    documents.insert_table(s, document_id=doc.id, heading=None, headers="[]", rows="[]")
    documents.insert_image(s, document_id=doc.id, file_path="images/i.png", page=1, ocr_text="")
    s.commit()
    # 建索引行（chunk_fts/chunk_vec 由 chunks.index_chunk 建）
    from app.memory import chunks as chunk_store
    chunk_store.index_chunk(s, chunk.id, [0.1] * 512, "正文")
    s.commit()

    assert trash.purge_document(s, doc.id) is True
    s.commit()
    assert s.query(Chunk).count() == 0 and s.query(DocTable).count() == 0 and s.query(Image).count() == 0
    assert s.query(Document).count() == 0
    assert not (files_dir / "a.pdf").exists() and not (images_dir / "i.png").exists()
    assert s.execute(text("SELECT count(*) FROM chunk_fts")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM chunk_vec")).scalar() == 0


def _fact(summary: str) -> ExtractedFact:
    return ExtractedFact(
        type="item_location", subject_key="身份证:位置", summary=summary, confidence=0.9
    )


def test_purge_record_removes_fact_chain_and_indexes(tmp_path):
    s = _session(tmp_path)
    raw = store.insert_raw(s, kind="text", content="身份证放书房")
    s.commit()
    old = store.insert_fact(s, _fact("身份证放书房"), raw.id)
    s.commit()
    new = store.insert_fact(s, _fact("身份证放客厅"), raw.id)
    s.commit()
    old.superseded_by = new.id  # 同一 raw 内被覆盖的旧值：也一并清除
    old.valid_to = store.utcnow()
    fts.index_fact(s, old.id, "身份证放书房")
    fts.index_fact(s, new.id, "身份证放客厅")
    vector.index_fact(s, old.id, [0.1] * 512)
    vector.index_fact(s, new.id, [0.2] * 512)
    s.commit()

    assert trash.purge_record(s, raw.id) is True
    s.commit()
    assert s.query(Fact).count() == 0 and s.query(RawEvidence).count() == 0
    assert s.execute(text("SELECT count(*) FROM memory_fts")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM memory_vec")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM vec_map")).scalar() == 0


def test_purge_record_clears_secrets_along_production_link(tmp_path):
    """生产链路回归：mask_text 打码 → link 关联 → purge 后加密副本被清。

    修复前：生产建 Secret 时从不写 `fact_id`，purge 的 `DELETE FROM secrets WHERE fact_id`
    是空操作（加密副本残留）。修复后记录接口在 fact 建好时补关联，本测试按同一链路断言。
    """
    s = _session(tmp_path)
    fernet = Fernet(Fernet.generate_key())
    masked, mapping = mask_text("我的手机号 13800000001", s, fernet)
    s.commit()
    raw = store.insert_raw(s, kind="text", content=masked)
    s.commit()
    fact = store.insert_fact(s, _fact(masked), raw.id)
    s.commit()
    link_secrets_to_fact(s, mapping, fact.id)
    s.commit()
    assert s.query(Secret).count() == 1

    assert trash.purge_record(s, raw.id) is True
    s.commit()
    assert s.query(Secret).count() == 0


def test_purge_record_keeps_other_raw_fact_and_nulls_dangling_pointer(tmp_path):
    """跨界覆盖：purge 新值记录 → 另一 raw 的旧值 fact 保留，仅 superseded_by 置 NULL。"""
    s = _session(tmp_path)
    raw1 = store.insert_raw(s, kind="text", content="身份证放书房")
    raw2 = store.insert_raw(s, kind="text", content="身份证放客厅")
    s.commit()
    old = store.insert_fact(s, _fact("身份证放书房"), raw1.id)
    s.commit()
    new = store.insert_fact(s, _fact("身份证放客厅"), raw2.id)
    s.commit()
    old.superseded_by = new.id
    old.valid_to = store.utcnow()
    s.commit()
    old_valid_to = old.valid_to

    assert trash.purge_record(s, raw2.id) is True
    s.commit()

    s.expire_all()
    left = s.get(Fact, old.id)
    assert left is not None and left.superseded_by is None
    assert left.valid_to == old_valid_to            # 历史事实本身不动
    assert s.query(RawEvidence).count() == 1        # 只删掉 raw2


def test_purge_tolerates_missing_files(tmp_path):
    """磁盘文件按「无活对象引用」删：共享路径只在两侧都不活时才真删；不存在 → 忽略。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "shared.pdf").write_bytes(b"x")
    doc = documents.insert_document(s, kind="document", title="t", file_path="files/shared.pdf", status="done")
    shared = store.insert_raw(s, kind="document", file_path="files/shared.pdf")
    gone = store.insert_raw(s, kind="text", content="x", file_path="files/从未存在.pdf")
    s.commit()

    assert trash.purge_document(s, doc.id) is True     # 文档清掉，但记录还活着引用 → 文件留住
    assert (files_dir / "shared.pdf").exists()
    assert trash.purge_record(s, shared.id) is True    # 两侧都不在了 → 文件才真删
    assert trash.purge_record(s, gone.id) is True      # 从未存在 → 照常删行
    s.commit()
    assert s.query(Document).count() == 0 and s.query(RawEvidence).count() == 0
    assert not (files_dir / "shared.pdf").exists()


def test_purge_record_also_purges_linked_document_in_trash(tmp_path):
    """联动彻底删除：一对都在回收站 → 删任一，两条记录行、索引与磁盘文件全清。"""
    from app.memory import chunks as chunk_store

    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    doc, raw = _pair(s, files_dir)
    chunk = documents.insert_chunk(s, document_id=doc.id, seq=0, text="正文", parent_text="正文")
    s.commit()
    chunk_store.index_chunk(s, chunk.id, [0.1] * 512, "正文")
    s.commit()
    assert trash.soft_delete_record(s, raw.id)  # 联动：两条都进站
    s.commit()

    assert trash.purge_record(s, raw.id) is True
    s.commit()
    assert s.query(RawEvidence).count() == 0 and s.query(Document).count() == 0
    assert s.query(Chunk).count() == 0
    assert s.execute(text("SELECT count(*) FROM chunk_fts")).scalar() == 0
    assert not (files_dir / "a.pdf").exists()
    assert trash.purge_document(s, doc.id) is False  # 已随联动物理消失


def test_purge_record_keeps_live_linked_document_and_file(tmp_path):
    """防误伤：另一半还活着（历史半边状态）→ 彻底删除只清记录，文档与磁盘文件都保住。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    doc, raw = _pair(s, files_dir)
    assert trash.soft_delete_record(s, raw.id)
    assert trash.restore_document(s, doc.id)  # 造出「记录在站内、文档活着」的半边状态
    s.commit()

    assert trash.purge_record(s, raw.id) is True
    s.commit()
    assert s.query(RawEvidence).count() == 0
    assert s.get(Document, doc.id) is not None        # 活文档不动
    assert (files_dir / "a.pdf").exists()             # 活文档的附件文件不删


def test_purge_document_keeps_live_linked_record_and_file(tmp_path):
    """对称：文档在站内、记录活着 → 彻底删除文档不动记录与文件。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    doc, raw = _pair(s, files_dir)
    assert trash.soft_delete_document(s, doc.id)
    assert trash.restore_record(s, raw.id)
    s.commit()

    assert trash.purge_document(s, doc.id) is True
    s.commit()
    assert s.get(Document, doc.id) is None
    assert s.get(RawEvidence, raw.id) is not None
    assert (files_dir / "a.pdf").exists()


def test_purge_missing_object_returns_false(tmp_path):
    assert trash.purge_document(_session(tmp_path), "不存在") is False
    assert trash.purge_record(_session(tmp_path), "不存在") is False


def test_sweep_expired_purges_only_over_ttl(tmp_path):
    s = _session(tmp_path)
    old = documents.insert_document(s, kind="document", title="老", status="done")
    new = documents.insert_document(s, kind="document", title="新", status="done")
    s.commit()
    trash.soft_delete_document(s, old.id)
    trash.soft_delete_document(s, new.id)
    # 把 old 的删除时间伪造成 31 天前
    s.execute(text("UPDATE documents SET deleted_at = :t WHERE id = :id"),
              {"t": utcnow() - timedelta(days=31), "id": old.id})
    s.commit()
    assert trash.sweep_expired(s) == 1
    assert s.get(Document, old.id) is None
    assert s.get(Document, new.id) is not None


def test_sweep_expired_keeps_within_ttl(tmp_path):
    """到期边界：29 天（未到期）不动，只有 31 天（超期）被清。"""
    s = _session(tmp_path)
    keep = documents.insert_document(s, kind="document", title="29天", status="done")
    drop = documents.insert_document(s, kind="document", title="31天", status="done")
    s.commit()
    trash.soft_delete_document(s, keep.id)
    trash.soft_delete_document(s, drop.id)
    s.execute(text("UPDATE documents SET deleted_at = :t WHERE id = :id"),
              {"t": utcnow() - timedelta(days=29), "id": keep.id})
    s.execute(text("UPDATE documents SET deleted_at = :t WHERE id = :id"),
              {"t": utcnow() - timedelta(days=31), "id": drop.id})
    s.commit()
    assert trash.sweep_expired(s) == 1
    assert s.get(Document, keep.id) is not None
    assert s.get(Document, drop.id) is None


def test_sweep_continues_after_single_failure(tmp_path, monkeypatch, caplog):
    # 两个超期对象；monkeypatch purge_document 让第一条抛错 → sweep 不中断、第二条仍被清
    s = _session(tmp_path)
    bad = documents.insert_document(s, kind="document", title="坏", status="done")
    good = documents.insert_document(s, kind="document", title="好", status="done")
    raw = store.insert_raw(s, kind="text", content="超期记录")
    s.commit()
    trash.soft_delete_document(s, bad.id)
    trash.soft_delete_document(s, good.id)
    trash.soft_delete_record(s, raw.id)
    s.execute(text("UPDATE documents SET deleted_at = :t"), {"t": utcnow() - timedelta(days=31)})
    s.execute(text("UPDATE raw_evidence SET deleted_at = :t"), {"t": utcnow() - timedelta(days=31)})
    s.commit()

    original = trash.purge_document

    def flaky_purge_document(session, doc_id):
        if doc_id == bad.id:
            raise RuntimeError("模拟单条清除失败")
        return original(session, doc_id)

    monkeypatch.setattr(trash, "purge_document", flaky_purge_document)
    with caplog.at_level("WARNING"):
        assert trash.sweep_expired(s) == 2
    assert s.get(Document, bad.id) is not None   # 失败那条留在回收站（不留半删痕迹）
    assert s.get(Document, good.id) is None      # 同批后续对象不中断
    assert s.get(RawEvidence, raw.id) is None    # 记录也不受文档失败影响
    assert "回收站到期清扫：单条失败已跳过" in caplog.text


def test_empty_trash_purges_everything_and_counts(tmp_path):
    """一键清空：站内记录与文档全部彻底删除，返回各类型计数；索引行与磁盘文件同清。"""
    from app.memory import chunks as chunk_store

    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    (files_dir / "a.pdf").write_bytes(b"x")
    (files_dir / "r.txt").write_bytes(b"y")
    doc = documents.insert_document(
        s, kind="document", title="t", file_path="files/a.pdf", status="done"
    )
    s.commit()
    chunk = documents.insert_chunk(s, document_id=doc.id, seq=0, text="正文", parent_text="正文")
    s.commit()
    chunk_store.index_chunk(s, chunk.id, [0.1] * 512, "正文")
    raw = store.insert_raw(s, kind="photo", content="身份证", file_path="files/r.txt")
    s.commit()
    fact = store.insert_fact(s, _fact("身份证放书房"), raw.id)
    s.commit()
    fts.index_fact(s, fact.id, "身份证放书房")
    vector.index_fact(s, fact.id, [0.2] * 512)
    s.commit()

    assert trash.soft_delete_document(s, doc.id) and trash.soft_delete_record(s, raw.id)
    s.commit()

    assert trash.empty_trash(s) == {"records": 1, "documents": 1}
    assert s.query(Document).count() == 0 and s.query(RawEvidence).count() == 0
    assert s.query(Chunk).count() == 0 and s.query(Fact).count() == 0
    assert s.execute(text("SELECT count(*) FROM chunk_fts")).scalar() == 0
    assert s.execute(text("SELECT count(*) FROM memory_fts")).scalar() == 0
    assert not (files_dir / "a.pdf").exists() and not (files_dir / "r.txt").exists()


def test_empty_trash_ignores_live_objects(tmp_path):
    """只作用于回收站内：活对象（未软删）原样保留。"""
    s = _session(tmp_path)
    before = documents.insert_document(s, kind="document", title="活着", status="done")
    dead = documents.insert_document(s, kind="document", title="已删", status="done")
    s.commit()
    assert trash.soft_delete_document(s, dead.id)
    s.commit()

    assert trash.empty_trash(s) == {"records": 0, "documents": 1}
    assert s.get(Document, before.id) is not None
    assert s.get(Document, dead.id) is None


def test_empty_trash_on_empty_station_returns_zero(tmp_path):
    """空站清空：返回 0 计数而非报错（不针对具体 id，无 404 语义）。"""
    assert trash.empty_trash(_session(tmp_path)) == {"records": 0, "documents": 0}


def test_empty_trash_continues_after_single_failure(tmp_path, monkeypatch, caplog):
    """单条失败跳过：失败对象保留、另一类型不受影响、返回计数只算成功。"""
    s = _session(tmp_path)
    bad = documents.insert_document(s, kind="document", title="坏", status="done")
    raw = store.insert_raw(s, kind="text", content="记录")
    s.commit()
    trash.soft_delete_document(s, bad.id)
    trash.soft_delete_record(s, raw.id)
    s.commit()

    def flaky_purge_document(session, doc_id):
        raise RuntimeError("模拟单条清除失败")

    monkeypatch.setattr(trash, "purge_document", flaky_purge_document)
    with caplog.at_level("WARNING"):
        assert trash.empty_trash(s) == {"records": 1, "documents": 0}
    assert s.get(Document, bad.id) is not None   # 失败那条留在回收站
    assert s.get(RawEvidence, raw.id) is None    # 记录照常清除
    assert "清空回收站：单条失败已跳过" in caplog.text


def test_empty_trash_linked_pair_counts_and_clears(tmp_path):
    """一对都在站内：一键清空 = 两条都物理消失，计数各算一次、文件同删。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    doc, raw = _pair(s, files_dir)
    assert trash.soft_delete_record(s, raw.id)  # 联动：两条都进站
    s.commit()

    assert trash.empty_trash(s) == {"records": 1, "documents": 1}
    assert s.query(Document).count() == 0 and s.query(RawEvidence).count() == 0
    assert not (files_dir / "a.pdf").exists()


def test_sweep_expired_linked_pair_clears_both(tmp_path):
    """一对同批进站、超期 → 清扫整对清理（两条都算清除），文件同删。"""
    s = _session(tmp_path)
    files_dir = tmp_path / "files"
    doc, raw = _pair(s, files_dir)
    assert trash.soft_delete_record(s, raw.id)
    s.commit()
    s.execute(text("UPDATE raw_evidence SET deleted_at = :t"), {"t": utcnow() - timedelta(days=31)})
    s.execute(text("UPDATE documents SET deleted_at = :t"), {"t": utcnow() - timedelta(days=31)})
    s.commit()

    assert trash.sweep_expired(s) == 2
    assert s.query(Document).count() == 0 and s.query(RawEvidence).count() == 0
    assert not (files_dir / "a.pdf").exists()


def test_purge_removes_material_rows_and_indexes(tmp_path, monkeypatch):
    """彻底删除清账：记录/文档的 materials 行与 material_fts/material_vec 索引行一并清除；
    无关材料行不受影响。"""
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    s = _session(tmp_path)
    raw = store.insert_raw(s, kind="text", content="身份证放书房")
    doc = documents.insert_document(
        s, kind="text", title="成绩总表", parsed_text="数学 95", status="done"
    )
    other = store.insert_raw(s, kind="text", content="另一条记录")
    s.commit()
    materials.upsert_for_record(s, raw, title="身份证放书房", model="m")
    materials.upsert_for_document(s, doc, model="m")
    materials.upsert_for_record(s, other, title="另一条记录", model="m")
    s.commit()
    assert s.query(Material).count() == 3

    assert trash.purge_record(s, raw.id) is True
    s.commit()
    assert trash.purge_document(s, doc.id) is True
    s.commit()

    assert s.query(Material).count() == 1  # 只剩无关那条
    rest = s.query(Material).one()
    assert (rest.source_type, rest.source_id) == ("record", other.id)
    assert s.execute(text("SELECT count(*) FROM material_fts")).scalar() == 1
    assert s.execute(
        text("SELECT count(*) FROM material_fts WHERE material_id = :m"), {"m": rest.id}
    ).scalar() == 1
    assert s.execute(text("SELECT count(*) FROM material_vec")).scalar() == 1
    assert s.execute(
        text("SELECT count(*) FROM material_vec WHERE rowid = :r"), {"r": rest.vec_rowid}
    ).scalar() == 1
