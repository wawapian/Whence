"""回收站：软删 / 恢复 / 回收站列表 / 彻底删除 / 到期清扫。

派生数据（chunks / 索引 / doc_tables / images）原地保留，只给主对象打 `deleted_at`
时间戳（与双时态 `valid_to` 同族：关闭可见性、不销毁数据）→ 恢复 = 抹掉时间戳，零重建。

彻底删除（purge）才真正逐表清理派生行、索引行与磁盘文件；不 commit，由调用方提交。
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import and_, or_, select, text
from sqlalchemy.orm import Session

from app.memory import qa_history
from app.memory.store import to_local, utcnow
from app.models.db import Document, Fact, RawEvidence, VecMap

TRASH_TTL_DAYS = 30

logger = logging.getLogger(__name__)


def soft_delete_document(session: Session, doc_id: str) -> bool:
    """移入回收站：设 `deleted_at=utcnow()`。不存在或已在回收站 → False。

    联动：同一次上传的组（所属记录及全部附件）中所有 live 对象一并移入、共用同一时间戳，
    已在站内的保留原时间戳——一次上传是「一个东西」。恢复仍逐条。
    """
    doc = session.get(Document, doc_id)
    if doc is None or doc.deleted_at is not None:
        return False
    now = utcnow()
    doc.deleted_at = now
    raw = _group_record(session, doc)
    if raw is not None:
        if raw.deleted_at is None:
            raw.deleted_at = now
        for other in _group_documents(session, raw, deleted=False):
            other.deleted_at = now
    qa_history.bump(session)
    return True


def soft_delete_record(session: Session, raw_id: str) -> bool:
    """移入回收站：设 `deleted_at=utcnow()`。不存在或已在回收站 → False。

    联动：同一次上传的组（本记录的全部附件）中所有 live 附件一并移入、共用同一时间戳。
    """
    raw = session.get(RawEvidence, raw_id)
    if raw is None or raw.deleted_at is not None:
        return False
    now = utcnow()
    raw.deleted_at = now
    for doc in _group_documents(session, raw, deleted=False):
        doc.deleted_at = now
    qa_history.bump(session)
    return True


def restore_document(session: Session, doc_id: str) -> bool:
    """从回收站恢复：抹 `deleted_at`。不存在或不在回收站 → False。"""
    doc = session.get(Document, doc_id)
    if doc is None or doc.deleted_at is None:
        return False
    doc.deleted_at = None
    qa_history.bump(session)
    return True


def restore_record(session: Session, raw_id: str) -> bool:
    """从回收站恢复：抹 `deleted_at`。不存在或不在回收站 → False。"""
    raw = session.get(RawEvidence, raw_id)
    if raw is None or raw.deleted_at is None:
        return False
    raw.deleted_at = None
    qa_history.bump(session)
    return True


def purge_document(session: Session, doc_id: str) -> bool:
    """彻底删除文档：派生行、索引、表格、图片与图片文件全清。不存在 → False。

    联动彻底删除：组（所属记录及其它附件）中**也在回收站**的对象一并清理（活对象永不触碰）。
    磁盘原文件按「无活对象引用」逐路径删（见 `_remove_file_unreferenced`）。
    不要求先软删；不 commit（调用方提交）；任一数据库步骤失败直接上抛，由调用方回滚。
    """
    doc = session.get(Document, doc_id)
    if doc is None:
        return False
    raw = _group_record(session, doc)
    others = [] if raw is None else [
        other for other in _group_documents(session, raw, deleted=True) if other.id != doc.id
    ]
    file_paths = [doc.file_path, *(other.file_path for other in others)]
    _purge_document_rows(session, doc)
    for other in others:
        _purge_document_rows(session, other)
    if raw is not None and raw.deleted_at is not None:
        file_paths.append(raw.file_path)
        _purge_record_rows(session, raw)
    for file_path in file_paths:
        _remove_file_unreferenced(session, file_path)
    qa_history.bump(session)
    return True


def purge_record(session: Session, raw_id: str) -> bool:
    """彻底删除记录：事实链（含被覆盖旧值）、加密值与索引行全清。不存在 → False。

    联动彻底删除：组内**也在回收站**的附件一并清理（活对象永不触碰）。
    跨界双时态指针：其它 raw 的 fact 若指向本次被删 fact（`superseded_by`），置 NULL——
    旧值 fact 本身及其 `valid_to` 保留，只是指向的目标没了。
    磁盘原文件按「无活对象引用」逐路径删（见 `_remove_file_unreferenced`）。
    不要求先软删；不 commit（调用方提交）；任一数据库步骤失败直接上抛，由调用方回滚。
    """
    raw = session.get(RawEvidence, raw_id)
    if raw is None:
        return False
    docs = _group_documents(session, raw, deleted=True)
    file_paths = [raw.file_path, *(doc.file_path for doc in docs)]
    _purge_record_rows(session, raw)
    for doc in docs:
        _purge_document_rows(session, doc)
    for file_path in file_paths:
        _remove_file_unreferenced(session, file_path)
    qa_history.bump(session)
    return True


def _purge_document_rows(session: Session, doc: Document) -> None:
    """只清文档自身的派生行、索引与文档行（含图片文件）；不动磁盘原文件、不走路联动。"""
    chunks = session.execute(
        text("SELECT id, vec_rowid FROM chunks WHERE document_id = :doc"), {"doc": doc.id}
    ).fetchall()
    for chunk_id, vec_rowid in chunks:
        session.execute(text("DELETE FROM chunk_fts WHERE chunk_id = :id"), {"id": chunk_id})
        if vec_rowid is not None:
            session.execute(text("DELETE FROM chunk_vec WHERE rowid = :r"), {"r": vec_rowid})
    session.execute(text("DELETE FROM chunks WHERE document_id = :doc"), {"doc": doc.id})
    _purge_material_rows(session, "document", doc.id)
    session.execute(text("DELETE FROM doc_tables WHERE document_id = :doc"), {"doc": doc.id})
    images = session.execute(
        text("SELECT file_path FROM images WHERE document_id = :doc"), {"doc": doc.id}
    ).fetchall()
    session.execute(text("DELETE FROM images WHERE document_id = :doc"), {"doc": doc.id})
    for (file_path,) in images:
        # 独立照片的 images 行引用的是 files/ 原文件（可能与记录共用）——统一按
        # 「无活对象引用才删」，防单独彻底删除文档时误删仍活着的记录的附件文件
        _remove_file_unreferenced(session, file_path)
    session.delete(doc)


def _purge_record_rows(session: Session, raw: RawEvidence) -> None:
    """只清记录自身的事实链、加密值与索引行；不动磁盘原文件、不走路联动。"""
    fact_ids = session.scalars(select(Fact.id).where(Fact.raw_id == raw.id)).all()
    for fact_id in fact_ids:
        session.execute(text("DELETE FROM secrets WHERE fact_id = :f"), {"f": fact_id})
        session.execute(text("DELETE FROM memory_fts WHERE fact_id = :f"), {"f": fact_id})
        for vec_rowid in session.scalars(
            select(VecMap.vec_rowid).where(VecMap.fact_id == fact_id)
        ):
            session.execute(text("DELETE FROM memory_vec WHERE rowid = :r"), {"r": vec_rowid})
        session.execute(text("DELETE FROM vec_map WHERE fact_id = :f"), {"f": fact_id})
    session.execute(
        text(
            "UPDATE facts SET superseded_by = NULL "
            "WHERE superseded_by IN (SELECT id FROM facts WHERE raw_id = :r)"
        ),
        {"r": raw.id},
    )
    session.execute(text("DELETE FROM facts WHERE raw_id = :r"), {"r": raw.id})
    _purge_material_rows(session, "record", raw.id)
    session.delete(raw)


def _purge_material_rows(session: Session, source_type: str, source_id: str) -> None:
    """清一份材料（记录/文档）的 materials 行与 material_fts/material_vec 索引行。

    先查 material_id 再逐表删；vec 按 `materials.vec_rowid`（未回填的行无向量行可删）。
    """
    rows = session.execute(
        text("SELECT id, vec_rowid FROM materials WHERE source_type = :t AND source_id = :s"),
        {"t": source_type, "s": source_id},
    ).fetchall()
    for material_id, vec_rowid in rows:
        session.execute(text("DELETE FROM material_fts WHERE material_id = :id"), {"id": material_id})
        if vec_rowid is not None:
            session.execute(text("DELETE FROM material_vec WHERE rowid = :r"), {"r": vec_rowid})
    session.execute(
        text("DELETE FROM materials WHERE source_type = :t AND source_id = :s"),
        {"t": source_type, "s": source_id},
    )


def _group_record(session: Session, doc: Document) -> RawEvidence | None:
    """附件所属的记录：`doc.raw_id` 直取；为 NULL（迁移前旧行）时按 `file_path` 兜底取最早一条。

    `doc.raw_id` 已设但目标行不存在（悬挂，如 raw 被单独彻底删除）时不回落到 `file_path` 匹配——
    与迁移回填口径一致，生产不可达，系有意为之。
    """
    if doc.raw_id:
        return session.get(RawEvidence, doc.raw_id)
    if not doc.file_path:
        return None
    return session.scalars(
        select(RawEvidence)
        .where(RawEvidence.file_path == doc.file_path)
        .order_by(RawEvidence.created_at)
        .limit(1)
    ).first()


def _group_documents(session: Session, raw: RawEvidence, *, deleted: bool) -> list[Document]:
    """同一次上传的附件：`raw_id == raw.id`，或（旧行 `raw_id IS NULL` 时）同 `file_path` 兜底。

    `deleted=False` 取未删（联动软删用）、`True` 取在回收站的（联动彻底删除用）。
    双键并存是过渡设计：新数据一律写 `raw_id`，`file_path` 键仅为历史数据兜底。
    """
    cond = Document.deleted_at.isnot(None) if deleted else Document.deleted_at.is_(None)
    clauses = [Document.raw_id == raw.id]
    if raw.file_path:
        clauses.append(and_(Document.raw_id.is_(None), Document.file_path == raw.file_path))
    return session.scalars(select(Document).where(or_(*clauses), cond)).all()


def _remove_file_unreferenced(session: Session, file_path: str | None) -> None:
    """删磁盘原文件：仅当没有任何活对象（未删的记录/文档）仍引用该路径。

    一次上传的记录与文档共用同一文件；彻底删除任一侧时另一半还可能活着——有人引用就不删。
    """
    if not file_path:
        return
    still = session.scalars(
        select(RawEvidence.id).where(
            RawEvidence.file_path == file_path, RawEvidence.deleted_at.is_(None)
        )
    ).first()
    if still is None:
        still = session.scalars(
            select(Document.id).where(
                Document.file_path == file_path, Document.deleted_at.is_(None)
            )
        ).first()
    if still is None:
        _remove_file(session, file_path)


def sweep_expired(session: Session) -> int:
    """清扫回收站里超过 TTL 的文档与记录，返回成功彻底清除的条数。

    到期判定在 SQL 里比较（`deleted_at < cutoff`）：SQLite 存 ISO 字符串、同格式可比，
    避免 Python 侧 naive/aware 混算。逐条提交，单条失败跳过（见 `_purge_each`）。
    """
    cutoff = utcnow() - timedelta(days=TRASH_TTL_DAYS)
    doc_ids = session.scalars(
        select(Document.id).where(Document.deleted_at.isnot(None), Document.deleted_at < cutoff)
    ).all()
    raw_ids = session.scalars(
        select(RawEvidence.id).where(RawEvidence.deleted_at.isnot(None), RawEvidence.deleted_at < cutoff)
    ).all()
    counts = _purge_each(session, doc_ids, raw_ids, label="回收站到期清扫")
    return counts["records"] + counts["documents"]


def empty_trash(session: Session) -> dict[str, int]:
    """清空回收站：站内全部对象逐个彻底删除，返回各类型成功清除条数。

    只作用于回收站内的对象（活对象不在范围）；空站返回 0 计数而非报错——
    不针对具体 id，没有 404 语义。逐条提交，单条失败跳过（见 `_purge_each`）。
    """
    doc_ids = session.scalars(
        select(Document.id).where(Document.deleted_at.isnot(None))
    ).all()
    raw_ids = session.scalars(
        select(RawEvidence.id).where(RawEvidence.deleted_at.isnot(None))
    ).all()
    return _purge_each(session, doc_ids, raw_ids, label="清空回收站")


def _purge_each(session: Session, document_ids: list[str], record_ids: list[str],
                label: str) -> dict[str, int]:
    """逐条彻底删除并提交；单条失败回滚该条、记 warning 继续。返回各类型实际清除条数。

    每成功一条即 commit——已清对象不因后续失败回退，失败对象不留半删痕迹。
    计数按**结果**核对（入参清单里的对象现在是否真的不在库）：联动把关联对象一并清掉的，
    只要它也在清单里就照样计入；失败留存的一律不计。
    """
    for purge, object_ids in ((purge_document, document_ids), (purge_record, record_ids)):
        for object_id in object_ids:
            try:
                if purge(session, object_id):
                    session.commit()
            except Exception:
                session.rollback()
                logger.warning("%s：单条失败已跳过", label, exc_info=True)
    counts = {"records": 0, "documents": 0}
    for object_id in record_ids:
        if session.get(RawEvidence, object_id) is None:
            counts["records"] += 1
    for object_id in document_ids:
        if session.get(Document, object_id) is None:
            counts["documents"] += 1
    return counts


def list_trash(session: Session) -> dict:
    """回收站内容（记录 + 文档，各按 `deleted_at` 倒序）。

    记录条目只暴露用户记录时的原始输入：`text`（原话，截断 100 字）与 `file_name`
    （附件原始文件名）——不展示 AI 摘要，让用户认得出「我删的是哪条」。
    """
    records = [
        {
            "id": raw.id,
            "text": (raw.content or "")[:100],
            "file_name": _file_name(session, raw),
            "deleted_at": to_local(raw.deleted_at),
            "days_left": _days_left(raw.deleted_at),
        }
        for raw in session.scalars(
            select(RawEvidence)
            .where(RawEvidence.deleted_at.isnot(None))
            .order_by(RawEvidence.deleted_at.desc())
        )
    ]
    documents = [
        {
            "id": doc.id,
            "title": doc.title,
            "raw_id": doc.raw_id,
            "kind": doc.kind,
            "deleted_at": to_local(doc.deleted_at),
            "days_left": _days_left(doc.deleted_at),
        }
        for doc in session.scalars(
            select(Document)
            .where(Document.deleted_at.isnot(None))
            .order_by(Document.deleted_at.desc())
        )
    ]
    return {"records": records, "documents": documents}


def _file_name(session: Session, raw: RawEvidence) -> str | None:
    """附件原始文件名：与记录同 `file_path` 的文档行 title（上传时记录的原名）。

    `file_path` 本身是 `files/<uuid>.<ext>`，对用户无意义，故不以它兜底；
    查不到对应文档行（如文档行已被单独彻底删除）→ None。
    """
    if not raw.file_path:
        return None
    return session.scalar(
        select(Document.title)
        .where(Document.file_path == raw.file_path)
        .order_by(Document.created_at)
        .limit(1)
    )


def _days_left(deleted_at: datetime) -> int:
    """剩余天数（整数天，可为 0）。SQLite 读回无时区，按 UTC 解释。"""
    if deleted_at.tzinfo is None:
        deleted_at = deleted_at.replace(tzinfo=timezone.utc)
    return TRASH_TTL_DAYS - (utcnow() - deleted_at).days


def _remove_file(session: Session, file_path: str | None) -> None:
    """删 data_dir 下的文件；未记录或文件已不存在 → 忽略（同一路径可能被两处引用）。"""
    if not file_path:
        return
    path = Path(file_path)
    if not path.is_absolute():
        path = _data_dir(session) / path
    path.unlink(missing_ok=True)


def _data_dir(session: Session) -> Path:
    """DB 文件所在目录即 data_dir：`file_path` 列存的是相对 data_dir 的路径。"""
    database = session.get_bind().url.database
    return Path(database).parent if database else Path(".")
