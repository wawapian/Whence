"""CHUNK 分块索引：chunk_vec 向量检索 + chunk_fts 全文检索。"""
import logging

import sqlite_vec
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.memory.fts import fts_query
from app.models.db import Chunk

logger = logging.getLogger(__name__)


def index_chunk(session: Session, chunk_id: str, vec: list[float], content: str) -> None:
    session.execute(
        text("INSERT INTO chunk_vec(embedding) VALUES (:vec)"),
        {"vec": sqlite_vec.serialize_float32(vec)},
    )
    rowid = session.execute(text("SELECT last_insert_rowid()")).scalar()
    session.execute(
        text("UPDATE chunks SET vec_rowid = :r WHERE id = :id"),
        {"r": rowid, "id": chunk_id},
    )
    session.execute(
        text("INSERT INTO chunk_fts(content, chunk_id) VALUES (:c, :id)"),
        {"c": content, "id": chunk_id},
    )


def vector_search(session: Session, vec: list[float], k: int = 10) -> list[str]:
    rows = session.execute(
        text(
            "SELECT c.id FROM chunk_vec v JOIN chunks c ON c.vec_rowid = v.rowid "
            "JOIN documents d ON d.id = c.document_id AND d.deleted_at IS NULL "
            "WHERE v.embedding MATCH :vec AND k = :k ORDER BY v.distance"
        ),
        {"vec": sqlite_vec.serialize_float32(vec), "k": k},
    ).fetchall()
    return [r[0] for r in rows]


def fts_search(session: Session, q: str, k: int = 10) -> list[str]:
    try:
        # chunk_fts 保持原名不别名：FTS5 的 MATCH 语法要求左操作数是 fts 表本身
        rows = session.execute(
            text(
                "SELECT c.id FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.chunk_id "
                "JOIN documents d ON d.id = c.document_id AND d.deleted_at IS NULL "
                "WHERE chunk_fts MATCH :q ORDER BY rank LIMIT :k"
            ),
            {"q": fts_query(q), "k": k},
        ).fetchall()
    except Exception:
        # FTS 失败只降级为无关键词命中，不打断向量检索主链路
        logger.warning("chunk_fts 查询失败（已降级为纯向量检索）", exc_info=True)
        return []
    return [r[0] for r in rows]


def chunk_text(session: Session, chunk_id: str) -> str:
    chunk = session.get(Chunk, chunk_id)
    return chunk.text if chunk else ""
