"""sqlite-vec 向量检索封装。"""
import sqlite_vec
from sqlalchemy import text
from sqlalchemy.orm import Session


def index_fact(session: Session, fact_id: str, vec: list[float]):
    session.execute(
        text("INSERT INTO memory_vec(embedding) VALUES (:vec)"),
        {"vec": sqlite_vec.serialize_float32(vec)},
    )
    vec_rowid = session.execute(text("SELECT last_insert_rowid()")).scalar()
    session.execute(
        text("INSERT INTO vec_map(vec_rowid, fact_id) VALUES (:rowid, :fact_id)"),
        {"rowid": vec_rowid, "fact_id": fact_id},
    )


def search(session: Session, vec: list[float], k: int = 10) -> list[str]:
    # LEFT JOIN 只排除「记录已进回收站」的 fact，无 fact 的孤儿索引行照旧命中
    rows = session.execute(
        text(
            "SELECT vm.fact_id FROM memory_vec v "
            "JOIN vec_map vm ON vm.vec_rowid = v.rowid "
            "LEFT JOIN facts f ON f.id = vm.fact_id "
            "LEFT JOIN raw_evidence r ON r.id = f.raw_id "
            "WHERE v.embedding MATCH :vec AND k = :k AND r.deleted_at IS NULL "
            "ORDER BY v.distance"
        ),
        {"vec": sqlite_vec.serialize_float32(vec), "k": k},
    ).fetchall()
    return [r[0] for r in rows]
