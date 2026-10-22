"""验收修复#4：chunks/materials 的 vec_rowid 索引钉子（EXPLAIN QUERY PLAN）。

背景：`chunks.vector_search` / `materials.vector_search` 的 join
`JOIN chunks c ON c.vec_rowid = v.rowid` 在 vec_rowid 无索引时，SQLite 对每个 KNN
命中行全表扫 chunks（生产库 1241 行、含长 text 列）——3.9s/次；补索引后计划变
SEARCH、10ms 级。成本约 O(k × N)，随语料线性放大，materials 同形状（现在行少不显）。
"""
import logging
import re

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from app.memory import chunks, materials
from app.models.db import init_db

INDEXES = {"chunks": "ix_chunks_vec_rowid", "materials": "ix_materials_vec_rowid"}


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=4)
    return Session(engine)


def _index_names(session, table):
    rows = session.execute(
        text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t"),
        {"t": table},
    ).fetchall()
    return {r[0] for r in rows}


def _production_plan(session, fn):
    """截获 fn 实际执行的生产 SQL 原文，返回其 EXPLAIN QUERY PLAN 的 detail 列表。

    截获而非复制 SQL：断言的就是检索链路真实跑的那条语句，杜绝测试里的 SQL 副本漂移。
    """
    seen = {}

    def _hook(conn, cursor, statement, parameters, context, executemany):
        if "embedding" in statement and "MATCH" in statement and not seen:
            seen["sql"] = statement
            seen["params"] = parameters

    bind = session.get_bind()
    event.listen(bind, "before_cursor_execute", _hook)
    try:
        fn(session)
    finally:
        event.remove(bind, "before_cursor_execute", _hook)
    rows = session.connection().exec_driver_sql(
        "EXPLAIN QUERY PLAN " + seen["sql"], seen["params"]
    ).fetchall()
    return [r[3] for r in rows]


def _scans(plan, alias):
    """计划里是否存在对别名 alias 的全表扫描（SCAN alias，排除 SEARCH/其他表）。"""
    return [d for d in plan if re.match(rf"SCAN {alias}\b", d)]


def _drop_indexes(session):
    for index in INDEXES.values():
        session.execute(text(f"DROP INDEX {index}"))
    session.commit()


def test_init_db_creates_vec_rowid_indexes(tmp_path):
    session = _session(tmp_path)
    for table, index in INDEXES.items():
        assert index in _index_names(session, table), f"{table} 缺索引 {index}"
        cols = session.execute(text(f"PRAGMA index_info({index})")).fetchall()
        assert [c[2] for c in cols] == ["vec_rowid"]  # 索引列就是 join 键


def test_chunk_vector_search_plan_searches_chunks_by_index(tmp_path):
    session = _session(tmp_path)
    plan = _production_plan(session, lambda s: chunks.vector_search(s, [0.1] * 4, k=30))
    assert "SEARCH c USING INDEX ix_chunks_vec_rowid" in " | ".join(plan), plan
    assert not _scans(plan, "c"), plan  # 不许再全表扫 chunks


def test_material_vector_search_plan_searches_materials_by_index(tmp_path):
    session = _session(tmp_path)
    plan = _production_plan(session, lambda s: materials.vector_search(s, [0.1] * 4, k=30))
    assert "SEARCH m USING INDEX ix_materials_vec_rowid" in " | ".join(plan), plan
    assert not _scans(plan, "m"), plan


def test_plan_pin_reverts_to_scan_without_index(tmp_path):
    """反向钉子：索引一旦不在，计划就退回 SCAN——证明上面两条断言不是空转。"""
    session = _session(tmp_path)
    _drop_indexes(session)
    chunk_plan = _production_plan(session, lambda s: chunks.vector_search(s, [0.1] * 4, k=30))
    material_plan = _production_plan(session, lambda s: materials.vector_search(s, [0.1] * 4, k=30))
    assert _scans(chunk_plan, "c"), chunk_plan
    assert _scans(material_plan, "m"), material_plan


def test_old_db_gets_indexes_on_next_init_db(tmp_path):
    """存量库路径：老库（此处以 DROP INDEX 模拟）下次 init_db 自动补建索引。"""
    session = _session(tmp_path)
    _drop_indexes(session)
    # 先确认真删掉了（否则「迁移后索引存在」的断言是空转）
    for table, index in INDEXES.items():
        assert index not in _index_names(session, table)

    init_db(session.get_bind(), dim=4)  # 幂等：再跑一次不报错，且索引回来了
    for table, index in INDEXES.items():
        assert index in _index_names(session, table), f"{table} 迁移后仍缺 {index}"


def test_index_migration_failure_logs_warning(tmp_path, caplog):
    """索引迁移失败不许再静默（终审 Minor#3）：chunks 缺 vec_rowid 列 → 建索引抛错 → warning
    点名索引；materials 正常则不记（只记真实失败），且失败不中断其余迁移。"""
    session = _session(tmp_path)
    _drop_indexes(session)
    session.execute(text("ALTER TABLE chunks DROP COLUMN vec_rowid"))
    session.commit()

    with caplog.at_level(logging.WARNING, logger="app.models.db"):
        init_db(session.get_bind(), dim=4)

    assert "建索引 ix_chunks_vec_rowid 失败" in caplog.text
    assert "ix_materials_vec_rowid" not in caplog.text  # 正常路径不记（防噪音）
    assert INDEXES["chunks"] not in _index_names(session, "chunks")  # 确是真失败（非空转）
    assert INDEXES["materials"] in _index_names(session, "materials")  # 不中断其余迁移
