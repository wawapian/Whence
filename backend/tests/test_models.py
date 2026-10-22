from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app.core import llm
from app.models.db import Chunk, Document, Fact, init_db


def _mk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    return engine


def test_tables_created(tmp_path):
    engine = _mk(tmp_path)
    names = set(inspect(engine).get_table_names())
    for t in ["raw_evidence", "facts", "secrets", "memory_fts", "memory_vec", "vec_map"]:
        assert t in names, f"{t} 缺失"


def test_images_table_created(tmp_path):
    engine = _mk(tmp_path)
    cols = {c["name"] for c in inspect(engine).get_columns("images")}
    for c in ["id", "document_id", "file_path", "page", "position", "ocr_text", "created_at"]:
        assert c in cols, f"{c} 缺失"


def test_fact_has_archive_flag(tmp_path):
    engine = _mk(tmp_path)
    cols = {c["name"] for c in inspect(engine).get_columns("facts")}
    assert "archived" in cols
    assert "superseded_by" in cols


def test_fact_json_roundtrip(tmp_path):
    engine = _mk(tmp_path)
    with Session(engine) as session:
        f = Fact(
            id="f1",
            type="item_location",
            summary="身份证放书房",
            fields={"place": "书房", "items": ["身份证"]},
            valid_from=datetime.now(timezone.utc),
        )
        session.add(f)
        session.commit()
        session.expire_all()
        loaded = session.get(Fact, "f1")
        assert loaded.fields == {"place": "书房", "items": ["身份证"]}


def test_document_and_chunk_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=4)
    with Session(engine) as s:
        s.add(Document(id="d1", kind="word", status="processing"))
        s.add(Chunk(id="c1", document_id="d1", seq=0, text="x", parent_text="x"))
        s.commit()
        assert s.query(Document).count() == 1
        assert s.query(Chunk).count() == 1


def test_embedding_dim_known_model():
    assert llm.embedding_dim("BAAI/bge-m3") == 1024
    assert llm.embedding_dim("BAAI/bge-small-zh-v1.5") == 512


# 原 test_embedding_dim_unknown_model_fallback（未知模型 → 加载它问维度）随 spec §3.4 删除：
# 表外模型现在直接抛 ValueError（不再为它加载权重），等价断言见 tests/test_llm_embed.py。


def test_init_db_dim_mismatch_raises(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    with pytest.raises(RuntimeError):
        init_db(engine, dim=1024)


def test_init_db_same_dim_no_raise(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    init_db(engine, dim=512)  # 同维度重复调用不应抛异常
