"""F5 覆盖洞③：reembed 重建材料索引（material_fts / material_vec / materials.vec_rowid）。"""
import sys

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.core import llm
from app.memory import documents, materials
from app.models.db import Material, init_db
from scripts import reembed


def test_reembed_rebuilds_material_index(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=1024)
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 1024)

    with Session(engine) as s:
        doc = documents.insert_document(
            s, kind="word", status="done", title="论文", parsed_text="正文"
        )
        s.flush()
        materials.upsert_for_document(s, doc, model="BAAI/bge-m3")
        s.commit()
        assert s.scalars(select(Material)).one().vec_rowid is not None

    # 清掉材料索引（模拟索引丢失/旧库）→ reembed 应整体重建
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM material_fts"))
        conn.execute(text("DELETE FROM material_vec"))
        conn.execute(text("UPDATE materials SET vec_rowid = NULL"))

    monkeypatch.setattr(sys, "argv", ["reembed.py", "--data-dir", str(tmp_path)])
    reembed.main()

    with Session(engine) as s:
        assert s.scalars(select(Material)).one().vec_rowid is not None
        assert s.execute(text("SELECT count(*) FROM material_fts")).scalar() == 1
        assert s.execute(text("SELECT count(*) FROM material_vec")).scalar() == 1
