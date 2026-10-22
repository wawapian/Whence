"""向量索引重建脚本：embedding 维度变更后一次性迁移。

用法：`uv run python scripts/reembed.py --data-dir ./data`

行为：
1. 连接 `{data_dir}/mem.db` 并加载 sqlite-vec 扩展。
2. 删除旧 `memory_vec` + `vec_map`，按当前默认模型维度重建 `memory_vec` 与 `vec_map`；
   同样删除并重建 `chunk_fts`/`chunk_vec` 与 `material_fts`/`material_vec`。
3. 遍历所有 Fact，镜像 `app/api/memories.py` 重建索引文本（summary + raw 正文前 2000 字，
   embed 取前 500 字），用当前模型重新嵌入并写回 `memory_vec` / `vec_map`。
4. 遍历所有 Chunk / Material，用当前模型重新嵌入并写回各自索引与 `vec_rowid`。

注意：本脚本会重建 Fact 向量索引、Chunk 索引（chunk_vec + chunk_fts）与
材料索引（material_vec + material_fts + `materials.vec_rowid`）。
运行时会经 llama-server 计算向量——**服务须先起**（见 `start_llama.bat` / `ops/`），
否则整脚本以 LlamaUnavailableError 终止。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite_vec
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import Session

from app.core import llm
from app.memory import chunks, materials, vector
from app.models.db import Base, Chunk, Fact, Material, RawEvidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按当前 embedding 模型重建全部向量/检索索引（Fact/Chunk/材料）")
    parser.add_argument("--data-dir", required=True, help="数据目录（内含 mem.db）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    db_path = data_dir / "mem.db"

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _setup(dbapi_conn, _record):
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    dim = llm.embedding_dim(llm.DEFAULT_EMBEDDING_MODEL)

    # 丢弃旧向量表 + 映射表，按当前模型维度重建
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS memory_vec"))
        conn.execute(text("DROP TABLE IF EXISTS vec_map"))
        conn.execute(text(f"CREATE VIRTUAL TABLE memory_vec USING vec0(embedding float[{dim}])"))
        conn.execute(text("DROP TABLE IF EXISTS chunk_fts"))
        conn.execute(text("DROP TABLE IF EXISTS chunk_vec"))
        conn.execute(text("CREATE VIRTUAL TABLE chunk_fts USING fts5(content, chunk_id UNINDEXED, tokenize='trigram')"))
        conn.execute(text(f"CREATE VIRTUAL TABLE chunk_vec USING vec0(embedding float[{dim}])"))
        conn.execute(text("DROP TABLE IF EXISTS material_fts"))
        conn.execute(text("DROP TABLE IF EXISTS material_vec"))
        conn.execute(text("CREATE VIRTUAL TABLE material_fts USING fts5(content, material_id UNINDEXED, tokenize='trigram')"))
        conn.execute(text(f"CREATE VIRTUAL TABLE material_vec USING vec0(embedding float[{dim}])"))
    Base.metadata.create_all(engine)  # 重建常规表 vec_map

    # 遍历所有 Fact 重建向量索引（镜像 app/api/memories.py 的索引文本构造）
    with Session(engine) as session:
        facts = session.scalars(select(Fact)).all()
        for fact in facts:
            raw = session.get(RawEvidence, fact.raw_id) if fact.raw_id else None
            if raw is not None and raw.content is not None:
                content = f"{fact.summary} {raw.content[:2000]}".strip()
            else:
                content = (fact.summary or "").strip()
            vector.index_fact(session, fact.id, llm.embed(content[:500]))
        session.commit()
        print(f"重建完成：{len(facts)} 条 Fact 已按 {llm.DEFAULT_EMBEDDING_MODEL}(dim={dim}) 重新写入向量索引")

    # 遍历所有 Chunk 回填 chunk 索引（chunk_vec + chunk_fts）
    with Session(engine) as session:
        all_chunks = session.scalars(select(Chunk)).all()
        for c in all_chunks:
            chunks.index_chunk(session, c.id, llm.embed(c.text), c.text)
        session.commit()
        print(f"重建完成：{len(all_chunks)} 条 Chunk 已按 {llm.DEFAULT_EMBEDDING_MODEL}(dim={dim}) 重新写入 chunk 索引")

    # 遍历所有 Material 回填材料索引（material_vec + material_fts + materials.vec_rowid）
    with Session(engine) as session:
        all_materials = session.scalars(select(Material)).all()
        for m in all_materials:
            content = f"{m.title or ''} {m.summary or ''}".strip()  # 镜像 materials._upsert 的索引文本构造
            materials.index_material(session, m.id, llm.embed(content), content)
        session.commit()
        print(f"重建完成：{len(all_materials)} 条材料已按 {llm.DEFAULT_EMBEDDING_MODEL}(dim={dim}) 重新写入 material 索引")


if __name__ == "__main__":
    main()
