"""万级规模冒烟：2 万条材料（照真实写入路径直插）下，目标材料须进检索 top-5；整测防病态 < 30s。"""
import time

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core import llm, retrieval
from app.memory import materials
from app.memory.store import new_id
from app.models.db import init_db

DIM = 512
N = 20000
TARGET = 12345          # 目标混在中段，防顺序侥幸
TARGET_WORD = "紫狸猫"  # 独特词：仅目标材料的 FTS 内容含它

# 独特向量：查询向量 = 目标向量（距离 0），其余材料向量与查询正交（距离更远）
_QUERY_VEC = [1.0] + [0.0] * (DIM - 1)


def _other_vec(i: int) -> list[float]:
    return [0.0, ((i % 7) + 1) / 10.0] + [0.0] * (DIM - 2)


def test_material_scale_smoke(tmp_path, monkeypatch):
    started = time.perf_counter()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: _QUERY_VEC)  # 查询向量替身，不调真模型

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=DIM)
    s = Session(engine)

    # 直插 2 万条 raw（满足材料可见性：record 源要求 raw 行存在且未软删）+ materials 行
    raws, mats, content_by_i = [], [], []
    for i in range(N):
        raw_id, material_id = new_id(), new_id()
        content = f"材料{i} 内容词{i % 100}"
        if i == TARGET:
            content = f"{content} {TARGET_WORD}"
        raws.append({"id": raw_id, "kind": "text", "content": content})
        mats.append({"id": material_id, "sid": raw_id, "title": f"材料{i}", "summary": content})
        content_by_i.append((material_id, content))
    s.execute(
        text("INSERT INTO raw_evidence(id, user_id, kind, content, created_at) "
             "VALUES (:id, 'me', :kind, :content, datetime('now'))"),
        raws,
    )
    s.execute(
        text("INSERT INTO materials(id, source_type, source_id, title, summary, created_at) "
             "VALUES (:id, 'record', :sid, :title, :summary, datetime('now'))"),
        mats,
    )
    # 索引行照 materials.index_material 真实写入路径（material_vec + vec_rowid 回填 + material_fts）
    for i, (material_id, content) in enumerate(content_by_i):
        materials.index_material(
            s, material_id, _QUERY_VEC if i == TARGET else _other_vec(i), content
        )
    s.commit()
    target_id = content_by_i[TARGET][0]

    t0 = time.perf_counter()
    results = retrieval.search_materials_multi(s, [TARGET_WORD], top_k=5)
    search_elapsed = time.perf_counter() - t0
    total_elapsed = time.perf_counter() - started

    assert target_id in results  # 2 万条材料里目标进 top-5
    # 耗时只打印、不 hard assert；ASCII 输出（重定向到文件时不受控制台编码影响）
    print(f"[scale] materials={N} search={search_elapsed * 1000:.0f}ms total={total_elapsed:.1f}s")
    assert total_elapsed < 30, f"规模冒烟病态慢：{total_elapsed:.1f}s"
