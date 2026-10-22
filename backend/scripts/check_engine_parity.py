"""跨引擎向量一致性检查：库里既有的向量（旧 torch 版 bge-m3）vs 现算的向量（llama-server）。

用途（spec §6-V1 / §7-R2）：
- 「重嵌是否必要」的实证——余弦越高，换引擎对既有索引的扰动越小；
- 启动参数（--embeddings / pooling）配错的早期信号——余弦异常低就先查服务参数，
  而不是先怀疑数据。

只读：不写库、不 bump 版本号。
用法：.venv\\Scripts\\python -m scripts.check_engine_parity --data-dir ./data-reembed [--limit 20]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sqlite_vec
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from app.core import llm
from app.memory.store import fact_search_text

_FACT_SQL = """
SELECT m.fact_id, v.embedding
FROM memory_vec v JOIN vec_map m ON m.vec_rowid = v.rowid
LIMIT :limit
"""
_CHUNK_SQL = """
SELECT c.id, c.text, v.embedding
FROM chunk_vec v JOIN chunks c ON c.vec_rowid = v.rowid
WHERE c.text IS NOT NULL AND c.text != ''
LIMIT :limit
"""


def _engine(data_dir: Path):
    engine = create_engine(f"sqlite:///{data_dir / 'mem.db'}")

    @event.listens_for(engine, "connect")
    def _setup(dbapi_conn, _record):
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.enable_load_extension(False)

    return engine


def _cosine(a: bytes, b: list[float]) -> float:
    old = np.frombuffer(a, dtype=np.float32)
    new = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(old) * np.linalg.norm(new))
    return float(np.dot(old, new) / denom) if denom else 0.0


def _collect(engine, data_dir: Path, limit: int, source: str) -> list[tuple[str, str, str, bytes]]:
    """返回 [(类别, id, 当时嵌入的文本, 旧向量)]。"""
    rows: list[tuple[str, str, str, bytes]] = []
    with engine.connect() as conn:
        if source in ("facts", "both"):
            for fact_id, blob in conn.execute(text(_FACT_SQL), {"limit": limit}).all():
                with Session(engine) as session:
                    # 与写入侧同源：summary + 正文前 2000 字，嵌入取前 500 字（api/memories.py）
                    text_ = fact_search_text(session, fact_id)[:500]
                rows.append(("fact", fact_id, text_, blob))
        if source in ("chunks", "both"):
            for chunk_id, chunk_text, blob in conn.execute(text(_CHUNK_SQL), {"limit": limit}).all():
                rows.append(("chunk", chunk_id, chunk_text, blob))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="跨引擎向量一致性（旧索引 vs llama-server）")
    parser.add_argument("--data-dir", default="./data", help="库所在目录（建议用副本）")
    parser.add_argument("--limit", type=int, default=20, help="每类取样条数")
    parser.add_argument("--source", choices=["facts", "chunks", "both"], default="both")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "mem.db").exists():
        print(f"找不到 {data_dir / 'mem.db'}", file=sys.stderr)
        return 2

    rows = _collect(_engine(data_dir), data_dir, args.limit, args.source)
    if not rows:
        print("没取到样本（库里没有向量？）")
        return 1

    scores: list[float] = []
    failures: list[tuple[str, str]] = []
    for kind, item_id, text_, blob in rows:
        try:
            new = llm.embed(text_)
        except Exception as exc:  # 单条坏输入不该掩掉整批样本——如实报出并继续
            failures.append((item_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{kind}] {item_id}  ⚠️ 嵌入失败（{type(exc).__name__}）：{exc}", file=sys.stderr)
            continue
        cos = _cosine(blob, new)
        scores.append(cos)
        print(f"[{kind}] {item_id}  余弦={cos:.6f}  文本={text_[:30]!r}")

    if not scores:
        print("全部样本嵌入失败——先查 llama-server 是否在跑。", file=sys.stderr)
        return 1

    arr = np.asarray(scores)
    print(
        f"\n成功 {len(arr)} 条（失败 {len(failures)} 条）：最小 {arr.min():.6f} | "
        f"中位 {np.median(arr):.6f} | 均值 {arr.mean():.6f}"
    )
    if arr.min() < 0.9:
        print("⚠️ 最小值偏低——先查 llama-server 启动参数（--embeddings / pooling），别急着怪数据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
