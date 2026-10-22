"""存量装饰小图清账（V2 口径）：只把「疑似装饰小图」的描述块从检索里撤下。

文件与 images 行**都保留**（原件永不丢）；命中的图标记 `decorative=1`
（此后不自动描述、不进可用图片清单、不参与补录）。判据与入库同一套（保守启发式）。

用法：
  python scripts/prune_decorative.py --data-dir ./data            # dry-run：只列清单
  python scripts/prune_decorative.py --data-dir ./data --apply    # 执行：标记 + 撤描述块
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.api.files import is_likely_decorative
from app.config import Settings
from app.core import llm
from app.memory import documents
from app.models.db import Chunk, Image, init_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="存量疑似装饰小图清账（只撤描述块；文件与行保留）")
    parser.add_argument("--data-dir", default="./data", help="数据目录（内含 mem.db）")
    parser.add_argument("--apply", action="store_true", help="实际执行（默认 dry-run 只列清单）")
    return parser.parse_args()


def _find_caption_chunk(session: Session, image: Image) -> Chunk | None:
    """该图的描述块：文本与入库时逐字一致 → 精确匹配（找不到 = 该图没进过索引）。"""
    if not image.caption:
        return None
    doc = documents.get_document(session, image.document_id)
    if doc is None:
        return None
    expected = f"图片描述（{documents.image_source(doc, image)}）：{image.caption}"
    return session.scalars(
        select(Chunk).where(Chunk.document_id == image.document_id, Chunk.text == expected)
    ).first()


def prune(session: Session, data_dir: Path, apply: bool) -> None:
    rows = session.scalars(select(Image).where(Image.decorative == 0)).all()
    candidates: list[Image] = []
    for img in rows:
        if not img.file_path:
            continue
        path = data_dir / img.file_path
        if not path.is_file():
            continue
        if is_likely_decorative(path.read_bytes()):
            candidates.append(img)

    print(f"扫描 {len(rows)} 张图，疑似装饰小图 {len(candidates)} 张：")
    for img in candidates:
        path = data_dir / img.file_path
        has_caption = "有" if img.caption else "无"
        print(
            f"  - {img.file_path}（{path.stat().st_size} 字节，"
            f"描述={has_caption}，文档={img.document_id[:8]}）"
        )
    if not apply:
        print("（dry-run：未做任何修改；核对后加 --apply 执行）")
        return

    marked = removed = 0
    for img in candidates:
        img.decorative = 1
        marked += 1
        chunk = _find_caption_chunk(session, img)
        if chunk is not None:
            session.execute(text("DELETE FROM chunk_fts WHERE chunk_id = :c"), {"c": chunk.id})
            if chunk.vec_rowid is not None:
                session.execute(
                    text("DELETE FROM chunk_vec WHERE rowid = :r"), {"r": chunk.vec_rowid}
                )
            session.delete(chunk)
            removed += 1
        session.commit()
    print(f"完成：标记 decorative=1 共 {marked} 张，撤下描述块 {removed} 个（文件与行均保留）")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    settings = Settings(data_dir=data_dir)
    engine = create_engine(
        f"sqlite:///{data_dir / 'mem.db'}", connect_args={"check_same_thread": False}
    )
    init_db(engine, llm.embedding_dim(settings.embedding_model))
    with Session(engine) as session:
        prune(session, data_dir, args.apply)


if __name__ == "__main__":
    main()
