"""存量回填：给 M2 之前入库、尚无材料行的记录/文档补建材料索引（幂等）。

用法：`uv run python scripts/materials_backfill.py --data-dir ./data`

背景（终审 F1）：M2 之前入库的档案没有 materials 行，材料层检索看不到它们；
新问答（qa_engine=tool_loop）走「先找材料」，不跑本脚本会导致存量档案在问答里失明。

规则（镜像入库时的语义）：
- 文档：全部未软删文档 → `upsert_for_document`；
- 记录：仅「有 fact」的未软删记录 → `upsert_for_record`（与记录接口一致：raw_only 不建材料）；
- 已软删的一律不回填（回收站恢复的存量条目如需材料，重跑本脚本）。
幂等：`materials._upsert` 同 (source_type, source_id) 恒一行，重复跑安全。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import llm
from app.memory import materials
from app.models.db import Document, Fact, RawEvidence, init_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为存量记录/文档补建材料层索引（幂等）")
    parser.add_argument("--data-dir", default="./data", help="数据目录（内含 mem.db）")
    return parser.parse_args()


def backfill(session: Session, model: str) -> tuple[int, int]:
    """回填所有未软删文档 + 有 fact 的未软删记录；返回 (记录数, 文档数)。"""
    docs = session.scalars(select(Document).where(Document.deleted_at.is_(None))).all()
    for doc in docs:
        materials.upsert_for_document(session, doc, model=model)

    raws = session.scalars(
        select(RawEvidence)
        .where(RawEvidence.deleted_at.is_(None))
        .where(RawEvidence.id.in_(select(Fact.raw_id).where(Fact.raw_id.is_not(None))))
    ).all()
    for raw in raws:
        latest = session.scalars(
            select(Fact).where(Fact.raw_id == raw.id).order_by(Fact.created_at.desc()).limit(1)
        ).first()
        materials.upsert_for_record(
            session, raw, title=latest.summary if latest else None, model=model
        )

    session.commit()
    return len(raws), len(docs)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    settings = Settings(data_dir=data_dir)
    engine = create_engine(
        f"sqlite:///{data_dir / 'mem.db'}", connect_args={"check_same_thread": False}
    )
    init_db(engine, llm.embedding_dim(settings.embedding_model))
    with Session(engine) as session:
        n_records, n_docs = backfill(session, settings.embedding_model)
    print(
        f"回填完成：{n_records} 条记录、{n_docs} 份文档已写入材料层"
        f"（模型 {settings.embedding_model}）"
    )


if __name__ == "__main__":
    main()
