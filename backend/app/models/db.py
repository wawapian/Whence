"""数据模型与建表。"""
import logging
import re
from datetime import datetime, timezone

import sqlite_vec
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    Text,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RawEvidence(Base):
    """RAW 原始证据层：原文/原图/原文档，永不改删。"""

    __tablename__ = "raw_evidence"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="me")
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # text / photo / document
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 记录整理标记：0=待整理（accept-fast 后由后台整理）1=已整理 NULL=存量旧数据（不回填不重投）
    consolidated: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Fact(Base):
    """FACT 事实台账层：结构化事实，双时态。"""

    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="me")
    subject_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    fields: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    state: Mapped[str] = mapped_column(Text, nullable=False, default="confirmed")
    archived: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(Text, ForeignKey("facts.id"), nullable=True)
    raw_id: Mapped[str | None] = mapped_column(Text, ForeignKey("raw_evidence.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class Secret(Base):
    """敏感值映射表（本地）：占位符 -> 真实值。"""

    __tablename__ = "secrets"

    placeholder: Mapped[str] = mapped_column(Text, primary_key=True)  # [[SECRET_1]]
    value: Mapped[str] = mapped_column(Text, nullable=False)
    fact_id: Mapped[str | None] = mapped_column(Text, nullable=True)


class VecMap(Base):
    """向量 rowid -> fact_id 映射。"""

    __tablename__ = "vec_map"

    vec_rowid: Mapped[int] = mapped_column(Integer, primary_key=True)
    fact_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)


class Document(Base):
    """DOCUMENT 文档解析层：解析后的文档。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="me")
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # word / pdf / txt / ...
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_id: Mapped[str | None] = mapped_column(Text, nullable=True)  # 上游 raw_evidence.id（轻量迁移列，不做 FK）
    parsed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    allow_cloud: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    multimodal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ocr_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    engine_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    needs_ocr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="processing")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Image(Base):
    """IMAGES 图片理解层：文档内嵌图片及 OCR 文本。"""

    __tablename__ = "images"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    document_id: Mapped[str] = mapped_column(
        Text, ForeignKey("documents.id"), nullable=False
    )
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 1 = 疑似装饰小图（只降级：不自动描述/不进可用图片清单；文件与行永久保留）
    decorative: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class Chunk(Base):
    """CHUNK 分块层：文档分块文本与向量映射。"""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    document_id: Mapped[str] = mapped_column(
        Text, ForeignKey("documents.id"), nullable=False
    )
    # 建立依赖顺序：flush 时先插 Document 再插 Chunk，避免外键约束报错
    document: Mapped["Document"] = relationship()
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parent_text: Mapped[str] = mapped_column(Text, nullable=False)
    vec_rowid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class Material(Base):
    """MATERIAL 材料层：每份材料（记录/文档各一行）一条可检索索引，入库时生成。"""

    __tablename__ = "materials"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    source_type: Mapped[str] = mapped_column(Text, nullable=False)  # record / document
    source_id: Mapped[str] = mapped_column(Text, nullable=False)  # raw_evidence.id / documents.id
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    vec_rowid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class DocTable(Base):
    """DOC_TABLE 文档表格层：表格全量行数据（headers/rows 为 JSON 字符串）。"""

    __tablename__ = "doc_tables"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    document_id: Mapped[str] = mapped_column(
        Text, ForeignKey("documents.id"), nullable=False
    )
    # 建立依赖顺序：flush 时先插 Document 再插 DocTable，避免外键约束报错
    document: Mapped["Document"] = relationship()
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    headers: Mapped[str] = mapped_column(Text, nullable=False)
    rows: Mapped[str] = mapped_column(Text, nullable=False)
    suspect: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class QaHistory(Base):
    """QA_HISTORY 问答历史：每次提问落一行（缓存命中/未命中都落；作废 ≠ 删除）。"""

    __tablename__ = "qa_history"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid4
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)  # 用户问的原文
    resolved_question: Mapped[str | None] = mapped_column(Text, nullable=True)  # 无 history 恒 NULL
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str] = mapped_column(Text, nullable=False)  # JSON 字符串
    documents: Mapped[str] = mapped_column(Text, nullable=False)  # JSON 字符串
    question_vec: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # float32 BLOB
    store_version: Mapped[int] = mapped_column(Integer, nullable=False)  # 写入时的记忆库版本号
    from_cache: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0/1
    execution: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class StoreMeta(Base):
    """STORE_META 全局键值元数据：当前仅 store_version（记忆库版本号）。"""

    __tablename__ = "store_meta"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


def init_db(engine, dim: int = 512) -> None:
    """建常规表 + 虚拟表（FTS5 / vec0），加载 sqlite-vec 扩展并开启外键。"""

    @event.listens_for(engine, "connect")
    def _setup(dbapi_conn, _record):
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(qa_history)"))}
        if cols and "execution" not in cols:
            conn.execute(text("ALTER TABLE qa_history ADD COLUMN execution TEXT NOT NULL DEFAULT '{}'"))
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(images)"))}
        if cols and "caption" not in cols:
            try:
                conn.execute(text("ALTER TABLE images ADD COLUMN caption TEXT"))
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(images)"))}
        if cols and "decorative" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE images ADD COLUMN decorative INTEGER NOT NULL DEFAULT 0")
                )
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(doc_tables)"))}
        if cols and "suspect" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE doc_tables ADD COLUMN suspect INTEGER NOT NULL DEFAULT 0")
                )
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(documents)"))}
        if cols and "deleted_at" not in cols:
            try:
                conn.execute(text("ALTER TABLE documents ADD COLUMN deleted_at DATETIME"))
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(documents)"))}
        if cols and "raw_id" not in cols:
            try:
                conn.execute(text("ALTER TABLE documents ADD COLUMN raw_id TEXT"))
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
            # 存量回填：同 file_path 配对，取 created_at 最早的 raw；
            # 放在「列缺失」块内，第二次 init_db 走「列已存在」分支，天然不重复回填
            conn.execute(text(
                "UPDATE documents SET raw_id = (SELECT r.id FROM raw_evidence r "
                "WHERE r.file_path = documents.file_path ORDER BY r.created_at LIMIT 1) "
                "WHERE raw_id IS NULL AND file_path IS NOT NULL"
            ))
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(raw_evidence)"))}
        if cols and "deleted_at" not in cols:
            try:
                conn.execute(text("ALTER TABLE raw_evidence ADD COLUMN deleted_at DATETIME"))
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(raw_evidence)"))}
        if cols and "consolidated" not in cols:
            try:
                conn.execute(text("ALTER TABLE raw_evidence ADD COLUMN consolidated INTEGER"))
            except Exception:
                # 并发进程可能刚刚已补列：忽略重复列错误，保持迁移幂等
                pass
        # 向量检索 join 键补索引（幂等迁移）：`JOIN chunks c ON c.vec_rowid = v.rowid`
        # 在 vec_rowid 无索引时，SQLite 对每个 KNN 命中行全表扫 chunks（生产库 1241 行含
        # 长 text 列，3.9s/次；成本 ≈O(k×N)，随语料线性放大）。有索引后计划
        # `SCAN c` → `SEARCH c USING INDEX ix_chunks_vec_rowid`，10ms 级。materials 同形状
        # （现在行少不显，语料一大即同一炸弹）。既有库下次启动经 init_db 自动补建。
        for table, index in (("chunks", "ix_chunks_vec_rowid"), ("materials", "ix_materials_vec_rowid")):
            try:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index} ON {table}(vec_rowid)"))
            except Exception:
                # 重复建（并发进程抢先）可忽略；其余失败（缺列/库锁等）不能再静默——
                # 建索引失败 = 性能修复无声回退（终审 Minor#3），留 warning 点名索引
                logger.warning("建索引 %s 失败（已跳过）", index, exc_info=True)
        conn.execute(
            text("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content, fact_id UNINDEXED, tokenize='trigram')")
        )
        conn.execute(
            text(f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{dim}])")
        )
        conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(content, chunk_id UNINDEXED, tokenize='trigram')"))
        conn.execute(text(f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(embedding float[{dim}])"))
        conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS material_fts USING fts5(content, material_id UNINDEXED, tokenize='trigram')"))
        conn.execute(text(f"CREATE VIRTUAL TABLE IF NOT EXISTS material_vec USING vec0(embedding float[{dim}])"))
        # 维度护栏：存量库的 vec0 表按旧模型维度建过就不能再开（查询会错配）；
        # 三张 vec 表逐一校验（memory_vec/chunk_vec/material_vec，reembed.py 负责重建）
        for table in ("memory_vec", "chunk_vec", "material_vec"):
            stored_sql = conn.execute(
                text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table},
            ).scalar()
            if not stored_sql:
                continue
            m = re.search(r"float\[(\d+)\]", stored_sql)
            if m and int(m.group(1)) != dim:
                raise RuntimeError(
                    f"{table} 向量维度不匹配：表内为 {int(m.group(1))}d，当前 embedding 模型输出 {dim}d。"
                    f"请先运行 `uv run python scripts/reembed.py --data-dir <DATA_DIR>` 重建向量表。"
                )
