"""MATERIAL 材料层：每份材料（记录/文档各一行）一条可检索索引，入库时生成。

摘要规则已决（spec §4）：勾选可理解的文档复用云端 summary / 图片 caption；
其余抽取式兜底（本地生成、不调模型）——保证每份材料都有条目，检索质量分层如实接受。
"""
import json
import logging

import sqlite_vec
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core import llm
from app.memory import qa_history
from app.memory.fts import fts_query
from app.memory.store import new_id
from app.models.db import DocTable, Document, Image, Material, RawEvidence

logger = logging.getLogger(__name__)

SUMMARY_MAX = 200  # 材料摘要统一截断长度（spec §4）
TITLE_MAX = 40  # 记录标题兜底取原话开头的前 40 字


def index_material(session: Session, material_id: str, vec: list[float], content: str) -> None:
    """写 material_vec/material_fts 索引行并回填 `vec_rowid`（照 chunks.index_chunk 的模式）。"""
    session.execute(
        text("INSERT INTO material_vec(embedding) VALUES (:vec)"),
        {"vec": sqlite_vec.serialize_float32(vec)},
    )
    rowid = session.execute(text("SELECT last_insert_rowid()")).scalar()
    session.execute(
        text("UPDATE materials SET vec_rowid = :r WHERE id = :id"),
        {"r": rowid, "id": material_id},
    )
    session.execute(
        text("INSERT INTO material_fts(content, material_id) VALUES (:c, :id)"),
        {"c": content, "id": material_id},
    )


def upsert_for_record(session: Session, raw: RawEvidence, *, title: str | None, model: str) -> Material:
    """记录材料：title 一般由调用方传 fact 摘要，兜底原话前 40 字；summary = 原话前 200 字。"""
    title = title or (raw.content or "")[:TITLE_MAX] or "记录"
    return _upsert(session, "record", raw.id, title, (raw.content or "")[:SUMMARY_MAX], model)


def _table_index_text(session: Session, doc: Document, limit: int = SUMMARY_MAX) -> str:
    """结构化表的索引文本：表标题 + 表头 + 首行样本（供材料摘要）。

    表格原文不进正文索引（`structify_tables` 已把正文里的表换成一行摘要），列名与取值
    只能靠这里进材料检索——成绩表/岗位表这类「内容全在表里」的文档，当云总结缺失或被
    质量门丢弃时（`cloud.summarize` 的拒绝文本判定），这是唯一带信号的材料文本。
    """
    tables = session.scalars(
        select(DocTable).where(DocTable.document_id == doc.id).order_by(DocTable.id)
    ).all()
    parts: list[str] = []
    for table in tables:
        try:
            headers = [h for h in json.loads(table.headers) if h]
            rows = json.loads(table.rows)
        except Exception:
            continue  # 解析损坏的表：跳过该表，不影响其余
        prefix = f"{table.heading}：" if table.heading else ""
        sample = " ".join(c for c in (rows[0] if rows else []) if c)
        parts.append(f"{prefix}表格 {len(rows)} 行，表头：{'、'.join(headers)}；首行：{sample}")
    return "；".join(parts)[:limit]


def _table_values_text(session: Session, doc: Document, limit: int) -> str:
    """表的取值文本（表标题+表头+首行样本，见 `_table_index_text`），供材料摘要前置。"""
    return _table_index_text(session, doc, limit=limit)


def upsert_for_document(session: Session, doc: Document, *, model: str) -> Material:
    """文档材料：**结构化表的取值文本前置** + 摘要类文本（doc.summary → caption → 正文前 200 字 → 标题）。

    表类文档的云摘要常是「共 N 行×M 列」式空话甚至抱怨文本，真正可检索的值（公司名/班名/列名）
    只在表里——材料摘要必须带上表的取值，否则按值提问根本命中不了（v3 实测：样例甲/样例乙两题抓错表）。
    """
    image = session.scalars(
        select(Image).where(Image.document_id == doc.id).order_by(Image.created_at).limit(1)
    ).first()
    caption = image.caption if image else None
    prose = doc.summary or caption or (doc.parsed_text or "")[:SUMMARY_MAX] or doc.title
    values = _table_values_text(session, doc, limit=120)
    if values and values not in (prose or ""):
        summary = f"{values}｜{prose}"[:SUMMARY_MAX]
    else:
        summary = (prose or doc.title)[:SUMMARY_MAX]
    material = _upsert(session, "document", doc.id, doc.title, summary, model)
    # 文档侧材料在分块/表格与云总结之后的**更晚事务**里才落库（api/files.py）：不 bump 的话
    # 窗口内提问的缓存行会以「已是最新版本」存活——把「没找到」缓存住。record 侧与 insert_fact
    # 同事务（已覆盖），不重复 bump。
    qa_history.bump(session)
    return material


def _upsert(session: Session, source_type: str, source_id: str,
            title: str | None, summary: str | None, model: str) -> Material:
    """幂等：同一 (source_type, source_id) 恒一行；已存在则先删旧索引行再重建（重嵌入）。"""
    material = session.scalars(
        select(Material).where(Material.source_type == source_type, Material.source_id == source_id)
    ).first()
    if material is None:
        material = Material(id=new_id(), source_type=source_type, source_id=source_id)
        session.add(material)
    material.title = title
    material.summary = summary
    _clear_index(session, material.id)
    content = f"{title or ''} {summary or ''}".strip()
    index_material(session, material.id, llm.embed(content, model=model), content)
    return material


def _clear_index(session: Session, material_id: str) -> None:
    """幂等重建：删旧 FTS/向量行。vec_rowid 以库内现值为准（会话属性可能过期/未回填）。"""
    session.execute(text("DELETE FROM material_fts WHERE material_id = :id"), {"id": material_id})
    rowid = session.execute(
        text("SELECT vec_rowid FROM materials WHERE id = :id"), {"id": material_id}
    ).scalar()
    if rowid is not None:
        session.execute(text("DELETE FROM material_vec WHERE rowid = :r"), {"r": rowid})


# 可见性（检索两路共用）：显式两路 OR——document 源要求 documents 行存在且未软删；
# record 源要求 raw_evidence 行存在且未软删。边界：源对象被彻底删除（purge）而材料行残留
# ——理论上不应发生（purge 已清账），真遇到则查不到（源行不存在），不报错。
_VISIBLE = (
    "((m.source_type = 'document' AND d.id IS NOT NULL AND d.deleted_at IS NULL) "
    "OR (m.source_type = 'record' AND r.id IS NOT NULL AND r.deleted_at IS NULL))"
)


def vector_search(session: Session, vec: list[float], k: int = 10) -> list[str]:
    """material_vec 向量检索，返回材料 id（可见性过滤见上方两路 OR 注）。"""
    rows = session.execute(
        text(
            "SELECT m.id FROM material_vec v JOIN materials m ON m.vec_rowid = v.rowid "
            "LEFT JOIN documents d ON d.id = m.source_id "
            "LEFT JOIN raw_evidence r ON r.id = m.source_id "
            f"WHERE v.embedding MATCH :vec AND k = :k AND {_VISIBLE} "
            "ORDER BY v.distance"
        ),
        {"vec": sqlite_vec.serialize_float32(vec), "k": k},
    ).fetchall()
    return [r[0] for r in rows]


def fts_search(session: Session, q: str, k: int = 10) -> list[str]:
    """material_fts 关键词检索，返回材料 id（可见性过滤见上方两路 OR 注）。

    F4 备忘：trigram 分词器对 **2 字**中文查询零命中（「成绩」不行、「身份证」可以），
    短标题/摘要只剩向量路——M3 后按实测再定要不要调优（详见开发日志会话 23）。
    """
    try:
        # material_fts 保持原名不别名：FTS5 的 MATCH 语法要求左操作数是 fts 表本身
        rows = session.execute(
            text(
                "SELECT material_fts.material_id FROM material_fts "
                "JOIN materials m ON m.id = material_fts.material_id "
                "LEFT JOIN documents d ON d.id = m.source_id "
                "LEFT JOIN raw_evidence r ON r.id = m.source_id "
                f"WHERE material_fts MATCH :q AND {_VISIBLE} "
                "ORDER BY rank LIMIT :k"
            ),
            {"q": fts_query(q), "k": k},
        ).fetchall()
    except Exception:
        # FTS 失败只降级为无关键词命中，不打断向量检索主链路
        logger.warning("material_fts 查询失败（已降级为纯向量检索）", exc_info=True)
        return []
    return [r[0] for r in rows]
