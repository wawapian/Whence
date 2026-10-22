"""DOCUMENT / CHUNK 读写。"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory import qa_history
from app.memory.store import new_id
from app.models.db import Chunk, DocTable, Document, Image
from app.models.db import RawEvidence
import re
import unicodedata


def normalized_name(value: str) -> str:
    return re.sub(r"[\s_\-：:《》〈〉]+", "", unicodedata.normalize("NFKC", value or "")).casefold()


def named_documents(session: Session, question: str) -> list[Document]:
    """Literal document identity wins over semantic neighbours; never pick a deleted copy."""
    q = normalized_name(question)
    return [d for d in list_documents(session) if d.title and normalized_name(d.title) in q]


def file_terms(keyword: str) -> list[str]:
    text = re.sub(r"帮我|请|查找|寻找|找到|找出|找一下|找一份|找个|找|查看|打开|我之前上传的|我上传的|我的|之前|那个|这份|一下", " ", keyword)
    text = re.sub(r"超市|小票|收据|发票|原件|文件|文档|照片|图片|截图|合同|论文|报告|便签|的", " ", text)
    return [t.strip() for t in re.split(r"[\s，。！？?、]+", text) if len(t.strip()) >= 2]


def find_files(session: Session, keyword: str | None = None, limit: int = 20) -> list[Document]:
    """Find originals by literal filename, OCR/body, caption summary and attached user words.

    Named entities must match. Generic words such as '小票' do not pull unrelated papers.
    No cloud or vector model is required for this local evidence lookup.
    """
    docs = list_documents(session)
    if not keyword:
        return docs[:limit]
    exact = named_documents(session, keyword)
    if exact:
        return exact[:limit]
    terms = file_terms(keyword) or [keyword.strip()]
    raw_ids = [d.raw_id for d in docs if d.raw_id]
    raws = {r.id: r.content or "" for r in session.scalars(select(RawEvidence).where(
        RawEvidence.id.in_(raw_ids), RawEvidence.deleted_at.is_(None)))} if raw_ids else {}
    matches = []
    for d in docs:
        fields = [d.title or "", d.summary or "", d.parsed_text or "", raws.get(d.raw_id, "")]
        body = normalized_name(" ".join(fields))
        if all(normalized_name(term) in body for term in terms):
            score = sum(4 if normalized_name(term) in normalized_name(d.title or "") else 1 for term in terms)
            matches.append((score, d))
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [d for _, d in matches[:limit]]


def insert_document(session: Session, **fields) -> Document:
    """新增 Document（id 自动生成，status 默认 processing），返回已 add 的对象。"""
    fields.setdefault("status", "processing")
    doc = Document(id=new_id(), **fields)
    session.add(doc)
    qa_history.bump(session)
    return doc


def insert_chunk(session: Session, **fields) -> Chunk:
    """新增 Chunk（id 自动生成），返回已 add 的对象。"""
    chunk = Chunk(id=new_id(), **fields)
    session.add(chunk)
    qa_history.bump(session)
    return chunk


def insert_image(session: Session, **fields) -> Image:
    """新增 Image（id 自动生成），返回已 add 的对象。"""
    image = Image(id=new_id(), **fields)
    session.add(image)
    qa_history.bump(session)
    return image


def insert_table(session: Session, suspect: bool = False, **fields) -> DocTable:
    """新增 DocTable（id 自动生成，suspect 标记转 0/1），返回已 add 的对象。"""
    table = DocTable(id=new_id(), suspect=int(suspect), **fields)
    session.add(table)
    qa_history.bump(session)
    return table


def get_document(session: Session, id: str) -> Document | None:
    """按主键读取 Document，不存在返回 None。"""
    return session.get(Document, id)


def list_documents(session: Session) -> list[Document]:
    """全部未删除文档（文档列表读入口），按 created_at 倒序。"""
    return list(
        session.scalars(
            select(Document)
            .where(Document.deleted_at.is_(None))
            .order_by(Document.created_at.desc())
        ).all()
    )


def visible_doc_tables(session: Session, doc_ids: list[str]) -> list[DocTable]:
    """doc_ids 中未删除文档的表格，按 created_at/id 升序，最多 5 张（统计路径候选）。"""
    if not doc_ids:
        return []
    return list(
        session.scalars(
            select(DocTable)
            .join(Document, DocTable.document_id == Document.id)
            .where(Document.deleted_at.is_(None), DocTable.document_id.in_(doc_ids))
            .order_by(DocTable.created_at, DocTable.id)
            .limit(5)
        ).all()
    )


def visible_images(session: Session, doc_ids: list[str]) -> list[Image]:
    """doc_ids 中未删除文档的图片；疑似装饰小图不进可用图片集（只降级，行仍在库）。

    排序：有页码的按页码在前（"图 N" ↔ "第 N 页"可对应），无页码的后置、按采集顺序。
    """
    if not doc_ids:
        return []
    return list(
        session.scalars(
            select(Image)
            .join(Document, Image.document_id == Document.id)
            .where(
                Document.deleted_at.is_(None),
                Image.document_id.in_(doc_ids),
                Image.decorative == 0,
            )
            .order_by(Image.page.is_(None), Image.page, Image.created_at)
        ).all()
    )


def get_chunk(session: Session, chunk_id: str) -> Chunk | None:
    """按主键读取 Chunk，不存在返回 None。"""
    return session.get(Chunk, chunk_id)


def image_source(doc, image) -> str:
    """图片出处标注：《文档标题》第N页 / 标题>节 / 正文中。"""
    where = f"第{image.page}页" if image.page else (image.position or "正文中")
    return f"《{doc.title}》{where}" if doc and doc.title else where
