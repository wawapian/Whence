"""文档检索与列表接口。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import llm, retrieval, trash
from app.deps import get_session, get_settings
from app.memory import documents, store
from app.api.files import readable_original_path

router = APIRouter(prefix="/api")


class DocumentSearchIn(BaseModel):
    question: str
    top_k: int = 10


@router.post("/documents/search")
def search_documents(
    payload: DocumentSearchIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    ids = retrieval.search_documents(
        session,
        payload.question,
        top_k=payload.top_k,
        model=settings.embedding_model,
        rerank_engine=settings.rerank_engine,
        rerank_model=settings.rerank_model,
        client=llm.build_client(settings),
        llm_model=settings.llm_model,
    )
    results = []
    for cid in ids:
        chunk = documents.get_chunk(session, cid)
        if chunk is None:
            continue
        doc = documents.get_document(session, chunk.document_id)
        results.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "title": doc.title if doc else None,
            "heading_path": chunk.heading_path,
            "page": chunk.page,
            "text": chunk.text,
            "parent_text": chunk.parent_text,
        })
    return {"results": results}


@router.get("/documents")
def list_documents(session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    """未删除文档列表（不含已删，供前端「文档」区）。"""
    return {
        "documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "kind": doc.kind,
                "status": doc.status,
                "allow_cloud": bool(doc.allow_cloud),  # 文件库/预览栏"可否上云"标（本地取原件不受限）
                "raw_id": doc.raw_id,
                "summary": doc.summary,
                "preview": (doc.parsed_text or "")[:350],
                "size_bytes": _file_size(session, settings, doc),
                "created_at": store.to_local(doc.created_at),  # 本机时区带偏移输出（前端直接显示）
            }
            for doc in documents.list_documents(session)
        ]
    }


def _file_size(session, settings, doc):
    path = readable_original_path(session, settings.data_dir, doc=doc)
    try:
        return path.stat().st_size if path else None
    except OSError:
        return None


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, session: Session = Depends(get_session)):
    """移入回收站（软删）：设 `deleted_at`；不存在或已在回收站 → 404。"""
    if not trash.soft_delete_document(session, doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    session.commit()
