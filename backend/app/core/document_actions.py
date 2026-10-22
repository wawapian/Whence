"""Required document tools on the existing answer spine: find originals and read whole bodies."""
import re
from sqlalchemy import select

from app.core import llm, summarize
from app.memory import documents as repository
from app.memory.store import resolve_data_file
from app.models.db import Chunk
from app.models.schemas import QueryResponse, DocumentSource


def is_find_request(question: str) -> bool:
    return bool(re.search(r"查找|找一下|帮我找|帮我查|寻找|找到|找出|找一份|找个|找份", question)) and bool(
        re.search(r"小票|收据|发票|文件|原件|文档|照片|截图|合同|报告|论文|图片|pdf", question, re.I)
    ) and not bool(re.search(r"多少钱|多少元|哪些商品|总结|概括|分析|怎么|如何", question))


def _file(doc, settings):
    try:
        available = bool(resolve_data_file(settings.data_dir, doc.file_path))
    except (ValueError, OSError):
        available = False
    return {"document_id": doc.id, "title": doc.title or "未命名原件", "kind": doc.kind,
            "available": available, "preview": (doc.summary or doc.parsed_text or "")[:180]}


def try_handle(session, question, settings):
    """Explicit document actions require deterministic tool completion before any reply."""
    if is_find_request(question):
        found = repository.find_files(session, question, limit=10)
        if not found:
            return None  # The normal hybrid retriever may still resolve a paraphrase.
        return QueryResponse(answer=f"找到了 {len(found)} 份匹配的原件，可直接打开查看。",
            execution={"files": [_file(d, settings) for d in found], "action": "find_files",
                       "trace": [{"tool": "list_files", "matched": len(found), "status": "ok"}]})
    if not summarize.is_summary_request(question):
        return None
    targets = repository.named_documents(session, question)
    if not targets:
        # Reuse the existing material selection, without sending neighbouring texts to a summary.
        from app.core import retrieval
        from app.models.db import Material, Document
        ids = retrieval.search_materials_multi(session, [question], top_k=8, model=settings.embedding_model)
        mats = session.scalars(select(Material).where(Material.id.in_(ids))).all()
        target_ids = [m.source_id for m in mats if m.source_type == "document"]
        targets = session.scalars(select(Document).where(Document.id.in_(target_ids), Document.deleted_at.is_(None))).all()
        if not targets and len(repository.list_documents(session)) == 1:
            targets = repository.list_documents(session)
        elif len(targets) != 1 and settings.qa_engine == "legacy":
            # Preserve the original identity + attached-memory selection for old libraries
            # that predate material indexes. Full reading still uses the new single engine.
            from app.core import rewrite
            from app.core.answer import _identify_target_document
            client = llm.build_client(settings)
            queries = rewrite.retrieval_queries(question, settings, client)
            chunk_ids = retrieval.search_documents_multi(session, queries, top_k=10,
                model=settings.embedding_model, rerank_engine=settings.rerank_engine,
                rerank_model=settings.rerank_model, client=client, llm_model=settings.llm_model)
            hit_chunks = session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()
            target_id = _identify_target_document(session, question, hit_chunks, settings)
            targets = [session.get(Document, target_id)] if target_id else []
    if len(targets) != 1:
        return QueryResponse(answer="请指定要总结的那一份文档，以免把不同论文的内容混在一起。",
                             status="partial", execution={"files": [_file(d, settings) for d in targets],
                                                        "action": "choose_summary_document"})
    target = targets[0]
    result = summarize.summarize_with_coverage(session, target.id, client=llm.build_client(settings), model=settings.llm_model)
    chunks = session.scalars(select(Chunk).where(Chunk.document_id == target.id).order_by(Chunk.seq).limit(3)).all()
    refs = [DocumentSource(chunk_id=c.id, document_id=target.id, title=target.title,
                           heading_path=c.heading_path, page=c.page, text=c.text) for c in chunks]
    return QueryResponse(answer=result.text, status=result.status, documents=refs,
        execution={"reading": result.reading, "summary_version": 1, "action": "full_summary",
                   "files": [_file(target, settings)],
                   "trace": [{"tool": "summarize_document", "document_id": target.id,
                              "status": result.status, "segments": result.reading.get("segments", [])}]})
