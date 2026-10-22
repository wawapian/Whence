"""记录接口：文字 / 照片 / 文档。

**accept-fast**：请求只做「附件落盘 + RAW/文档落库」即返回（毫秒级）；
打码/抽取/事实/索引/材料全部交给后台 **串行整理**（`consolidate_record`）——
整理必须保序（双时态「后说的覆盖先说的」依赖提交顺序），故走 record 车道（1 工人 FIFO）；
附件完整解析、切片、嵌入走 doc 车道（4 工人并行）。
重启丢任务由启动扫尾重投兜底（见 core/recovery.py + raw_evidence.consolidated 标记）。
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import exists, select, text
from sqlalchemy.orm import Session

from app.api.files import process_document
from app.config import Settings
from app.core import extraction, llm, parsing, queue, trash
from app.core.masking import link_secrets_to_fact, load_fernet, mask_text
from app.deps import _session_factory, get_session, get_settings
from app.memory import bitemporal, documents, fts, materials, store, vector
from app.models.db import Document, Fact, RawEvidence
from app.models.schemas import ExtractedFact

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _raw_only_summary(raw: RawEvidence, file_name: str | None) -> str:
    """无事实纯记录的时间线摘要：原话优先，无文字则「（照片/文件）文件名（上传原名）」。"""
    content = (raw.content or "").strip()
    if content:
        return content[:60]
    label = "照片" if raw.kind == "photo" else "文件"
    name = file_name or (Path(raw.file_path).name if raw.file_path else "")
    return f"（{label}）{name}" if name else f"（{label}）"


@router.get("/memories")
def list_memories(session: Session = Depends(get_session), limit: int = 50):
    """时间线 = 事实条目 + **无事实的纯记录**（照片等 raw_only）——存了东西就要看得见。"""
    facts = session.scalars(
        select(Fact)
        .join(RawEvidence, RawEvidence.id == Fact.raw_id)
        .where(RawEvidence.deleted_at.is_(None))
        .order_by(Fact.created_at.desc())
        .limit(limit)
    ).all()
    entries = [
        {
            "fact_id": f.id,
            "raw_id": f.raw_id,  # 前端删除按钮按记录（raw）删除需要它；无记录时可为 None
            "type": f.type,
            "summary": f.summary,
            "fields": f.fields or {},
            "subject_key": f.subject_key,
            "valid_from": store.to_local(f.valid_from),
            "valid_to": store.to_local(f.valid_to),
            "superseded_by": f.superseded_by,
            "state": f.state,
            "created_at": store.to_local(f.created_at).isoformat() if f.created_at else None,
        }
        for f in facts
    ]
    # 纯记录（尚无事实，如纯照片 / 整理中的新记录）：同样进时间线
    pending = session.scalars(
        select(RawEvidence)
        .where(
            RawEvidence.deleted_at.is_(None),
            ~exists().where(Fact.raw_id == RawEvidence.id),
        )
        .order_by(RawEvidence.created_at.desc())
        .limit(limit)
    ).all()
    # 文件名取上传原名（Document.title；raw.file_path 的存储名是 uuid，对用户无意义）
    name_by_raw: dict[str, str] = {}
    if pending:
        ids = [r.id for r in pending]
        for doc in session.scalars(
            select(Document).where(Document.raw_id.in_(ids)).order_by(text("rowid"))
        ).all():
            if doc.raw_id not in name_by_raw and doc.title:
                name_by_raw[doc.raw_id] = doc.title
    for raw in pending:
        entries.append(
            {
                "fact_id": None,
                "raw_id": raw.id,
                "type": raw.kind,
                "summary": _raw_only_summary(raw, name_by_raw.get(raw.id)),
                "consolidated": raw.consolidated,
                "created_at": store.to_local(raw.created_at).isoformat() if raw.created_at else None,
            }
        )
    # 附件 id：时间线点记录→打开原件需要（一次查询建映射，防 N+1；软删的不算）
    raw_ids = [e["raw_id"] for e in entries if e["raw_id"]]
    docs_by_raw: dict[str, list[str]] = {}
    if raw_ids:
        for doc in session.scalars(
            select(Document).where(
                Document.raw_id.in_(raw_ids), Document.deleted_at.is_(None)
            ).order_by(text("rowid"))
        ):
            docs_by_raw.setdefault(doc.raw_id, []).append(doc.id)
    for entry in entries:
        entry["doc_ids"] = docs_by_raw.get(entry["raw_id"], [])

    entries.sort(key=lambda e: e["created_at"] or "", reverse=True)
    return entries[:limit]


@router.post("/memories")
def create_memory(
    text: str = Form(""),
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    sensitive: bool = Form(True),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    # 前端「可理解」勾选框 → sensitive=false → 允许上云（云总结 / 查询时 VLM 看图）
    allow_cloud = 0 if sensitive else 1
    # 多附件：新 `files` 列表 + 兼容旧单键 `file`（两者合并，`file` 在前）
    attachments = ([file] if file is not None else []) + list(files or [])

    # 落盘 + 分类（同步段只做这个；正文解析挪到后台整理）
    saved = []  # [(file_path, kind)]
    for upload in attachments:
        file_path = store.save_upload(upload, settings.data_dir)
        saved.append((file_path, store.classify_kind(upload.content_type)))

    kinds = [k for (_, k) in saved]
    if not kinds:
        raw_kind = "text"
    elif all(k == "photo" for k in kinds):
        raw_kind = "photo"
    else:
        raw_kind = "document"
    # 原话优先：写了文字立刻入库；无原话的正文回填在后台整理时补（consolidate_record）
    # file_path 只落首个附件（可逆取舍：全组路径可经 documents.raw_id 反查）
    raw = store.insert_raw(
        session, kind=raw_kind, content=text or None,
        file_path=saved[0][0] if saved else None,
    )
    raw.consolidated = 0  # 待整理标记（启动扫尾据此重投）
    session.flush()  # RAW 先落库，满足 Document.raw_id 关联

    docs = []  # [(doc_id, file_path)]；每份附件各自成 Document，同一 raw_id 成组
    for upload, (file_path, kind) in zip(attachments, saved):
        doc = documents.insert_document(
            session, kind=kind, title=upload.filename, file_path=file_path,
            status="processing", allow_cloud=allow_cloud, raw_id=raw.id,
        )
        docs.append((doc.id, file_path))
    session.commit()

    # 后台：记录整理（串行保序）+ 逐份附件完整 ingestion（并行）
    queue.submit(consolidate_record, settings, raw.id, lane="record")
    for doc_id, file_path in docs:
        queue.submit(process_document, settings, doc_id, file_path, "auto", allow_cloud, 0)

    return {"raw_id": raw.id, "state": "accepted"}


def _extraction_desc(text: str, docs: list[Document]) -> str:
    """抽取用描述（与旧同步版逐字一致）：原话 / 单附件「上传了文档：X」/ 多附件「上传了 N 个文件：…」。"""
    if text:
        return text
    names = [d.title or "未知文件" for d in docs]
    if len(docs) == 1:
        return f"上传了文档：{names[0]}"
    return f"上传了 {len(docs)} 个文件：{'、'.join(names)}"


def consolidate_record(settings: Settings, raw_id: str) -> None:
    """后台整理一条记录（record 车道串行执行）：正文回填 → 打码 → 抽取 → 事实/索引/材料。

    - raw 不存在或已软删 → 跳过（防「删了还冒出来」）；
    - 无原话时解析附件回填 raw.content（仅当为空；单份失败不影响其余）；
    - 无正文可整理 → 标 consolidated=1 返回（raw_only，与旧行为一致：不建事实/材料）；
    - 抽取出的**每条**事实逐条落库/建索引（材料仍一份，title 取首条摘要）——一条记录可含
      多个陈述（发现#2：位置类陈述曾被整条漏抽）；
    - 打码/抽取失败/零事实 → pending 兜底；未预期异常 → 尽力写 pending 兜底事实，时间线不丢记录。
    """
    session = _session_factory(settings)()
    try:
        raw = session.get(RawEvidence, raw_id)
        if raw is None or raw.deleted_at is not None:
            return  # 已彻底删除 / 已进回收站：不整理
        docs = session.scalars(
            select(Document).where(Document.raw_id == raw.id).order_by(text("rowid"))
        ).all()

        user_text = (raw.content or "").strip()
        if not user_text:
            parts: list[str] = []
            for doc in docs:
                if doc.kind != "document" or not doc.file_path:
                    continue
                try:
                    parsed = parsing.parse_document(settings.data_dir / doc.file_path)
                except Exception:
                    parsed = ""  # 单份失败不影响其余（与旧行为一致）
                if parsed:
                    parts.append(parsed)
            doc_text = "\n\n".join(parts)
            if doc_text:
                raw.content = doc_text  # 原话优先：只在本为空时回填
        else:
            doc_text = ""

        if not (user_text or doc_text):
            raw.consolidated = 1  # raw_only：无事实可建（纯照片 / 解析不出正文）
            session.commit()
            return

        desc = _extraction_desc(user_text, docs)
        placeholders: dict[str, str] = {}
        state = "pending"
        facts: list[ExtractedFact] = []
        try:
            masked, placeholders = mask_text(desc, session, load_fernet(settings.data_dir))
            try:
                facts = extraction.extract(llm.build_client(settings), settings.llm_model, masked)
            except Exception:
                facts = []  # 抽取失败：回落 pending，原文已在 RAW，不丢数据
            if facts:
                state = "confirmed"
        except Exception:
            # 打码失败同样兜底 pending（摘要=本地内容前 50 字，仅本地存储）
            logger.warning("consolidate 打码失败（回落 pending）: %s", raw_id, exc_info=True)
        if not facts:
            # 抽取异常 / 零事实（空列表）：pending 兜底单事实（时间线不丢记录）
            facts = [ExtractedFact(type="other", summary=desc[:50], confidence=0.0)]

        # 逐条落库：顺序＝抽取输出顺序（同一记录内同 subject_key 的后续条覆盖前条）
        idx_text = user_text or doc_text
        for i, fact in enumerate(facts):
            f = bitemporal.upsert(session, fact, raw.id, state=state)
            if i == 0:
                # 打码先于 fact 存在，此刻补上关联（彻底删除按记录内全部 fact_id 清账，
                # 挂首条即够；逐条挂则 fact_id 被后续覆盖、徒增写入）
                link_secrets_to_fact(session, placeholders, f.id)
            if docs:
                f.fields = {**(f.fields or {}), "sensitive": not bool(docs[0].allow_cloud)}
            # 建索引（FTS5 + 向量）：summary + 正文（截断）——与 RAW 同源（原话优先）
            content = f"{fact.summary} {idx_text[:2000]}".strip()
            fts.index_fact(session, f.id, content)
            vector.index_fact(
                session, f.id, llm.embed(content[:500], model=settings.embedding_model)
            )
        # 材料层（fail-closed 语义随本事务）：材料与事实同事务保持一致；title 取首条摘要
        materials.upsert_for_record(
            session, raw, title=facts[0].summary or desc[:50], model=settings.embedding_model
        )
        raw.consolidated = 1
        session.commit()
    except Exception:
        logger.exception("consolidate_record 未预期异常，尝试兜底 pending 事实：%s", raw_id)
        try:
            session.rollback()
            raw = session.get(RawEvidence, raw_id)
            if raw is not None and raw.deleted_at is None:
                summary = (raw.content or "记录")[:50]
                bitemporal.upsert(
                    session,
                    ExtractedFact(type="other", summary=summary, confidence=0.0),
                    raw.id,
                    state="pending",
                )
                raw.consolidated = 1
                session.commit()
        except Exception:
            logger.exception("consolidate_record 兜底也失败（记录仍可见为 raw）: %s", raw_id)
    finally:
        session.close()


@router.delete("/memories/{raw_id}", status_code=204)
def delete_memory(raw_id: str, session: Session = Depends(get_session)):
    """移入回收站（软删）：设 `deleted_at`；不存在或已在回收站 → 404。"""
    if not trash.soft_delete_record(session, raw_id):
        raise HTTPException(status_code=404, detail="record not found")
    session.commit()
