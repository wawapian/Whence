"""文档上传接口：落盘 + 建 Document + 后台跑 ingestion 管线。"""
import json
import logging
import mimetypes
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import filetype
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image as PILImage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import cloud, llm, queue
from app.core.ingest import pipeline
from app.deps import _session_factory, get_session, get_settings
from app.memory import chunks, documents, materials, store
from app.models.db import Chunk, Document, Image

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)

# 入库 caption：中性描述提示词（只做完整描述，不掺入提问）；每文档上限（防极端文档烧调用）
_NEUTRAL_CAPTION_PROMPT = "请完整描述这张图片的内容（图片类型、关键信息、图中出现的文字与数字）。"
MAX_CAPTIONS_PER_DOC = 20


def _image_extension(blob: bytes) -> str | None:
    """从图片字节魔数识别扩展名；无法识别返回 None。"""
    kind = filetype.guess(blob)
    if kind is None:
        return None
    ext = (kind.extension or "").lower()
    if ext == "jpeg":
        return "jpg"
    return ext or None


def _save_image_blob(data_dir: Path, blob: bytes) -> str | None:
    """保存图片字节到 `data_dir/images/<uuid>.<ext>`，返回相对 data_dir 路径；格式不可识别返回 None。"""
    ext = _image_extension(blob)
    if ext is None:
        return None
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    name = store.new_id()
    (images_dir / f"{name}.{ext}").write_bytes(blob)
    return f"images/{name}.{ext}"


def is_likely_decorative(blob: bytes, *, max_px: int = 1024, uniform_ratio: float = 0.995) -> bool:
    """装饰小图启发式（**保守**）：只打「极小 且 均匀纯色块」——只降级索引，绝不丢弃。

    - 面积 ≤ max_px（默认 32×32）：真正的"极小"；印章/色板/二维码通常更大；
    - 且单一颜色占比 ≥ uniform_ratio：真正均匀的色块（0.5% 抗锯齿容差）。
      印章有文字/星徽、二维码密布边缘——颜色多或分布不均，均不会命中；
    - 解码失败/非图片 → False（保守：保住）。
    """
    try:
        with PILImage.open(BytesIO(blob)) as img:
            width, height = img.size
            if width * height > max_px:
                return False
            colors = img.convert("RGB").getcolors(maxcolors=16)
            if colors is None:
                return False  # 颜色 >16 种：有细节
            dominant = max(count for count, _ in colors)
            return dominant / (width * height) >= uniform_ratio
    except Exception:
        return False


@router.post("/files")
def upload_file(
    file: UploadFile = File(...),
    description: str = Form(""),
    allow_cloud: int = Form(0),
    multimodal: int = Form(0),
    ocr: str = Form("auto"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """接收文件：落盘、建 Document(status=processing)、提交后台任务。"""
    file_path = store.save_upload(file, settings.data_dir)
    doc = documents.insert_document(
        session,
        kind=store.classify_kind(file.content_type),
        title=file.filename,
        file_path=file_path,
        status="processing",
        allow_cloud=allow_cloud,
        multimodal=multimodal,
    )
    session.commit()
    task_id = queue.submit(process_document, settings, doc.id, file_path, ocr, allow_cloud, multimodal)
    return {"document_id": doc.id, "task_id": task_id, "status": "processing"}


@router.get("/files/{doc_id}")
def get_file(doc_id: str, session: Session = Depends(get_session)):
    """查询文档处理状态 + 正文预览 + 分块数 + 摘要。"""
    doc = documents.get_document(session, doc_id)
    # get_document 按设计不过滤软删（内部按主键取）；读路径在此显式排除已删文档，
    # 已删不得继续可读（spec §4/§5）；回收站/恢复不走本端点，404 无副作用
    if doc is None or doc.deleted_at is not None:
        raise HTTPException(status_code=404, detail="document not found")
    chunk_count = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)
    )
    return {
        "status": doc.status,
        "parsed_text": (doc.parsed_text or "")[:500],
        "chunk_count": chunk_count or 0,
        "summary": doc.summary,
    }


_INLINE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".txt", ".md", ".csv"}


def _content_disposition(name: str, inline: bool) -> str:
    """Content-Disposition 头（RFC 5987 原名，中文文件名安全）。"""
    kind = "inline" if inline else "attachment"
    return f"{kind}; filename*=UTF-8''{quote(name)}"


def _wants_inline(disposition: str | None, suffix: str) -> bool:
    asked = (disposition or "").strip().lower()
    if asked == "inline":
        return True
    if asked == "attachment":
        return False
    return suffix in _INLINE_SUFFIXES  # 缺省：图片/PDF 直接看，其余下载


# rel_path 的「未指定」哨兵：None 是合法入参（图片行 file_path 可为空 → 该图本就该 404），
# 不能拿 None 表示「缺省取 doc.file_path」
_DOC_FILE_PATH = object()


def readable_original_path(session: Session, data_dir: Path, *, doc: Document | None,
                           rel_path: str | None | object = _DOC_FILE_PATH) -> Path | None:
    """原件可读性**单一判定**（未来鉴权/数据隔离只在此加层）：不可读 → None（端点 404）。

    条件：文档存在、未软删、目标相对路径可安全解析且盘上存在。`rel_path` 缺省 = `doc.file_path`
    （文档端点）；图片端点显式传 `img.file_path`（可为 None → `resolve_data_file` 拒绝 → 404，
    绝不回退成文档文件）。**不看 `allow_cloud`**——取原件是纯本地读取、不出机
    （spec §3.2；与「上云闸」是两条闸）。
    """
    if doc is None or doc.deleted_at is not None:
        return None
    target = doc.file_path if rel_path is _DOC_FILE_PATH else rel_path
    try:
        return store.resolve_data_file(data_dir, target)
    except ValueError:
        return None


def _file_response(path: Path, disposition: str | None, *,
                   name: str, suffix: str, media_type: str | None = None) -> FileResponse:
    """组装 FileResponse：`name/suffix/media_type` 按**实际服务的文件**由调用方给定——
    文档端点 = 上传原名 / 文档路径后缀 / 按原名猜（契约承诺 RFC 5987 原名）；
    图片端点 = 存储名 / 实际图片路径后缀 / 按实际图片路径猜。
    `media_type` 为空 → 按实际路径后缀猜，再兜底 application/octet-stream。"""
    media_type = media_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    inline = _wants_inline(disposition, suffix)
    if suffix in {".txt", ".md", ".csv"}:
        media_type = "text/plain"
    return FileResponse(
        path, media_type=media_type,
        headers={"Content-Disposition": _content_disposition(name, inline), "X-Content-Type-Options": "nosniff"},
    )


@router.get("/files/{doc_id}/raw")
def get_file_raw(doc_id: str, disposition: str | None = None,
                 session: Session = Depends(get_session),
                 settings: Settings = Depends(get_settings)):
    """取原件（本地读取、不出机）：图片/PDF 缺省 inline，其余 attachment；软删/越界/缺失 → 404。"""
    doc = documents.get_document(session, doc_id)
    path = readable_original_path(session, settings.data_dir, doc=doc)
    if path is None:
        raise HTTPException(status_code=404, detail="file not available")
    return _file_response(
        path, disposition,
        name=doc.title or path.name,  # 契约承诺：文档端点用上传原名（RFC 5987）
        suffix=Path(doc.file_path or doc.title or "").suffix.lower(),
        media_type=mimetypes.guess_type(doc.title or "")[0],
        
    )


@router.get("/images/{image_id}/raw")
def get_image_raw(image_id: str, disposition: str | None = None,
                  session: Session = Depends(get_session),
                  settings: Settings = Depends(get_settings)):
    """看图的原件——吐**图片本体**（`images.file_path`，本地读取、不出机）：
    图片行不存在 / `file_path` 为空 / 所属文档软删 / 路径非法 / 文件缺失 → 404。"""
    img = session.get(Image, image_id)
    if img is None:
        raise HTTPException(status_code=404, detail="image not available")
    doc = documents.get_document(session, img.document_id)
    path = readable_original_path(session, settings.data_dir, doc=doc, rel_path=img.file_path)
    if path is None:
        raise HTTPException(status_code=404, detail="image not available")
    return _file_response(path, disposition, name=path.name, suffix=path.suffix.lower())


def process_document(settings: Settings, doc_id: str, file_path: str, ocr: str, allow_cloud: int, multimodal: int):
    """后台任务（可复用）：独立 session 跑 ingestion 管线，回写 Document 与 Chunk/Image/索引。"""
    session = _session_factory(settings)()
    try:
        ocr_name = "off" if ocr == "off" else settings.ocr_engine
        ingest = pipeline.ingest
        parser_options = {}
        result = ingest(
            Path(settings.data_dir) / file_path,
            ocr_name=ocr_name,
            chunk_strategy=settings.chunk_strategy,
            chunk_max=settings.chunk_max_chars,
            chunk_overlap=settings.chunk_overlap,
            advanced_engine=settings.advanced_engine,
            **parser_options,
        )
        doc = session.get(Document, doc_id)
        doc.parsed_text = result.parsed.full_text
        doc.ocr_used = int(result.parsed.ocr_used)
        doc.engine_used = result.parsed.engine_used
        doc.needs_ocr = int(result.parsed.needs_ocr) if result.parsed.needs_ocr is not None else None
        doc.status = "done"
        image_rows = []
        doc_file_path = file_path  # 入参：文档自身在 files/ 的路径（独立照片直接引用它，不复制）
        for img in result.parsed.images:
            image_file_path = None
            decorative = 0
            try:
                if result.parsed.kind == "image":
                    # 独立照片（随手拍）：images 行直接引用原文件，省一份复制
                    image_file_path = doc_file_path
                else:
                    image_file_path = _save_image_blob(settings.data_dir, img.blob)
                if settings.image_filter == "on":
                    decorative = int(is_likely_decorative(img.blob))
            except Exception:
                logger.exception("document %s image save failed", doc_id)
            image_rows.append(
                documents.insert_image(
                    session,
                    document_id=doc_id,
                    file_path=image_file_path,
                    page=img.page,
                    position=img.position,
                    ocr_text=(img.ocr_text or ""),
                    decorative=decorative,
                )
            )
        n_decorative = sum(row.decorative for row in image_rows)
        if n_decorative:
            # 如实记录：疑似装饰小图只降级（不描述、不进图片清单），文件与行永久保留
            logger.info("document %s: 疑似装饰小图 %d 张（已降级，不自动描述）", doc_id, n_decorative)
        # 先算完全部分块向量（慢，纯本地 CPU，不碰库），再进入一次性短事务写库。
        # 反过来会把 SQLite 写锁攥住整个嵌入时长（大表可达分钟级）——期间任何别的
        # 写入（删除等）都会 busy 超时失败（用户实测：大 Excel 入库期间删除报 500）
        chunk_vecs = [llm.embed(c.text, model=settings.embedding_model) for c in result.chunks]
        for i, (c, vec) in enumerate(zip(result.chunks, chunk_vecs)):
            chunk = documents.insert_chunk(
                session,
                document_id=doc_id,
                seq=i,
                text=c.text,
                heading_path=c.heading_path,
                page=c.page,
                parent_text=c.parent_text,
            )
            chunks.index_chunk(session, chunk.id, vec, c.text)
        for t in result.tables:
            documents.insert_table(
                session,
                document_id=doc_id,
                heading=t.heading,
                headers=json.dumps(t.headers, ensure_ascii=False),
                rows=json.dumps(t.rows, ensure_ascii=False),
                suspect=t.suspect,
            )
        session.commit()

        # 入库 caption：与云总结同一原则，在本地工作落库后单独执行——单图失败仅该图无 caption，
        # 不破坏已完成的解析/分块；也不在 VLM 网络调用期间持有 SQLite 写事务。
        # 整阶段 + 每图双层保护：客户端构造失败→跳过整个阶段；描述/向量化/索引失败→只回滚该图。
        # 任何 caption 阶段异常都不得逃到外层 handler 把已完成的文档改标 failed。
        if allow_cloud and settings.vlm_model:
            try:
                vlm_client = cloud.build_vlm_client(settings)
                for i, (img_ref, img_row) in enumerate(zip(result.parsed.images, image_rows)):
                    if img_row.decorative:
                        continue  # 疑似装饰小图：只降级，不自动描述（也不占用每文档名额）
                    if i >= MAX_CAPTIONS_PER_DOC:
                        break
                    if not img_row.file_path:
                        continue
                    try:
                        caption = cloud.describe_image(
                            vlm_client, settings.vlm_model, img_ref.blob, _NEUTRAL_CAPTION_PROMPT
                        )
                        if not caption:
                            logger.warning("document %s image #%d caption empty, skipped", doc_id, i)
                            continue
                        img_row.caption = caption
                        text = f"图片描述（{documents.image_source(doc, img_row)}）：{caption}"
                        chunk = documents.insert_chunk(
                            session,
                            document_id=doc_id,
                            seq=len(result.chunks) + i,
                            text=text,
                            heading_path="图片",
                            parent_text=text,
                        )
                        chunks.index_chunk(session, chunk.id,
                                           llm.embed(text, model=settings.embedding_model), text)
                        session.commit()
                    except Exception:
                        session.rollback()  # 只丢弃该图的部分写入；前几图与主体文档均已提交
                        logger.warning("document %s image #%d caption failed", doc_id, i, exc_info=True)
                        continue
            except Exception:
                logger.warning("document %s caption stage failed, skipped", doc_id, exc_info=True)

        # 云总结在本地工作落库后单独执行：失败仅无 summary，不破坏已完成的解析/分块（spec §8）
        # 表格类文档（正文只剩表格摘要行）跳过：退化输入只会换回抱怨文本、毒化材料索引
        if allow_cloud and not pipeline.is_table_only_text(result.parsed.full_text):
            try:
                doc.summary = cloud.summarize(
                    llm.build_client(settings), settings.llm_model, result.parsed.full_text
                )
                session.commit()
            except Exception:
                logger.exception("document %s cloud summary failed", doc_id)

        # 材料层：入库后生成材料索引（摘要规则见 app/memory/materials.py）。
        # 失败仅记 warning 并回滚本阶段，不得把已完成的文档改标 failed（同 caption/云总结阶段的原则）。
        # fail-soft 与记录侧 material 的 fail-closed 是刻意不同（F2）：记录重试=重发一句话；
        # 文档重跑解析代价高，材料失败不该拖垮已完成的主体（见 api/memories.py 对应注释）
        try:
            materials.upsert_for_document(session, doc, model=settings.embedding_model)
            session.commit()
        except Exception:
            session.rollback()
            logger.warning("document %s material upsert failed", doc_id, exc_info=True)
    except Exception:
        logger.exception("document %s processing failed", doc_id)
        session.rollback()
        doc = session.get(Document, doc_id)
        if doc is not None:
            doc.status = "failed"
            session.commit()
        raise
    finally:
        session.close()
