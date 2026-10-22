"""PDF 解析：PyMuPDF 逐页提取文字并按坐标排序；扫描件检测。"""
import logging
from pathlib import Path

import pymupdf as fitz

from app.core.ingest import ocr
from app.core.ingest.models import ImageRef, ParsedDocument

logger = logging.getLogger(__name__)

# 表格边界框（x0, y0, x1, y1）
BBox = tuple[float, float, float, float]


def detect_scanned(text: str) -> bool:
    """扫描件判定：去首尾空白后为空（文字长度≈0）即视为扫描页。"""
    return text.strip() == ""


def _inside(bbox: tuple[float, float, float, float], regions: list[BBox]) -> bool:
    """行 bbox 是否完全落在任一表格边界框内（完全落内才剔除，边界附近的正文行不受影响）。"""
    return any(
        region[0] <= bbox[0] and region[1] <= bbox[1]
        and bbox[2] <= region[2] and bbox[3] <= region[3]
        for region in regions
    )


def _page_text(page, skip_regions: list[BBox] | None = None) -> str:
    """单页文字：按 bbox y,x 排序（上到下、左到右）后逐行拼接；完全落在 skip_regions 内的行剔除。"""
    lines: list[tuple[float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 仅文字块，跳过图片块
            continue
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if not text:
                continue
            bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            if skip_regions and _inside(bbox, skip_regions):
                continue  # 表格区域文字：garbled 原文不进正文（表格另走结构化来源）
            lines.append((bbox[1], bbox[0], text))  # (y, x) 上到下、左到右
    lines.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(item[2] for item in lines)


def parse_pdf_pages(
    path: Path, skip_regions_by_page: dict[int, list[BBox]] | None = None
) -> list[str]:
    """逐页提取文字（按 bbox y,x 排序），返回每页文本列表。

    skip_regions_by_page：页序号（0 起）→ 表格 bbox 列表；有值的页剔除表格区域内的文本行。
    """
    doc = fitz.open(str(path))
    page_texts: list[str] = []
    try:
        for i, page in enumerate(doc):
            regions = skip_regions_by_page.get(i) if skip_regions_by_page else None
            page_texts.append(_page_text(page, regions))
    finally:
        doc.close()
    return page_texts


def parse_pdf_pages_with_images(
    path: Path,
    ocr_name: str = "off",
    skip_regions_by_page: dict[int, list[BBox]] | None = None,
) -> tuple[list[str], list[ImageRef]]:
    """逐页提取文字并对嵌入图片逐图 OCR（仅原生文字层非空的页），返回 (每页文本, 图片引用)。

    有 skip_regions_by_page 的页返回剔除表格区域行的文本（供 full_text）；是否 OCR 嵌入图片
    只看未过滤文字层（表格区域不影响判定）。
    """
    engine = ocr.get_ocr_engine(ocr_name)
    doc = fitz.open(str(path))
    page_texts: list[str] = []
    images: list[ImageRef] = []
    try:
        for i, page in enumerate(doc):
            regions = skip_regions_by_page.get(i) if skip_regions_by_page else None
            raw_text = _page_text(page)
            text = _page_text(page, regions) if regions else raw_text
            if raw_text.strip() and engine is not None:
                for img in page.get_images(full=True):
                    try:
                        blob = doc.extract_image(img[0])["image"]
                        rec = engine.recognize(blob)
                    except Exception:
                        logger.warning("嵌入图片 OCR 失败（第 %d 页，已跳过）", i + 1, exc_info=True)
                        continue
                    if rec:
                        text = (text + "\n" + rec).strip()
                    images.append(ImageRef(blob=blob, page=i + 1, ocr_text=rec))
            page_texts.append(text)
    finally:
        doc.close()
    return page_texts, images


def parse_pdf_with_images(
    path: Path,
    ocr_name: str = "off",
    page_texts: list[str] | None = None,
    images: list[ImageRef] | None = None,
    skip_regions_by_page: dict[int, list[BBox]] | None = None,
) -> ParsedDocument:
    """解析 PDF 并对嵌入图片逐图 OCR，图片文字与原生文字层合并（仅原生文字层非空的页 OCR 嵌入图片）。

    full_text 用过滤版（表格区域行已剔除）；ocr_used / needs_ocr 一律按未过滤文本判定——
    过滤会掏空全表页的文本，否则会被误判为「扫描页/用过 OCR」。
    """
    if page_texts is None:
        page_texts, images = parse_pdf_pages_with_images(path, ocr_name, skip_regions_by_page)
    native_pages = parse_pdf_pages(path)  # 未过滤
    filtered_pages = (
        parse_pdf_pages(path, skip_regions_by_page) if skip_regions_by_page else native_pages
    )
    # 与未过滤版对齐比较：差异只来自嵌入图片 OCR 追加的文字，过滤本身不算「用过 OCR」
    ocr_used = any(a != b for a, b in zip(page_texts, filtered_pages))
    return ParsedDocument(
        kind="pdf",
        title=None,
        full_text="\n".join(page_texts),
        headings=[],
        blocks=[],
        engine_used=("pymupdf+ocr" if ocr_used else "pymupdf"),
        ocr_used=ocr_used,
        needs_ocr=any(detect_scanned(page) for page in native_pages),
        images=images or [],
    )


def parse_pdf(path: Path) -> ParsedDocument:
    """解析 PDF 为 ParsedDocument：逐页提文字并拼接；needs_ocr 按页判定（任一页为空即含扫描页）。"""
    page_texts = parse_pdf_pages(path)
    full_text = "\n".join(page_texts)
    return ParsedDocument(
        kind="pdf",
        title=None,
        full_text=full_text,
        headings=[],
        blocks=[],
        engine_used="pymupdf",
        ocr_used=False,
        needs_ocr=any(detect_scanned(page) for page in page_texts),
    )
