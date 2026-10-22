"""六个处理单元 + 注册表 + 排链（数据层骨架：applies 声明式适用性）。

解析函数自 pipeline.py 原样迁入（命名与逻辑零改动），由对应单元的 `run()` 调用；
Unknown 兜底对识别不出的文件置 `engine_used="unsupported"`，如实标注读不动。
Excel 单元为试金石后的第一个新格式（只加类 + 注册，未动识别层）。
"""
from pathlib import Path

from app.core.ingest import docx, ocr, pdf, pdf_tables
from app.core.ingest.excel_docling import EXCEL_SUFFIXES, parse_excel_docling
from app.core.ingest.models import ImageRef, ParsedDocument
from app.core.ingest.pdf_docling import parse_pdf_docling
from app.core.ingest.processor import IngestContext, Processor
from app.core.ingest.report import DataReport, has_text_layer
from app.core.tables import parse_markdown_table, table_summary


def _empty(kind: str) -> ParsedDocument:
    """构建空解析结果（"other" 或无法解析时使用）。"""
    return ParsedDocument(
        kind=kind,
        title=None,
        full_text="",
        headings=[],
        blocks=[],
        engine_used="none",
        ocr_used=False,
        needs_ocr=None,
    )


def _parse_text(path: Path) -> ParsedDocument:
    """纯文本：优先 UTF-8，失败回退 gb18030；两者皆失败则上抛（标记 failed）。"""
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
        engine = "utf-8"
    except UnicodeDecodeError:
        text = data.decode("gb18030")
        engine = "gb18030"
    return ParsedDocument(
        kind="text",
        title=None,
        full_text=text,
        headings=[],
        blocks=[],
        engine_used=engine,
        ocr_used=False,
        needs_ocr=None,
    )


def _parse_image(path: Path, ocr_name: str) -> ParsedDocument:
    """图片：读字节后交给 OCR 引擎识别；引擎关闭(off)则返回空文本。

    照片本体登记为一张 `ImageRef`（position="整图"）——独立图片不依附其他文档，
    登记后与文档内嵌图走同一套机械：入库 caption（勾选时）/ 可用图片清单 /
    view_image 工具 / caption_backfill 补录。未勾选的照片仍永不出本机。
    """
    data = path.read_bytes()
    engine = ocr.get_ocr_engine(ocr_name)
    if engine is None:
        return ParsedDocument(
            kind="image",
            title=None,
            full_text="",
            headings=[],
            blocks=[],
            engine_used=ocr_name,
            ocr_used=False,
            needs_ocr=None,
            images=[ImageRef(blob=data, page=None, position="整图", ocr_text="")],
        )
    ocr_text = engine.recognize(data)
    return ParsedDocument(
        kind="image",
        title=None,
        full_text=ocr_text,
        headings=[],
        blocks=[],
        engine_used=ocr_name,
        ocr_used=True,
        needs_ocr=None,
        images=[ImageRef(blob=data, page=None, position="整图", ocr_text=ocr_text)],
    )


def _ocr_pdf_pages(
    path: Path,
    parsed: ParsedDocument,
    ocr_name: str,
    page_texts: list[str] | None = None,
) -> ParsedDocument:
    """扫描页 OCR：仅 OCR 空/图片页，与已有文字层按页合并（有文字的页保留原文）。

    page_texts 传入时即为已含嵌入图片 OCR 文字的每页文本，避免重算原生文字层丢掉图片 OCR 结果；
    该文本可能已按表格区域过滤（全表页过滤后为空），此时按未过滤文字层判定该页不算扫描页。
    """
    engine = ocr.get_ocr_engine(ocr_name)
    if engine is None:
        return parsed
    import pymupdf as fitz

    native_pages = page_texts if page_texts is not None else pdf.parse_pdf_pages(path)
    doc = fitz.open(str(path))
    merged: list[str] = []
    ocr_used = False
    try:
        for i, page in enumerate(doc):
            native = native_pages[i] if i < len(native_pages) else ""
            if not native.strip() and not pdf._page_text(page).strip():
                merged.append(engine.recognize(page.get_pixmap().tobytes("png")))
                ocr_used = True
            else:
                merged.append(native)  # 有文字层的页保留原生文字（表格区域文字已过滤）
    finally:
        doc.close()
    parsed.full_text = "\n".join(merged)
    parsed.ocr_used = ocr_used
    if ocr_used:
        parsed.engine_used = f"pymupdf+{ocr_name}"
    return parsed


def _replace_table_blocks(parsed: ParsedDocument) -> None:
    """docling 表块就地换成摘要：坐标法已出结构化表，表块原文不再进检索索引。"""
    for block in parsed.blocks:
        if block.type != "table":
            continue
        parsed_table = parse_markdown_table(block.text)
        block.text = (
            table_summary(parsed_table) if parsed_table is not None else "（表格内容见结构化数据）"
        )


def _parse_pdf(path: Path, ocr_name: str, advanced_engine: str) -> ParsedDocument:
    """PDF：有文字层先走坐标法出表，无文字层维持现状（docling 表块交给兜底）。

    advanced_engine=="docling" 且可用走 Docling（坐标法产出 ≥1 张表时其表块换摘要；
    零产出则保留表块，由 structify_tables 兜底解析），异常降级轻量路径；扫描件
    （needs_ocr 且 ocr_name != "off"）走逐页 OCR。两条路径收尾都赋 `parsed.tables`。
    """
    coord_tables = pdf_tables.extract_pdf_tables(path) if has_text_layer(path) else []
    page_texts: list[str] | None = None
    if advanced_engine == "docling":
        try:
            parsed = parse_pdf_docling(path)
        except Exception:
            parsed = pdf.parse_pdf(path)
            parsed.engine_used = "fallback"  # docling 未安装或解析失败 → 降级标记
        else:
            if coord_tables:
                _replace_table_blocks(parsed)  # 坐标法已出表：docling 表块只留摘要
    else:
        # 轻量路径：**过检**表格区域的文字不进 full_text（文本层里表格文字常被拼坏）；
        # 未过表样自检的区域不剔除——正文保住（假表连带正文缺损的教训）
        skip_regions = pdf_tables.table_regions(path) if coord_tables else None
        page_texts, images = pdf.parse_pdf_pages_with_images(
            path, ocr_name, skip_regions_by_page=skip_regions
        )
        parsed = pdf.parse_pdf_with_images(
            path,
            ocr_name,
            page_texts=page_texts,
            images=images,
            skip_regions_by_page=skip_regions,
        )

    if parsed.needs_ocr and ocr_name != "off":
        parsed = _ocr_pdf_pages(path, parsed, ocr_name, page_texts=page_texts)
    parsed.tables = coord_tables
    return parsed


class WordProcessor:
    """Word（.docx/.doc）处理单元。"""

    name = "word"

    def applies(self, report: DataReport) -> bool:
        return report.kind == "word"

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        return docx.parse_docx(path, ocr_name=ctx.ocr_name)


class PdfProcessor:
    """PDF（含扫描件 OCR）处理单元。"""

    name = "pdf"

    def applies(self, report: DataReport) -> bool:
        return report.kind == "pdf"

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        return _parse_pdf(path, ctx.ocr_name, ctx.advanced_engine)


class ImageProcessor:
    """图片（OCR）处理单元。"""

    name = "image"

    def applies(self, report: DataReport) -> bool:
        return report.kind == "image"

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        return _parse_image(path, ctx.ocr_name)


class ExcelProcessor:
    """Excel（.xlsx/.xlsm）处理单元：Docling MsExcel 后端（sheet→页、自动表格检测）。

    适用性只看 `report.suffix`（识别层不知道 excel 这个 kind，也不需要知道）。
    """

    name = "excel"

    def applies(self, report: DataReport) -> bool:
        return report.suffix in EXCEL_SUFFIXES

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        return parse_excel_docling(path)


class TextProcessor:
    """纯文本处理单元。"""

    name = "text"

    def applies(self, report: DataReport) -> bool:
        return report.kind == "text"

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        return _parse_text(path)


class UnknownProcessor:
    """未知格式兜底处理单元：**不参与声明式匹配**，仅当链里没有其他单元适用时接盘。

    （M4/试金石：未知格式的判定从「kind 白名单」改为「无人接管」——新增格式处理器
    只需加一个类 + 注册一行，不影响本类，也不改识别层 detect.py。）
    """

    name = "unknown"

    def applies(self, report: DataReport) -> bool:
        return False  # 永不声明式命中；由 build_chain 在链空时兜底

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        parsed = _empty("other")
        parsed.engine_used = "unsupported"  # 唯一显式行为变更：如实标注读不动
        return parsed


_UNKNOWN = UnknownProcessor()

# 注册顺序即链优先顺序：Word → Pdf → Image → Excel → Text → Unknown（兜底末位、不参与匹配）
PROCESSORS: list[Processor] = [
    WordProcessor(),
    PdfProcessor(),
    ImageProcessor(),
    ExcelProcessor(),
    TextProcessor(),
    _UNKNOWN,
]


def build_chain(report: DataReport) -> list[Processor]:
    """按注册顺序返回全部 `applies` 为真的处理单元；无人适用 → 兜底单元。

    试金石口径（M4 验证成立，spec §8）：**新增格式 = 加一个处理单元 + 注册一行**。
    处理器可用 `report.kind` 或 `report.suffix` 等声明适用性；兜底单元不参与匹配，
    链空时接盘。出现多个适用单元时按注册顺序排在前者优先（链[0] 即生效单元）。
    """
    chain = [p for p in PROCESSORS if p.applies(report)]
    return chain or [_UNKNOWN]
