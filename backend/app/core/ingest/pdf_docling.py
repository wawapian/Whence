"""PDF 高级档：Docling 转 markdown（公式 LaTeX + 表格）；失败由 pipeline 回退。"""
from pathlib import Path

from app.core.ingest.markdown import parse_markdown
from app.core.ingest.models import ParsedDocument
from app.core.ingest.pdf import detect_scanned

# 惰性导入：首次调用时才加载 docling，避免 pipeline import 变重；未安装时为 None。
DocumentConverter = None


def parse_pdf_docling(path: Path) -> ParsedDocument:
    """Docling 转换 PDF → markdown，再解析为 blocks/headings（复用 markdown 解析器）。"""
    global DocumentConverter
    if DocumentConverter is None:
        from docling.document_converter import DocumentConverter as _DC

        DocumentConverter = _DC
    markdown = DocumentConverter().convert(str(path)).document.export_to_markdown()
    return parse_markdown(
        markdown,
        kind="pdf",
        engine_used="docling",
        ocr_used=False,
        needs_ocr=detect_scanned(markdown),
    )
