"""文档解析：从 .docx / .pdf 提取文本（本地，不上云）。"""
import re
from pathlib import Path

# 乱码质量门：对无 ToUnicode 映射的中文 PDF，不同解析器表现不一——pymupdf/docling 能读，
# 旧 pypdf 路会把内容流字节当文本吐出（实测 7588 字符乱码，含大量控制字符）。与其把乱码
# 写进记录内容与索引，不如如实视为「解析不出」（回填空，记录走 raw_only）。
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
_CONTROL_MAX_RATIO = 0.01


def _pdf_text(file_path: Path) -> str:
    """PDF 文字层抽取（pymupdf，与文档解析管线同引擎口径）。"""
    import pymupdf as fitz

    doc = fitz.open(str(file_path))
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _looks_like_text(text: str) -> bool:
    """乱码门：控制字符/替换符占比超阈值 → 判定为乱码，不得当作正文。"""
    if not text:
        return False
    garbage = len(_CONTROL_RE.findall(text))
    return garbage <= len(text) * _CONTROL_MAX_RATIO


def parse_document(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    try:
        if ext == ".docx":
            import docx

            d = docx.Document(str(file_path))
            text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
        elif ext == ".pdf":
            text = _pdf_text(file_path)
        else:
            return ""
    except Exception:
        return ""  # 解析失败（损坏文件等）：返回空，不影响记录
    return text if _looks_like_text(text) else ""
