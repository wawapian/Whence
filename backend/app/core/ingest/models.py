"""Ingestion 流水线内存数据结构。"""
from dataclasses import dataclass, field

from app.core.tables import ParsedTable


@dataclass
class Heading:
    """文档标题层级。"""

    level: int
    text: str


@dataclass
class Block:
    """解析后的文档块。"""

    type: str  # "paragraph" | "heading" | "table" | "image"
    text: str
    heading_path: str | None = None
    page: int | None = None


@dataclass
class ImageRef:
    """文档内嵌图片及其 OCR 文本。"""

    blob: bytes
    page: int | None = None
    position: str | None = None
    ocr_text: str = ""


@dataclass
class ParsedDocument:
    """解析后的完整文档。"""

    kind: str  # word | pdf | image | text
    title: str | None
    full_text: str
    headings: list[Heading]
    blocks: list[Block]
    engine_used: str
    ocr_used: bool = False
    needs_ocr: bool | None = None
    images: list[ImageRef] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)


@dataclass
class TextChunk:
    """文档分块文本（区别于 DB 模型 Chunk）。"""

    text: str
    heading_path: str | None
    page: int | None
    parent_text: str
    seq: int
