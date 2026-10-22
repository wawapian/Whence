"""Processor 统一接口与摄取上下文（数据层骨架）。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.ingest.models import ParsedDocument
from app.core.ingest.report import DataReport


@dataclass
class IngestContext:
    """一次摄取透传给处理单元的上下文（引擎选择等）。"""

    ocr_name: str = "rapidocr"
    advanced_engine: str = "off"


class Processor(Protocol):
    """处理单元统一接口：`applies` 声明式适用性 + `run` 统一执行入口。"""

    name: str

    def applies(self, report: DataReport) -> bool:
        """依据识别报告声明本单元是否适用。"""
        ...

    def run(self, path: Path, report: DataReport, ctx: IngestContext) -> ParsedDocument:
        """解析 path，返回 ParsedDocument。"""
        ...
