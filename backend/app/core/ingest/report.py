"""统一识别报告：格式类型 + 关键表征，供各处理单元消费。"""
from dataclasses import dataclass
from pathlib import Path

from app.core.ingest import detect, pdf


@dataclass
class DataReport:
    """一次识别的产出：kind + suffix + 文字层。

    后续字段（表格/公式/图片）在对应处理单元落地时增补。
    """

    kind: str
    suffix: str
    has_text_layer: bool | None


def has_text_layer(path: Path) -> bool:
    """任一页 strip 后非空即有文字层（电子版 PDF）；读不了（缺失/损坏）按无文字层处理。"""
    try:
        return any(text.strip() for text in pdf.parse_pdf_pages(path))
    except Exception:
        return False


def build_report(path: Path) -> DataReport:
    """识别 path 的 kind/suffix；has_text_layer 仅 PDF 判定，其余类型为 None。"""
    kind = detect.detect_format(path)
    return DataReport(
        kind=kind,
        suffix=path.suffix.lower(),
        has_text_layer=has_text_layer(path) if kind == "pdf" else None,
    )
