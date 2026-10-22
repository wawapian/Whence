"""Processor 接口 + 注册表：applies 适用性表与 build_chain 链选择（不碰真文件）。"""
import pytest

from app.core.ingest.report import DataReport


def _report(kind: str, suffix: str = ".bin") -> DataReport:
    """构造识别报告：本测试只关心 kind 与 suffix，文字层填中性值。"""
    return DataReport(kind=kind, suffix=suffix, has_text_layer=None)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("word", "WordProcessor"),
        ("pdf", "PdfProcessor"),
        ("image", "ImageProcessor"),
        ("text", "TextProcessor"),
        ("other", None),   # 四类之外无人「声明式」接管 → 由兜底单元在链空时接盘
        ("video", None),
    ],
)
def test_applies_table(kind, expected):
    """表驱动：四类各恰有一个单元声明适用；其余 kind 无单元声明（兜底不参与匹配）。"""
    from app.core.ingest import processors

    report = _report(kind)
    applied = [type(p).__name__ for p in processors.PROCESSORS if p.applies(report)]
    assert applied == ([expected] if expected else [])


def test_build_chain_single_link_per_kind():
    """每类恰返回 1 个链接；四类之外（无人接管）→ UnknownProcessor 兜底。"""
    from app.core.ingest import processors

    cases = [
        ("word", processors.WordProcessor),
        ("pdf", processors.PdfProcessor),
        ("image", processors.ImageProcessor),
        ("text", processors.TextProcessor),
        ("other", processors.UnknownProcessor),   # 链空 → 兜底
        ("video", processors.UnknownProcessor),
    ]
    for kind, expected_cls in cases:
        chain = processors.build_chain(_report(kind))
        assert len(chain) == 1, f"{kind}: 链长应恰为 1，实为 {len(chain)}"
        assert isinstance(chain[0], expected_cls)


def test_registry_order_word_to_unknown():
    """注册顺序即链优先顺序：Word → Pdf → Image → Excel → Text → Unknown（兜底末位）。"""
    from app.core.ingest import processors

    assert [type(p).__name__ for p in processors.PROCESSORS] == [
        "WordProcessor",
        "PdfProcessor",
        "ImageProcessor",
        "ExcelProcessor",
        "TextProcessor",
        "UnknownProcessor",
    ]


def test_register_new_processor_is_add_class_plus_one_line(tmp_path, monkeypatch):
    """试金石（spec §8）：新增格式 = 加一个处理单元 + 注册一行，其余零改动。

    假「未来格式解析器」只按 `report.suffix` 声明适用性（用尚未被认领的 .ods），
    走真 ingest 全链路；detect / report / pipeline 一行未改。
    """
    from app.core.ingest import pipeline, processors
    from app.core.ingest.models import ParsedDocument

    class SheetStub:
        """假「未来格式解析器」：仅声明 .ods 归它管。"""

        name = "sheet-stub"

        def applies(self, report: DataReport) -> bool:
            return report.suffix == ".ods"

        def run(self, path, report, ctx):
            return ParsedDocument(
                kind="ods", title=None, full_text="单元格内容 学习率 0.001",
                headings=[], blocks=[], engine_used="stub", ocr_used=False, needs_ocr=None,
            )

    # 「注册一行」：插在兜底单元之前
    monkeypatch.setattr(
        processors, "PROCESSORS",
        [*processors.PROCESSORS[:-1], SheetStub(), processors.PROCESSORS[-1]],
    )

    ods = tmp_path / "表格.ods"
    ods.write_bytes(b"PK\x03\x04fake-ods")
    result = pipeline.ingest(ods, ocr_name="off")

    assert result.parsed.engine_used == "stub"      # 新单元接管（此前会归 unknown / unsupported）
    assert "单元格内容" in result.parsed.full_text
    assert result.chunks and result.chunks[0].text  # 照常切片
