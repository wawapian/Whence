"""PDF 高级档（docling → markdown）测试。"""


def test_docling_engine_used(monkeypatch, tmp_path):
    import app.core.ingest.pdf_docling as dl

    class _Doc:
        def export_to_markdown(self):
            return "# 标题\n\n公式 $a^2+b^2=c^2$\n\n|A|B|\n|---|---|\n|1|2|"

    class _Conv:
        def __init__(self):
            self.document = _Doc()

        def convert(self, source):
            return self

    monkeypatch.setattr(dl, "DocumentConverter", lambda: _Conv())
    parsed = dl.parse_pdf_docling(tmp_path / "f.pdf")
    assert "$a^2+b^2=c^2$" in parsed.full_text
    assert parsed.engine_used == "docling"


def test_docling_missing_falls_back(monkeypatch, tmp_path):
    import app.core.ingest.processors as proc
    from app.core.ingest.models import ParsedDocument

    monkeypatch.setattr(
        proc, "parse_pdf_docling", lambda p: (_ for _ in ()).throw(ImportError("no docling"))
    )
    monkeypatch.setattr(
        proc.pdf,
        "parse_pdf",
        lambda p: ParsedDocument(
            kind="pdf",
            title=None,
            full_text="轻量路径",
            headings=[],
            blocks=[],
            engine_used="pymupdf",
            ocr_used=False,
            needs_ocr=False,
        ),
    )
    # advanced_engine=docling 但 docling 抛 ImportError → 走 parse_pdf 轻量路径，engine_used="fallback"
    parsed = proc._parse_pdf(tmp_path / "f.pdf", "off", "docling")
    assert parsed.engine_used == "fallback"
    assert parsed.full_text == "轻量路径"
