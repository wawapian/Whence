from app.core.ingest import pdf


def test_parse_pdf_extracts_text(tmp_path):
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "身份证放在书房", fontname="china-s")
    p = tmp_path / "a.pdf"; doc.save(str(p)); doc.close()
    parsed = pdf.parse_pdf(p)
    assert "身份证" in parsed.full_text
    assert parsed.needs_ocr is False


def test_detect_scanned():
    assert pdf.detect_scanned("") is True
    assert pdf.detect_scanned("   \n") is True
    assert pdf.detect_scanned("有文字") is False


def test_parse_pdf_mixed_pages_needs_ocr(tmp_path):
    """混合 PDF（首页有文字 + 空页）应判 needs_ocr=True（按页判定，而非全文）。"""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "第一页文字", fontname="china-s")
    doc.new_page()  # 第二页为空 → 扫描页
    p = tmp_path / "mixed.pdf"; doc.save(str(p)); doc.close()
    parsed = pdf.parse_pdf(p)
    assert parsed.needs_ocr is True
    assert "第一页文字" in parsed.full_text


def test_parse_pdf_mixed_pages_ocr_merges(tmp_path, monkeypatch):
    """混合 PDF：仅 OCR 空页并与原生文字层合并，有文字的页不被替换。"""
    import fitz
    from app.core.ingest import processors
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "第一页文字", fontname="china-s")
    doc.new_page()  # 空页
    p = tmp_path / "mixed.pdf"; doc.save(str(p)); doc.close()

    calls = []
    class FakeEngine:
        def recognize(self, image: bytes) -> str:
            calls.append(1)
            return "OCR页文字"

    monkeypatch.setattr(processors.ocr, "get_ocr_engine", lambda name: FakeEngine())
    parsed = processors._parse_pdf(p, "rapidocr", "off")
    assert "第一页文字" in parsed.full_text  # 原生文字层保留
    assert "OCR页文字" in parsed.full_text   # OCR 结果合并进来
    assert len(calls) == 1                   # 只 OCR 空页
    assert parsed.ocr_used is True
