from pathlib import Path

import pymupdf as fitz
from PIL import Image

from app.core.ingest import ocr, pdf, processors


class _FakeOcr:
    def recognize(self, image: bytes) -> str:
        return "查重报告 相似度15%"


def _make_pdf_with_image(tmp_path) -> Path:
    png = tmp_path / "r.png"
    Image.new("RGB", (1, 1), "white").save(png)
    p = tmp_path / "f.pdf"
    d = fitz.open()
    page = d.new_page()
    page.insert_text((72, 72), "这是一段正文", fontname="china-s")
    page.insert_image(fitz.Rect(72, 100, 200, 200), filename=str(png))
    d.save(str(p))
    d.close()
    return p


def test_pdf_image_ocr_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "get_ocr_engine", lambda name: _FakeOcr())
    parsed = pdf.parse_pdf_with_images(_make_pdf_with_image(tmp_path), ocr_name="rapidocr")
    assert "查重报告" in parsed.full_text
    assert parsed.ocr_used is True


def test_pdf_ocr_off(tmp_path):
    parsed = pdf.parse_pdf_with_images(_make_pdf_with_image(tmp_path), ocr_name="off")
    # 无 OCR 文字，但正文在
    assert "查重报告" not in parsed.full_text
    assert "这是一段正文" in parsed.full_text
    assert parsed.ocr_used is False


def _make_pdf_image_only(tmp_path) -> Path:
    png = tmp_path / "r.png"
    Image.new("RGB", (1, 1), "white").save(png)
    p = tmp_path / "img_only.pdf"
    d = fitz.open()
    page = d.new_page()
    page.insert_image(fitz.Rect(72, 100, 200, 200), filename=str(png))
    d.save(str(p))
    d.close()
    return p


def test_pdf_image_only_page_not_ocrd(tmp_path, monkeypatch):
    # 仅图片、无文字层的扫描页：不在这里 OCR 嵌入图片，保留 needs_ocr=True 交给 _ocr_pdf_pages
    monkeypatch.setattr(ocr, "get_ocr_engine", lambda name: _FakeOcr())
    parsed = pdf.parse_pdf_with_images(_make_pdf_image_only(tmp_path), ocr_name="rapidocr")
    assert "查重报告" not in parsed.full_text
    assert parsed.ocr_used is False
    assert parsed.needs_ocr is True


def _make_pdf_text_with_image_and_scanned(tmp_path) -> Path:
    png = tmp_path / "r.png"
    Image.new("RGB", (1, 1), "white").save(png)
    p = tmp_path / "mixed_img.pdf"
    d = fitz.open()
    page1 = d.new_page()
    page1.insert_text((72, 72), "这是正文", fontname="china-s")
    page1.insert_image(fitz.Rect(72, 100, 200, 200), filename=str(png))
    d.new_page().insert_image(fitz.Rect(72, 100, 200, 200), filename=str(png))  # 仅图片的扫描页
    d.save(str(p))
    d.close()
    return p


def test_mixed_pdf_keeps_image_ocr_and_ocrs_scanned(tmp_path, monkeypatch):
    # 文字页的嵌入图片 OCR 文字 + 扫描页的 pixmap OCR 都应保留在最终 full_text 中。
    calls: list[int] = []

    class _FakeOcr:
        def recognize(self, image: bytes) -> str:
            calls.append(1)
            return "截图文字" if len(calls) == 1 else "扫描页OCR"

    monkeypatch.setattr(ocr, "get_ocr_engine", lambda name: _FakeOcr())
    parsed = processors._parse_pdf(_make_pdf_text_with_image_and_scanned(tmp_path), "rapidocr", "off")
    assert "这是正文" in parsed.full_text
    assert "截图文字" in parsed.full_text   # 文字页嵌入图片的 OCR 结果保留
    assert "扫描页OCR" in parsed.full_text  # 扫描页被 pixmap OCR
    assert parsed.ocr_used is True
