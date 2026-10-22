import docx as _docx
import pymupdf as fitz


def test_report_pdf_with_text_layer(tmp_path):
    from helpers_pdf import make_table_pdf
    from app.core.ingest.report import build_report
    p = tmp_path / "t.pdf"; make_table_pdf(p)
    r = build_report(p)
    assert r.kind == "pdf" and r.suffix == ".pdf" and r.has_text_layer is True


def test_report_pdf_without_text_layer(tmp_path):
    from app.core.ingest.report import build_report
    doc = fitz.open(); doc.new_page()
    p = tmp_path / "blank.pdf"; doc.save(str(p)); doc.close()
    r = build_report(p)
    assert r.kind == "pdf" and r.has_text_layer is False


def test_report_other_kinds_have_none(tmp_path):
    from app.core.ingest.report import build_report
    d = _docx.Document(); d.add_paragraph("x")
    docx_p = tmp_path / "a.docx"; d.save(str(docx_p))
    png = tmp_path / "b.png"; png.write_bytes(b"\x89PNG\r\n\x1a\n")
    txt = tmp_path / "c.txt"; txt.write_text("hello", encoding="utf-8")
    assert build_report(docx_p).kind == "word"
    assert build_report(docx_p).has_text_layer is None
    assert build_report(png).has_text_layer is None
    assert build_report(txt).has_text_layer is None


def test_report_unknown_kind_other(tmp_path):
    from app.core.ingest.report import build_report
    f = tmp_path / "d.xyz"; f.write_bytes(b"\x00\x01\x02\x03")
    r = build_report(f)
    assert r.kind == "other" and r.suffix == ".xyz" and r.has_text_layer is None
