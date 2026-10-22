import docx as _docx
from app.core.ingest import detect


def test_detect_format(tmp_path):
    d = _docx.Document(); d.add_paragraph("x")
    docx = tmp_path / "a.docx"; d.save(str(docx))
    assert detect.detect_format(docx) == "word"
    pdf = tmp_path / "b.pdf"; pdf.write_bytes(b"%PDF-1.7\n")
    assert detect.detect_format(pdf) == "pdf"
    png = tmp_path / "c.png"; png.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert detect.detect_format(png) == "image"


def test_detect_spoofed_extension(tmp_path):
    f = tmp_path / "fake.jpg"
    f.write_bytes(b"%PDF-1.7\n")   # 实际是 PDF
    assert detect.detect_format(f) == "pdf"
