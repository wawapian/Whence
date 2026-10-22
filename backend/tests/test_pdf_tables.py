import pymupdf

from app.core.ingest import pdf_tables
from helpers_pdf import make_table_pdf


def test_extract_pdf_tables_basic(tmp_path):
    p = tmp_path / "t.pdf"; make_table_pdf(p)
    ts = pdf_tables.extract_pdf_tables(p)
    assert len(ts) == 1
    assert ts[0].headers == ["course", "credit"]
    assert ts[0].rows == [["PE", "1.0"]]


def test_extract_pdf_tables_no_table_returns_empty(tmp_path):
    doc = pymupdf.open(); page = doc.new_page(); page.insert_text((72, 72), "plain text")
    p = tmp_path / "plain.pdf"; doc.save(str(p)); doc.close()
    assert pdf_tables.extract_pdf_tables(p) == []


def test_extract_pdf_tables_broken_file_returns_empty(tmp_path):
    p = tmp_path / "bad.pdf"; p.write_bytes(b"not a pdf")
    assert pdf_tables.extract_pdf_tables(p) == []


def test_table_regions_returns_page_bboxes(tmp_path):
    p = tmp_path / "t.pdf"; make_table_pdf(p)
    regions = pdf_tables.table_regions(p)
    assert 0 in regions and len(regions[0]) == 1
    assert pdf_tables.table_regions(tmp_path / "nope.pdf") == {}


def test_is_likely_table_gate():
    """表样自检：短格子真表放行；段落感/图内长文/单行列的"表"拒绝。"""
    assert pdf_tables._is_likely_table([["课程", "学分"], ["体育", "0.5"]]) is True
    # 表头是长句子（正文/图内标注被切成"表"）
    assert pdf_tables._is_likely_table(
        [["这是一整段大约超过二十四个字的正文内容被误当成了表头行"], ["x", "y"]]
    ) is False
    # 单元格整体段落感（中位超 30）
    para = [["这是差不多四十来字的一段完整描述性正文内容啊"], ["另一段差不多四十来字的完整描述性正文内容"]]
    assert pdf_tables._is_likely_table(para) is False
    # 单行 / 单列
    assert pdf_tables._is_likely_table([["只有一行", "两格"]]) is False
    assert pdf_tables._is_likely_table([["一"], ["二"], ["三"]]) is False
