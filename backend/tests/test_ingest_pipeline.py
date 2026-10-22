from app.core.ingest import pipeline


def test_ingest_text_gb18030_fallback(tmp_path):
    """GBK/GB18030 编码的 .txt：UTF-8 解码失败后回退 gb18030，正文应被提取。"""
    p = tmp_path / "gbk.txt"
    p.write_bytes("中文内容 GBK 编码".encode("gb18030"))
    result = pipeline.ingest(p)
    assert "中文内容" in result.parsed.full_text


def test_ingest_corrupt_file(tmp_path):
    p = tmp_path / "bad.docx"; p.write_bytes(b"not a docx")
    result = pipeline.ingest(p)   # 不抛异常
    assert result.parsed.full_text == ""


def test_ingest_word_routes_to_docx_and_chunks(tmp_path, monkeypatch):
    from app.core.ingest import docx
    from app.core.ingest.models import Block, Heading, ParsedDocument
    fake = ParsedDocument(
        kind="word", title="T", full_text="一\n甲" * 50,
        headings=[Heading(1, "一")],
        blocks=[Block("heading", "一", heading_path="一"),
                Block("paragraph", "甲" * 500, heading_path="一")],
        engine_used="python-docx",
    )
    monkeypatch.setattr(docx, "parse_docx", lambda p, ocr_name="off": fake)
    p = tmp_path / "x.docx"; p.write_bytes(b"dummy")  # 让 detect_format 能打开、按扩展名判 word
    result = pipeline.ingest(p)
    assert result.parsed.engine_used == "python-docx"
    assert len(result.chunks) >= 1


def test_unknown_format_goes_to_unsupported(tmp_path):
    """魔数/扩展名都识别不出的文件：照常入库不炸，engine_used 如实标注 unsupported。"""
    p = tmp_path / "a.xyz"; p.write_bytes(b"\x00\x01\x02\x03")
    result = pipeline.ingest(p)
    assert result.parsed.kind == "other"
    assert result.parsed.engine_used == "unsupported"
    assert result.chunks == []


def test_empty_text_file_ok(tmp_path):
    """0 字节 .txt：照常走链不抛异常，kind=="text"、full_text==""。"""
    p = tmp_path / "empty.txt"; p.write_bytes(b"")
    result = pipeline.ingest(p)
    assert result.parsed.kind == "text"
    assert result.parsed.full_text == ""
