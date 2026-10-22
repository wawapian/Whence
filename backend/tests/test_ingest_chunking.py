from app.core.ingest import chunking
from app.core.ingest.models import Block, Heading, ParsedDocument


def test_auto_structural_when_headings():
    doc = ParsedDocument(kind="word", title="T", full_text="", headings=[Heading(1, "一")],
        blocks=[Block("heading", "一", heading_path="一"),
                Block("paragraph", "甲" * 1000, heading_path="一")], engine_used="python-docx")
    chunks = chunking.chunk_document(doc)  # auto → structural
    assert any(c.heading_path == "一" for c in chunks)
    assert sum(1 for c in chunks if "甲" in c.text) >= 2   # 超长拆多块


def test_auto_recursive_when_no_headings():
    doc = ParsedDocument(kind="text", title=None, full_text="", headings=[],
        blocks=[Block("paragraph", "乙" * 3000)], engine_used="txt")
    chunks = chunking.chunk_document(doc)  # auto → recursive
    assert all(len(c.text) <= 600 + 60 for c in chunks)
    assert len(chunks) >= 5


def test_parent_text_preserved():
    doc = ParsedDocument(kind="word", title="T", full_text="", headings=[Heading(1, "一")],
        blocks=[Block("heading", "一", heading_path="一"),
                Block("paragraph", "丙" * 500, heading_path="一")], engine_used="python-docx")
    chunks = chunking.chunk_document(doc, strategy="structural", max_chars=600, overlap=60)
    assert all(len(c.parent_text) > 0 for c in chunks)
