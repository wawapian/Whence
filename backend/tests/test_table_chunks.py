from app.core.ingest import chunking
from app.core.ingest.pipeline import ingest
from app.core.tables import ParsedTable


def test_chunk_tables_prefix_grouping_and_parent():
    t = ParsedTable(headers=["课程", "学分"], rows=[[f"课{i}", f"{i}.0"] for i in range(50)])
    chunks = chunking.chunk_tables([t], max_chars=100)
    assert len(chunks) > 1
    assert all(c.text.startswith("表头：课程 | 学分\n") for c in chunks)
    assert all(c.heading_path == "表格" and c.parent_text.startswith("表格（50行×2列") for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for i in range(50):
        assert f"课{i} | {i}.0" in joined


def test_chunk_tables_skips_table_without_rows():
    assert chunking.chunk_tables([ParsedTable(headers=["a"], rows=[])]) == []


def test_ingest_appends_table_chunks_with_continuous_seq(tmp_path):
    from helpers_pdf import make_table_pdf
    p = tmp_path / "t.pdf"; make_table_pdf(p)
    result = ingest(p, ocr_name="off")
    assert result.tables and [c.seq for c in result.chunks] == list(range(len(result.chunks)))
    assert any(c.text.startswith("表头：course | credit") for c in result.chunks)
