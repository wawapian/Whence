from app.core import tables


def test_parse_standard_table():
    text = "| 课程 | 学分 |\n|---|---|\n| 体育 | 1.0 |\n| 就业指导 | 0.5 |"
    t = tables.parse_markdown_table(text)
    assert t.headers == ["课程", "学分"]
    assert t.rows == [["体育", "1.0"], ["就业指导", "0.5"]]


def test_parse_empty_cell_and_ragged_row():
    text = "| a | b | c |\n|---|---|---|\n| 1 |  | 3 |\n| x | y |"
    t = tables.parse_markdown_table(text)
    assert t.rows[0] == ["1", "", "3"]
    assert t.rows[1] == ["x", "y", ""]  # 短行补空


def test_parse_no_table_returns_none():
    assert tables.parse_markdown_table("普通段落文本\n没有表格") is None


def test_parse_returns_first_table_only():
    text = "| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |"
    assert tables.parse_markdown_table(text).headers == ["a"]


def test_parse_table_not_at_block_start():
    text = "说明文字\n| a | b |\n|---|---|\n| 1 | 2 |"
    t = tables.parse_markdown_table(text)
    assert t.rows == [["1", "2"]]


def test_table_summary_format():
    t = tables.ParsedTable(headers=["课程", "学分", "成绩"], rows=[["a", "b", "c"], ["d", "e", "f"]])
    assert tables.table_summary(t) == "表格（2行×3列：课程、学分、成绩）"
