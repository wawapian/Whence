from app.core import tables


def test_clean_cell_preserves_data_semantics():
    assert tables._clean_cell("课 程\n名 称") == "课程名称"      # CJK 间空白删除
    assert tables._clean_cell("40\n60") == "40 60"               # 数字间不得拼接
    assert tables._clean_cell("通识必修 2.0") == "通识必修 2.0"   # 粘格原样保留
    assert tables._clean_cell("Data Science") == "Data Science"  # 英文词间空格保留
    assert tables._clean_cell("必修\n必修") == "必修"            # 合并格重复段落去重


def test_normalize_single_table_cleans_and_drops_empty_rows_cols():
    grid = [["课 程\n名 称", "", "学\n分"], ["", "", ""], ["大学计算机", "", "2.0"]]
    ts = tables.normalize_grid(grid)
    assert len(ts) == 1
    assert ts[0].headers == ["课程名称", "学分"]
    assert ts[0].rows == [["大学计算机", "2.0"]]
    assert ts[0].suspect is False


def test_split_title_row_detects_caption_above_header():
    """标题行识别：首行只有 1 个非空格、第二行 ≥2 个非空 → 首行是标题，不是表头。

    真实形态（会话 26 评测实测）：xlsx 里 A1 直接打字当标题（未合并），第二行才是
    「姓名|语文|数学|英语」——旧解析把标题当表头，列名全丢、表被标 suspect。
    """
    grid = [["三年级四班（东校区）期末成绩表", "", "", ""],
            ["姓名", "语文", "数学", "英语"],
            ["甲", "90", "80", "70"]]
    title, rest = tables.split_title_row(grid)
    assert title == "三年级四班（东校区）期末成绩表"
    assert rest[0] == ["姓名", "语文", "数学", "英语"]
    assert len(rest) == 2


def test_split_title_row_keeps_normal_grid():
    grid = [["课程", "学分"], ["体育", "0.5"]]
    assert tables.split_title_row(grid) == (None, grid)
    # 首行只有一格、第二行也仅一格 → 不像「标题 + 表头」，原样返回
    grid2 = [["说明", ""], ["值", ""]]
    assert tables.split_title_row(grid2) == (None, grid2)
    assert tables.split_title_row([["单行表"]]) == (None, [["单行表"]])


def test_normalize_splits_side_by_side_subtables():
    grid = [["课程", "学分", "课程", "学分", "课程", "学分"],
            ["a", "1.0", "b", "2.0", "c", "3.0"]]
    ts = tables.normalize_grid(grid)
    assert [t.headers for t in ts] == [["课程", "学分"]] * 3
    assert [t.rows[0][0] for t in ts] == ["a", "b", "c"]


def test_normalize_no_repeat_first_header_keeps_single_table():
    assert len(tables.normalize_grid([["课程", "学分"], ["体育", "1.0"]])) == 1


def test_normalize_empty_or_header_only_returns_empty():
    assert tables.normalize_grid([]) == []
    assert tables.normalize_grid([["课程", "学分"]]) == []


def test_normalize_suspect_on_empty_or_duplicate_header_name():
    assert tables.normalize_grid([["课程", "", "成绩"], ["a", "1", "b"]])[0].suspect is True
    assert tables.normalize_grid([["课程", "成绩", "成绩"], ["a", "1", "b"]])[0].suspect is True


def test_normalize_suspect_glued_cells_threshold():
    glued = [["课程", "学分"]] + [[f"c{i}", "通识必修 2.0"] for i in range(2)] \
            + [[f"d{i}", "1.0"] for i in range(8)]
    assert tables.normalize_grid(glued)[0].suspect is True     # 2/10 = 20% ≥ 10%
    one = [["课程", "学分"]] + [[f"c{i}", "通识必修 2.0"] for i in range(1)] \
          + [[f"d{i}", "1.0"] for i in range(20)]
    assert tables.normalize_grid(one)[0].suspect is False      # 仅 1 个 < 2 → 不标记


def test_normalize_split_segment_drops_ghost_empty_rows():
    # 某行在主网格非空、但切分后在子表段内全空 → 不得留在该段（幽灵空行）
    grid = [["课程", "学分", "课程", "学分"],
            ["a", "1.0", "x", "9.0"],
            ["b", "2.0", "", ""],
            ["c", "3.0", "", ""]]
    ts = tables.normalize_grid(grid)
    assert [t.rows for t in ts] == [[["a", "1.0"], ["b", "2.0"], ["c", "3.0"]],
                                    [["x", "9.0"]]]


def test_normalize_ghost_rows_do_not_dilute_suspect_ratio():
    # 第二子表真实数据行 2 行全粘连（100%）；20 个幽灵空行若留下会稀释成 2/22 < 10% → 误判 False
    grid = [["课程", "学分", "课程", "学分"]]
    grid += [[f"g{i}", "1.0", "通识必修 2.0", "1.0"] for i in range(2)]
    grid += [[f"c{i}", "1.0", "", ""] for i in range(20)]
    ts = tables.normalize_grid(grid)
    assert ts[1].rows == [["通识必修 2.0", "1.0"], ["通识必修 2.0", "1.0"]]
    assert ts[1].suspect is True


def test_normalize_suspect_uses_row_ratio_on_wide_table():
    # 16 列宽表、每行恰 1 个粘格：行级口径 10/10 → True；按格子数只有 10/160 ≈ 6% → 会漏报
    grid = [["c%d" % j for j in range(16)]] + [
        ["通识必修 2.0" if j == 3 else f"v{j}" for j in range(16)] for _ in range(10)]
    assert tables.normalize_grid(grid)[0].suspect is True
