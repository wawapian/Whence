"""表格统计通用引擎（Task 5）：多条件 AND / 分组 / avg / 排序取前 N。

legacy 的 Filter/apply_filter/compose_answer 不在此文件覆盖（tests/test_stats.py 管它们）。
"""
from app.core.stats import apply_query, build_query, render_query

HEADERS = ["姓名", "语文", "数学", "学历", "月薪", "入职"]
ROWS = [
    ["甲", "85", "90", "本科", "12000", "2026-03-05"],
    ["乙", "78", "88", "本科", "9000", "2026-04-11"],
    ["丙", "92", "76", "硕士", "15000", "2026-02-20"],
    ["丁", "80", "82", "硕士", "18000", "2026-03-18"],
]


def test_multicond_and_filter():
    q = build_query("t1", HEADERS, filters=[
        {"column": "语文", "op": ">=", "value": "80"},
        {"column": "数学", "op": ">=", "value": "80"},
    ], aggregate="count")
    assert q is not None
    out = apply_query(HEADERS, ROWS, q)
    assert out.aggregate_value == 2  # 甲（85/90）、丁（80/82）


def test_group_by_count_and_sum():
    q = build_query("t1", HEADERS, aggregate="sum", sum_column="月薪", group_by="学历")
    out = apply_query(HEADERS, ROWS, q)
    assert out.groups == [("本科", 21000.0, 2), ("硕士", 33000.0, 2)]
    text = render_query(out, suspect=False)
    assert "分组统计（按「学历」，2 组）：" in text
    assert "- 本科：合计 21000（2 行）" in text

    q2 = build_query("t1", HEADERS, group_by="学历")  # aggregate 缺省 → 自动按 count
    out2 = apply_query(HEADERS, ROWS, q2)
    assert out2.groups == [("本科", 2, 2), ("硕士", 2, 2)]


def test_avg_skips_unparsable_cells():
    rows = ROWS + [["戊", "70", "—", "博士", "20000", "2026-05-01"]]
    q = build_query("t1", HEADERS, aggregate="avg", sum_column="数学")
    out = apply_query(HEADERS, rows, q)
    assert out.aggregate_value == 84.0  # (90+88+76+82)/4
    assert out.skipped == 1
    # 「N 行」= 匹配行数（与 legacy sum 渲染同口径：含未计入求和的行）
    assert "均值结果：84（列「数学」，5 行）" in render_query(out, suspect=False)
    assert "另有 1 个单元格无法解析为数值，未计入" in render_query(out, suspect=False)

    tiny = ["成绩"]
    none_q = build_query("t1", tiny, aggregate="avg", sum_column="成绩")
    none_out = apply_query(tiny, [["—"], ["缺"]], none_q)
    assert none_out.aggregate_value is None
    assert "无法计算" in render_query(none_out, suspect=False)


def test_order_by_limit_topn():
    q = build_query("t1", HEADERS, aggregate="none", order_by="月薪", desc=True, limit=2)
    out = apply_query(HEADERS, ROWS, q)
    assert [r[0] for r in out.rows] == ["丁", "丙"]
    assert out.rows_truncated is True
    assert "…共 4 行" in render_query(out, suspect=False)


def test_time_range_via_two_conds():
    q = build_query("t1", HEADERS, filters=[
        {"column": "入职", "op": ">=", "value": "2026-03-01"},
        {"column": "入职", "op": "<=", "value": "2026-03-31"},
    ], aggregate="count")
    assert apply_query(HEADERS, ROWS, q).aggregate_value == 2  # 甲、丁


def test_filters_and_single_cond_conflict_rejected():
    assert build_query("t1", HEADERS, filters=[{"column": "语文", "op": ">=", "value": "80"}],
                       column="数学", op=">=", value="80", aggregate="count") is None


def test_unknown_column_rejected():
    assert build_query("t1", HEADERS, filters=[{"column": "不存在", "op": "==", "value": "x"}],
                       aggregate="count") is None
    assert build_query("t1", HEADERS, aggregate="count", group_by="不存在") is None
    assert build_query("t1", HEADERS, aggregate="count", order_by="不存在") is None


def test_order_by_unparsable_cells_go_last_and_disclosed():
    """终审 C1：列里混了不可解析单元格时，desc 排序不得把「面议」排到最前、也不得按字符串序排数字。"""
    headers = ["姓名", "月薪"]
    rows = [["甲", "12000"], ["乙", "9000"], ["丙", "面议"]]
    q = build_query("t1", headers, aggregate="none", order_by="月薪", desc=True, limit=2)
    out = apply_query(headers, rows, q)
    assert [r[0] for r in out.rows] == ["甲", "乙"]  # 数值降序；「面议」恒末尾、不进前 N
    assert out.order_skipped == 1
    text = render_query(out, suspect=False)
    assert "另有 1 行该列无法解析为数值，未参与排序。" in text

    q2 = build_query("t1", headers, aggregate="none", order_by="月薪", desc=False, limit=2)
    assert [r[0] for r in apply_query(headers, rows, q2).rows] == ["乙", "甲"]


def test_group_order_none_value_goes_last_under_desc():
    """终审 C1（分组同型）：聚合值不可计算的组不得冒充"最高组"挤进前 N。"""
    headers = ["部门", "金额"]
    rows = [["甲", "100"], ["乙", "面议"], ["丙", "300"]]
    q = build_query("t1", headers, aggregate="sum", sum_column="金额", group_by="部门",
                    order_by="金额", desc=True, limit=2)
    out = apply_query(headers, rows, q)
    assert [g[0] for g in out.groups] == ["丙", "甲"]  # 乙（无法计算）恒末尾


def test_group_by_over_cap_truncated_in_render():
    rows = [[f"组{i}", "1"] for i in range(35)]
    q = build_query("t1", ["组", "值"], aggregate="sum", sum_column="值", group_by="组")
    out = apply_query(["组", "值"], rows, q)
    text = render_query(out, suspect=False)
    assert "…共 35 组" in text


def test_filter_skipped_and_suspect_disclosure():
    rows = ROWS + [["己", "优秀", "90", "本科", "8000", "2026-06-01"]]
    q = build_query("t1", HEADERS, filters=[{"column": "语文", "op": ">=", "value": "80"}],
                    aggregate="count")
    out = apply_query(HEADERS, rows, q)
    assert out.filter_skipped == 1
    text = render_query(out, suspect=True)
    assert "另有 1 行该列无法解析为数值，未纳入筛选。" in text
    assert text.endswith("（该表格解析可能不完整）")


def test_single_cond_render_matches_legacy_format():
    """单条件渲染与既有工具文案逐字一致（既有测试零改动的锚点）。"""
    q = build_query("t1", HEADERS, aggregate="sum", sum_column="月薪")
    out = apply_query(HEADERS, ROWS, q)
    assert render_query(out, suspect=False) == "求和结果：54000（列「月薪」，4 行）"  # 12000+9000+15000+18000

    q2 = build_query("t1", HEADERS, column="学历", op="==", value="本科", aggregate="none")
    out2 = apply_query(HEADERS, ROWS, q2)
    text = render_query(out2, suspect=False)
    assert text.startswith("筛选结果（2 条）：\n1. 甲 | 85 | 90 | 本科 | 12000 | 2026-03-05")

    q3 = build_query("t1", HEADERS, column="学历", op="==", value="博士", aggregate="count")
    assert render_query(apply_query(HEADERS, ROWS, q3), suspect=False) == "没有符合条件的记录。"
