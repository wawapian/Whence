"""统计模块：条件抽取（LLM）/ 程序化筛选聚合（代码）/ 组答（确定性渲染 + sum LLM）。"""
import pytest

from app.core import stats


class FakeClient:
    """伪造 OpenAI 客户端：create 返回固定 content（或抛异常），记录调用。"""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = type("M", (), {"content": self.content})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def test_extract_filter_valid_json():
    client = FakeClient(content='{"table_id": "t1", "column": "学分", "op": "<", "value": "1", "aggregate": "none"}')
    f = stats.extract_filter(client, "m", "学分少于1的课程", [stats.TableCandidate("t1", ["课程", "学分"], [["体育", "1.0"]])])
    assert f == stats.Filter("t1", "学分", "<", "1", "none")


def test_extract_filter_json_in_prose_and_invalid():
    client = FakeClient(content='好的：{"table_id": "t1", "column": "学分", "op": "<", "value": "1", "aggregate": "count"} 完毕')
    assert stats.extract_filter(client, "m", "q", [stats.TableCandidate("t1", ["课程", "学分"], [])]).aggregate == "count"
    bad = FakeClient(content='{"table_id": "t1", "column": "不存在的列", "op": "<", "value": "1", "aggregate": "none"}')
    assert stats.extract_filter(bad, "m", "q", [stats.TableCandidate("t1", ["课程", "学分"], [])]) is None
    bad2 = FakeClient(content='{"table_id": "t1", "column": "学分", "op": "like", "value": "1", "aggregate": "none"}')
    assert stats.extract_filter(bad2, "m", "q", [stats.TableCandidate("t1", ["课程", "学分"], [])]) is None
    bad3 = FakeClient(content='{"table_id": "t1", "column": "学分", "op": "<", "value": "1", "aggregate": "sum", "sum_column": "不存在的列"}')
    assert stats.extract_filter(bad3, "m", "q", [stats.TableCandidate("t1", ["课程", "学分"], [])]) is None


def test_extract_filter_llm_error_and_non_json_return_none():
    assert stats.extract_filter(FakeClient(error=RuntimeError("down")), "m", "q", [stats.TableCandidate("t1", ["a"], [])]) is None
    assert stats.extract_filter(FakeClient(content="没有条件"), "m", "q", [stats.TableCandidate("t1", ["a"], [])]) is None


def test_extract_filter_whitespace_insensitive_header_match():
    # I1：docling 空格表头（「课 程 名 称」）与 LLM 输出「课程名称」空白无关匹配；回填存储表头
    tables = [stats.TableCandidate("t1", ["课 程 名 称", "学 分"], [["高数A", "3.0"]])]
    client = FakeClient(
        content='{"table_id": "t1", "column": "课程名称", "op": "contains", "value": "高数", "aggregate": "sum", "sum_column": "学分"}'
    )
    f = stats.extract_filter(client, "m", "高数课的学分合计", tables)
    assert f.column == "课 程 名 称" and f.sum_column == "学 分"
    assert stats.apply_filter(tables[0].headers, tables[0].sample_rows, f).aggregate_value == 3.0


def test_extract_filter_numeric_op_requires_parseable_value():
    # I2（终审）：数值 op + 比较值不可解析（如「小于1分」）→ 条件无法执行，拒绝而非放行成空结果；
    # 字符串 op 的比较值不受限（contains 子串匹配非数值是合法用法）
    tables = [stats.TableCandidate("t1", ["课程", "学分"], [["体育", "1.0"]])]
    for op in ("<", "<=", ">", ">="):
        client = FakeClient(
            content=f'{{"table_id": "t1", "column": "学分", "op": "{op}", "value": "小于1分", "aggregate": "none"}}'
        )
        assert stats.extract_filter(client, "m", "学分小于1分的课", tables) is None
    ok = FakeClient(
        content='{"table_id": "t1", "column": "课程", "op": "contains", "value": "小于1分", "aggregate": "none"}'
    )
    assert stats.extract_filter(ok, "m", "q", tables) is not None


def test_extract_filter_count_all_with_null_condition():
    # I4：「有几门课」→ 无筛选条件，统计全部行数
    client = FakeClient(content='{"table_id": "t1", "column": null, "op": null, "value": null, "aggregate": "count"}')
    f = stats.extract_filter(client, "m", "一共有几门课", [stats.TableCandidate("t1", ["课程", "学分"], [])])
    assert f == stats.Filter("t1", "", "", "", "count")


def test_extract_filter_no_condition_sum_with_sum_column():
    # I5：「总学分是多少」→ 无筛选条件求和：column/op/value 全空，sum_column 必填且空白无关回填存储表头
    tables = [stats.TableCandidate("t1", ["课 程", "学 分"], [["体育", "0.5"]])]
    client = FakeClient(
        content='{"table_id": "t1", "column": null, "op": null, "value": null, "aggregate": "sum", "sum_column": "学分"}'
    )
    f = stats.extract_filter(client, "m", "总学分是多少", tables)
    assert f == stats.Filter("t1", "", "", "", "sum", "学 分")
    rows = [["体育", "0.5"], ["高数", "3.0"], ["就业指导", "免修"]]
    out = stats.apply_filter(tables[0].headers, rows, f)
    assert out.rows == rows  # 不过滤：对全体行求和
    assert out.aggregate_value == 3.5 and out.skipped == 1  # 可解析值之和；「免修」跳过


def test_extract_filter_no_condition_sum_requires_valid_sum_column():
    tables = [stats.TableCandidate("t1", ["课程", "学分"], [])]
    missing = FakeClient(content='{"table_id": "t1", "column": null, "op": null, "value": null, "aggregate": "sum"}')
    assert stats.extract_filter(missing, "m", "总学分", tables) is None
    invalid = FakeClient(
        content='{"table_id": "t1", "column": null, "op": null, "value": null, "aggregate": "sum", "sum_column": "不存在"}'
    )
    assert stats.extract_filter(invalid, "m", "总学分", tables) is None
    # 混合形态（有 column 但 op/value 为空）维持拒绝
    mixed = FakeClient(content='{"table_id": "t1", "column": "学分", "op": null, "value": null, "aggregate": "sum"}')
    assert stats.extract_filter(mixed, "m", "总学分", tables) is None


def test_apply_filter_numeric_and_non_numeric():
    headers = ["课程", "学分"]
    rows = [["a", "0.5"], ["b", "1.0"], ["c", "免修"]]
    out = stats.apply_filter(headers, rows, stats.Filter("t", "学分", "<", "1"))
    assert out.rows == [["a", "0.5"]]  # 非数字单元格不匹配数值比较


def test_apply_filter_contains_and_eq():
    headers = ["课程", "性质"]
    rows = [["高数A1", "学基必修"], ["英语", "通识必修"]]
    assert stats.apply_filter(headers, rows, stats.Filter("t", "课程", "contains", "高数")).rows == [["高数A1", "学基必修"]]
    assert stats.apply_filter(headers, rows, stats.Filter("t", "性质", "==", "通识必修")).rows == [["英语", "通识必修"]]


def test_apply_filter_unknown_op_raises():
    # M1：未知操作符显式报错（校验白名单外不可达；调用方 try 兜底降级）
    with pytest.raises(ValueError, match="未知操作符"):
        stats.apply_filter(["课程"], [["a"]], stats.Filter("t", "课程", "like", "a"))


def test_apply_filter_count_all_with_empty_column():
    # I4：column 为空 = 无筛选条件，统计全部行
    headers = ["课程", "学分"]
    rows = [["体育", "0.5"], ["高数", "3.0"], ["就业指导", "0.5"]]
    out = stats.apply_filter(headers, rows, stats.Filter("t", "", "", "", aggregate="count"))
    assert out.rows == rows and out.aggregate_value == len(rows) == 3


def test_apply_filter_count_and_sum_with_skipped():
    headers = ["课程", "学分"]
    rows = [["实验A", "0.5"], ["实验B", "1.0"], ["实验C", "免修"]]
    # 筛选先行（字符串 contains 命中全部 3 行），聚合在结果集上执行
    cnt = stats.apply_filter(headers, rows, stats.Filter("t", "课程", "contains", "实验", aggregate="count"))
    assert cnt.aggregate_value == 3
    total = stats.apply_filter(
        headers, rows, stats.Filter("t", "课程", "contains", "实验", aggregate="sum", sum_column="学分")
    )
    assert total.aggregate_value == 1.5 and total.skipped == 1  # 「免修」不可解析 → 跳过并计数


def test_apply_filter_sum_on_numeric_filter_column():
    # M8a：数值过滤列 + sum 同列（未显式给 sum_column 时默认用 column）
    headers = ["课程", "学分"]
    rows = [["体育", "0.5"], ["高数", "1.0"], ["就业指导", "0.5"]]
    out = stats.apply_filter(headers, rows, stats.Filter("t", "学分", "<", "1", aggregate="sum"))
    assert out.rows == [["体育", "0.5"], ["就业指导", "0.5"]]
    assert out.aggregate_value == 1.0 and out.skipped == 0


def test_apply_filter_sum_no_fallback_when_target_not_numeric():
    # I2：无 sum_column 时只用筛选列求和；不做「首个可数值列」的回退猜测
    headers = ["课程", "学分"]
    rows = [["实验A", "0.5"], ["实验B", "1.0"]]
    out = stats.apply_filter(headers, rows, stats.Filter("t", "课程", "contains", "实验", aggregate="sum"))
    assert out.aggregate_value is None and out.skipped == 2
    explicit = stats.apply_filter(
        headers, rows, stats.Filter("t", "课程", "contains", "实验", aggregate="sum", sum_column="学分")
    )
    assert explicit.aggregate_value == 1.5 and explicit.skipped == 0


def test_compose_answer_empty_rows_returns_fixed_message_without_llm():
    client = FakeClient(content="不该被调用")
    out = stats.FilterOutcome(rows=[], aggregate_value=0, aggregate="count")
    assert stats.compose_answer(client, "m", "有几门课", ["课程"], out) == "没有符合条件的记录。"
    assert client.calls == []


def test_compose_answer_count_is_deterministic():
    client = FakeClient(content="不该被调用")
    out = stats.FilterOutcome(rows=[["体育", "0.5"], ["高数", "3.0"]], aggregate_value=2, aggregate="count")
    assert stats.compose_answer(client, "m", "有几门课", ["课程", "学分"], out) == "符合条件的有 2 条。"
    assert client.calls == []


def test_compose_answer_none_renders_numbered_rows_without_llm():
    # C1：none 确定性渲染——行数与结果行一一对应；行内容 = 非空单元格以「 | 」连接
    client = FakeClient(content="不该被调用")
    rows = [["电路分析基础实验", "学基必修", "0.5", "良"], ["体育", "", "1.0", ""]]
    out = stats.FilterOutcome(rows=rows, aggregate_value=None)
    answer = stats.compose_answer(client, "m", "有哪些课", ["课程", "性质", "学分", "成绩"], out)
    assert answer == (
        "筛选结果（2 条）：\n"
        "1. 电路分析基础实验 | 学基必修 | 0.5 | 良\n"
        "2. 体育 | 1.0"
    )
    assert client.calls == []


def test_compose_answer_sum_branch_prompt_carries_constraints():
    rows = [[f"课程{i}", "0.5"] for i in range(35)]
    client = FakeClient(content="共 35 门，总学分 17.5")
    out = stats.FilterOutcome(rows=rows, aggregate_value=17.5, skipped=2, aggregate="sum", sum_column="学分")
    answer = stats.compose_answer(client, "m", "总学分是多少", ["课程", "学分"], out)
    assert answer == "共 35 门，总学分 17.5"
    system, user = (m["content"] for m in client.calls[0]["messages"])
    assert "不得新增" in system and "不要编造" in system and "必须原样采用" in system
    assert "求和列：学分" in user and "聚合结果：17.5" in user
    assert "另有 2 个单元格无法解析为数值，未计入求和" in user
    assert "课程0 | 0.5" in user and "课程29 | 0.5" in user and "课程30" not in user
    assert "…共 35 行" in user  # 超 30 行只给前 30 行 + 总行数


def test_compose_answer_sum_without_value_marks_unavailable():
    # 求和列无可解析数值 → 不能编造 0，提示词标注无法计算
    client = FakeClient(content="无法计算总学分")
    out = stats.FilterOutcome(
        rows=[["实验A", "免修"]], aggregate_value=None, skipped=1, aggregate="sum", sum_column="学分"
    )
    assert stats.compose_answer(client, "m", "总学分", ["课程", "学分"], out) == "无法计算总学分"
    _, user = (m["content"] for m in client.calls[0]["messages"])
    assert "聚合结果：无法计算" in user and "另有 1 个单元格" in user


def test_compose_answer_sum_empty_content_returns_empty_string():
    # M11：sum 分支 content 为 None 时返回空串（调用方视为失败降级）
    client = FakeClient(content=None)
    out = stats.FilterOutcome(rows=[["a", "0.5"]], aggregate_value=0.5, aggregate="sum", sum_column="学分")
    assert stats.compose_answer(client, "m", "q", ["课程", "学分"], out) == ""


def test_apply_filter_counts_unparseable_rows_for_numeric_op():
    headers = ["课程", "学分"]
    rows = [["a", "0.5"], ["b", "通识必修 2.0"], ["c", "1.0"]]
    out = stats.apply_filter(headers, rows, stats.Filter("t", "学分", "<", "1"))
    assert out.rows == [["a", "0.5"]] and out.filter_skipped == 1


def test_apply_filter_string_op_has_no_filter_skipped():
    out = stats.apply_filter(["课程"], [["a"]], stats.Filter("t", "课程", "contains", "a"))
    assert out.filter_skipped == 0


def test_compose_count_and_none_disclose_filter_skipped():
    client = FakeClient(content="不该被调用")
    cnt = stats.FilterOutcome(rows=[["a", "1"]], aggregate_value=1, aggregate="count",
                              filter_skipped=2)
    text = stats.compose_answer(client, "m", "q", ["课程", "学分"], cnt)
    assert "2" in text and "无法解析" in text and client.calls == []
    listed = stats.FilterOutcome(rows=[["体育", "1.0"]], aggregate_value=None, filter_skipped=3)
    assert "3" in stats.compose_answer(client, "m", "q", ["课程", "学分"], listed)


def test_compose_empty_result_discloses_filter_skipped():
    empty = stats.FilterOutcome(rows=[], aggregate_value=0, aggregate="count", filter_skipped=4)
    text = stats.compose_answer(client := FakeClient(content="x"), "m", "q", ["课程"], empty)
    assert text.startswith("没有符合条件的记录") and "4" in text


def test_compose_answer_sum_branch_prompt_carries_filter_skipped():
    # filter_skipped 披露同样进 sum 组答提示词；既有 skipped 句保留
    client = FakeClient(content="共 2 门")
    out = stats.FilterOutcome(
        rows=[["a", "0.5"], ["b", "1.0"]], aggregate_value=1.5, skipped=1,
        aggregate="sum", sum_column="学分", filter_skipped=3,
    )
    stats.compose_answer(client, "m", "q", ["课程", "学分"], out)
    _, user = (m["content"] for m in client.calls[0]["messages"])
    assert "另有 3 行该列无法解析为数值，未纳入筛选。" in user
    assert "另有 1 个单元格无法解析为数值，未计入求和" in user


def test_compose_disclosure_text_pinned_exactly():
    # 确定性分支（empty/count）：披露文案逐字钉死（filter_skipped=0 时无披露由既有精确断言覆盖）
    client = FakeClient(content="不该被调用")
    cnt = stats.FilterOutcome(rows=[["a", "1"]], aggregate_value=1, aggregate="count", filter_skipped=2)
    assert stats.compose_answer(client, "m", "q", ["课程", "学分"], cnt) == (
        "符合条件的有 1 条。另有 2 行该列无法解析为数值，未纳入筛选。"
    )
    empty = stats.FilterOutcome(rows=[], aggregate_value=0, aggregate="count", filter_skipped=4)
    assert stats.compose_answer(client, "m", "q", ["课程"], empty) == (
        "没有符合条件的记录。另有 4 行该列无法解析为数值，未纳入筛选。"
    )
    assert client.calls == []
