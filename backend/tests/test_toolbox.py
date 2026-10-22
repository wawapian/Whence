"""工具箱 v1（toolbox）：六件工具单元测试 + 派发容错 + 隐私闸。"""
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import toolbox
from app.memory import documents, store
from app.models.db import Image, init_db


def _ctx(tmp_path, **kw):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    s = Session(engine)
    return s, toolbox.ToolContext(session=s, settings=Settings(data_dir=tmp_path, **kw))


def test_read_document_gates_when_not_allowed(tmp_path):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(
        s, kind="word", status="done", title="甲",
        parsed_text="联系人手机号 13800000001", allow_cloud=0,
    )
    s.commit()
    out = toolbox.read_document(ctx, doc.id)
    assert "13800000001" not in out
    assert "[[SECRET]]" in out
    assert doc.id in ctx.used_doc_ids


def test_read_document_verbatim_when_allowed(tmp_path):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(
        s, kind="word", status="done", title="甲",
        parsed_text="联系人手机号 13800000001", allow_cloud=1,
    )
    s.commit()
    assert "13800000001" in toolbox.read_document(ctx, doc.id)


def test_read_document_keyword_filter_and_missing(tmp_path):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(
        s, kind="word", status="done", title="甲",
        parsed_text="第一行 学习率\n第二行 准确率", allow_cloud=1,
    )
    s.commit()
    out = toolbox.read_document(ctx, doc.id, keyword="学习率")
    assert "学习率" in out and "准确率" not in out
    assert "没有含" in toolbox.read_document(ctx, doc.id, keyword="不存在词")
    assert "不存在或已删除" in toolbox.read_document(ctx, "no-such-id")


def test_measure_text_exact_and_chunk_fallback(tmp_path):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(
        s, kind="word", status="done", title="论文", parsed_text="一二三四五", allow_cloud=1
    )
    doc2 = documents.insert_document(s, kind="word", status="done", title="无正文", allow_cloud=1)
    s.flush()
    documents.insert_chunk(s, document_id=doc2.id, seq=0, text="abc", parent_text="abc")
    documents.insert_chunk(s, document_id=doc2.id, seq=1, text="de", parent_text="de")
    s.commit()

    assert "字数：5 个字符" in toolbox.measure_text(ctx, doc.id)
    assert "字数：5 个字符" in toolbox.measure_text(ctx, doc2.id, metric="chars")  # 分块回落
    assert "只支持 chars" in toolbox.measure_text(ctx, doc.id, metric="tokens")


def test_table_stats_filter_count_and_suspect(tmp_path):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(s, kind="word", status="done", title="成绩总表", allow_cloud=1)
    s.flush()
    table = documents.insert_table(
        s, document_id=doc.id, suspect=True,
        headers=json.dumps(["课程", "学分"], ensure_ascii=False),
        rows=json.dumps([["体育", "0.5"], ["高数", "3.0"], ["就业指导", "0.5"]], ensure_ascii=False),
    )
    s.commit()

    out = toolbox.table_stats(ctx, table.id, column="学分", op="<", value="1", aggregate="none")
    assert "体育" in out and "就业指导" in out and "高数" not in out
    assert "解析可能不完整" in out  # suspect 如实提示
    assert table.id in ctx.used_table_ids

    counted = toolbox.table_stats(ctx, table.id, aggregate="count")
    assert "符合条件的有 3 条。" in counted

    summed = toolbox.table_stats(ctx, table.id, aggregate="sum", sum_column="学分")
    assert "4" in summed  # 0.5 + 3.0 + 0.5 = 4（代码精确求和）


def test_table_stats_bad_args_and_missing(tmp_path):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(s, kind="word", status="done", title="成绩总表", allow_cloud=1)
    s.flush()
    table = documents.insert_table(
        s, document_id=doc.id,
        headers=json.dumps(["课程", "学分"], ensure_ascii=False),
        rows=json.dumps([["体育", "0.5"]], ensure_ascii=False),
    )
    s.commit()

    bad_col = toolbox.table_stats(ctx, table.id, column="不存在的列", op="<", value="1", aggregate="none")
    assert "参数不合法" in bad_col and "课程" in bad_col  # 附上表头供模型重试
    bad_val = toolbox.table_stats(ctx, table.id, column="学分", op="<", value="小于1分", aggregate="none")
    assert "参数不合法" in bad_val  # 数值 op 比较值不可解析 → 不执行（沿用 stats 校验）
    assert "表格不存在" in toolbox.table_stats(ctx, "no-such-table", aggregate="count")


def _seed_grade_table(s, suspect=False):
    doc = documents.insert_document(s, kind="word", status="done", title="成绩总表", allow_cloud=1)
    s.flush()
    table = documents.insert_table(
        s, document_id=doc.id, suspect=suspect,
        headers=json.dumps(["姓名", "语文", "数学", "学历", "月薪"], ensure_ascii=False),
        rows=json.dumps([
            ["甲", "85", "90", "本科", "12000"],
            ["乙", "78", "88", "本科", "9000"],
            ["丙", "92", "76", "硕士", "15000"],
            ["丁", "80", "82", "硕士", "18000"],
        ], ensure_ascii=False),
    )
    s.commit()
    return table


def test_table_stats_multicond(tmp_path):
    s, ctx = _ctx(tmp_path)
    table = _seed_grade_table(s)
    out = toolbox.table_stats(ctx, table.id, filters=[
        {"column": "语文", "op": ">=", "value": "80"},
        {"column": "数学", "op": ">=", "value": "80"},
    ], aggregate="count")
    assert "符合条件的有 2 条。" in out


def test_table_stats_group_by_count(tmp_path):
    s, ctx = _ctx(tmp_path)
    table = _seed_grade_table(s)
    out = toolbox.table_stats(ctx, table.id, group_by="学历")
    assert "分组统计（按「学历」，2 组）：" in out
    assert "- 本科：2 条" in out and "- 硕士：2 条" in out


def test_table_stats_avg(tmp_path):
    s, ctx = _ctx(tmp_path)
    table = _seed_grade_table(s)
    out = toolbox.table_stats(ctx, table.id, aggregate="avg", sum_column="月薪")
    assert "均值结果：13500（列「月薪」，4 行）" in out  # (12000+9000+15000+18000)/4


def test_table_stats_order_by_limit(tmp_path):
    s, ctx = _ctx(tmp_path)
    table = _seed_grade_table(s)
    out = toolbox.table_stats(ctx, table.id, aggregate="none", order_by="月薪", desc=True, limit=2)
    assert "筛选结果（4 条）：" in out and "…共 4 行" in out
    assert "丁" in out and "丙" in out and "乙" not in out


def test_table_stats_invalid_shows_usage(tmp_path):
    s, ctx = _ctx(tmp_path)
    table = _seed_grade_table(s)
    out = toolbox.table_stats(ctx, table.id,
                              filters=[{"column": "不存在", "op": "==", "value": "x"}],
                              aggregate="count")
    assert "参数不合法" in out and "filters" in out and "group_by" in out


def test_view_image_local_ocr_when_not_allowed(tmp_path, monkeypatch):
    from app.core import cloud

    def _never(*a, **k):
        raise AssertionError("未勾选可理解的图片不得调用 VLM")

    monkeypatch.setattr(cloud, "describe_image", _never)
    s, ctx = _ctx(tmp_path, vlm_model="qwen3-vl-plus")
    doc = documents.insert_document(s, kind="word", status="done", title="甲", allow_cloud=0)
    s.flush()
    img = Image(id="img1", document_id=doc.id, ocr_text="图里的文字：会议纪要")
    s.add(img)
    s.commit()

    out = toolbox.view_image(ctx, "img1", "图里有什么")
    assert "本地 OCR" in out and "会议纪要" in out
    assert img.id in ctx.used_image_ids and doc.id in ctx.used_doc_ids


def test_view_image_vlm_when_allowed(tmp_path, monkeypatch):
    from app.core import cloud

    monkeypatch.setattr(cloud, "describe_image", lambda c, m, blob, q: "一只猫")
    s, ctx = _ctx(tmp_path, vlm_model="qwen3-vl-plus")
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    (img_dir / "i.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    doc = documents.insert_document(s, kind="word", status="done", title="甲", allow_cloud=1)
    s.flush()
    s.add(Image(id="img1", document_id=doc.id, file_path="images/i.png"))
    s.commit()

    out = toolbox.view_image(ctx, "img1", "图里有什么")
    assert "一只猫" in out


def test_view_image_no_content_when_local_empty(tmp_path, monkeypatch):
    s, ctx = _ctx(tmp_path)
    doc = documents.insert_document(s, kind="word", status="done", title="甲", allow_cloud=0)
    s.flush()
    s.add(Image(id="img1", document_id=doc.id))
    s.commit()
    out = toolbox.view_image(ctx, "img1", "图里有什么")
    assert "没有可用的内容" in out


def test_list_files_keyword(tmp_path):
    s, ctx = _ctx(tmp_path)
    documents.insert_document(s, kind="word", status="done", title="模式识别课程设计")
    documents.insert_document(s, kind="word", status="done", title="实验报告")
    documents.insert_document(s, kind="word", status="done", title="已删", deleted_at=store.utcnow())
    s.commit()

    out = toolbox.list_files(ctx, keyword="模式")
    assert "模式识别课程设计" in out and "实验报告" not in out and "已删" not in out
    assert "没有匹配的文件" in toolbox.list_files(ctx, keyword="不存在")


def _add_fact(s, *, amount=None, currency=None, summary="午饭", raw_content=None,
              type_="consumption"):
    from datetime import datetime, timezone

    from app.models.db import Fact

    raw = store.insert_raw(s, kind="text", content=raw_content or summary)
    s.flush()
    fact = Fact(id=store.new_id(), type=type_, summary=summary,
                fields={"amount": amount, "currency": currency},
                valid_from=datetime.now(timezone.utc), raw_id=raw.id)
    s.add(fact)
    s.flush()
    return fact


def test_memory_stats_returns_rendered_totals(tmp_path):
    s, ctx = _ctx(tmp_path)
    _add_fact(s, amount=32.0, currency="CNY", summary="午饭")
    _add_fact(s, amount=28.0, currency="CNY", summary="打车")
    s.commit()
    out = toolbox.memory_stats(ctx, period="this_month")
    assert "消费合计：CNY 60.00（2 笔）" in out


def test_memory_stats_marks_used_facts(tmp_path):
    s, ctx = _ctx(tmp_path)
    f1 = _add_fact(s, amount=32.0, currency="CNY")
    f2 = _add_fact(s, amount=28.0, currency="CNY")
    s.commit()
    toolbox.memory_stats(ctx, period="this_month")
    assert ctx.used_fact_ids == [f2.id, f1.id]  # 时间倒序（后记的在前）


def test_memory_stats_invalid_period_readable(tmp_path):
    s, ctx = _ctx(tmp_path)
    out = toolbox.memory_stats(ctx, period="last_year")
    assert out.startswith("（统计参数不合法")
    assert "this_month" in out


def test_memory_stats_provenance_capped_to_displayed_rows(tmp_path):
    """终审：溯源只并入**渲染明细里出现过的** fact（上限=STATS_ROWS_MAX）——period=all 大库不爆 sources。"""
    from app.core.memory_stats import STATS_ROWS_MAX

    s, ctx = _ctx(tmp_path)
    for i in range(STATS_ROWS_MAX + 5):
        _add_fact(s, amount=1.0, currency="CNY", summary=f"第{i}笔")
    s.commit()
    out = toolbox.memory_stats(ctx, period="this_month")
    assert "…共 25 条" in out
    assert len(ctx.used_fact_ids) == STATS_ROWS_MAX


def test_memory_stats_masks_secret_numbers(tmp_path):
    s, ctx = _ctx(tmp_path)
    _add_fact(s, amount=32.0, currency="CNY", summary="给家人报备电话 13800000001")
    s.commit()
    out = toolbox.memory_stats(ctx, period="this_month")
    assert "13800000001" not in out
    assert "[[SECRET]]" in out


def test_run_tool_dispatch_and_fault_tolerance(tmp_path):
    s, ctx = _ctx(tmp_path)
    assert "未知工具" in toolbox.run_tool(ctx, "nope", {})
    assert "参数不合法" in toolbox.run_tool(ctx, "read_document", {"wrong_key": "x"})
    assert "表格不存在" in toolbox.run_tool(
        ctx, "table_stats", {"table_id": "missing", "aggregate": "count"}
    )


# --- 参数守卫（spec 2026-10-16-stats-param-guard）----------------------------
# 追加在本文件而非新建：复用上面的 _ctx/_add_fact/_seed_* 夹具，守卫本就是 toolbox 的行为。


def _seed_company_table(s):
    doc = documents.insert_document(s, kind="excel", status="done",
                                    title="jobs_006.xlsx", allow_cloud=1)
    s.flush()
    table = documents.insert_table(
        s, document_id=doc.id, suspect=False,
        headers=json.dumps(["公司", "岗位", "学历要求"], ensure_ascii=False),
        rows=json.dumps([
            ["样例科技有限公司", "调度员", "大专"],
            ["样例信息有限公司", "新媒体运营", "不限"],
        ], ensure_ascii=False),
    )
    s.commit()
    return table


def test_memory_stats_rejects_mismatched_period(tmp_path):
    """问题说「上个月」而模型填 this_month → 打回，且**不统计、不留痕**。"""
    s, ctx = _ctx(tmp_path)
    ctx.question = "上个月我一共花了多少钱"
    out = toolbox.memory_stats(ctx, period="this_month")
    assert "请用 last_month 重试" in out
    assert ctx.rejections["time"] == 1
    assert ctx.stats_calls == []


def test_memory_stats_forces_window_on_second_mismatch(tmp_path, monkeypatch):
    """打回一次后仍不符 → 强制用代码判定的窗口，并在口径里注明。"""
    s, ctx = _ctx(tmp_path)
    ctx.question = "上个月我一共花了多少钱"
    seen = {}
    real_collect = toolbox.memory_stats_core.collect

    def spy(session, **kw):
        seen.update(kw)
        return real_collect(session, **kw)

    monkeypatch.setattr(toolbox.memory_stats_core, "collect", spy)

    toolbox.memory_stats(ctx, period="this_month")            # 第一次：打回
    out = toolbox.memory_stats(ctx, period="this_month")      # 第二次：强制

    assert seen["period"] == "last_month"
    assert "已按此统计" in ctx.stats_calls[-1]["label"]
    assert "没有符合条件" in out                                # 空区间：既有渲染（口径由终答落地补）


def test_memory_stats_untouched_when_question_has_no_time_phrase(tmp_path):
    """问题里没有词表内的说法 → 零打扰（不校验、不计数）。"""
    s, ctx = _ctx(tmp_path)
    _add_fact(s, amount=10.0, currency="CNY")
    s.commit()
    ctx.question = "前一阵子花了多少钱"
    out = toolbox.memory_stats(ctx, period="this_month")
    assert "统计口径" in out
    assert ctx.rejections == {}
    assert ctx.stats_calls and ctx.stats_calls[0]["label"].startswith("按记录时间")


def test_table_stats_rejects_rewritten_entity(tmp_path):
    """问题里带引号的实体被改写（样例科技 ← 样例信息）→ 打回，且不执行统计。"""
    s, ctx = _ctx(tmp_path)
    table = _seed_company_table(s)
    ctx.question = "「样例信息有限公司」的岗位按学历要求分组"
    out = toolbox.table_stats(ctx, table_id=table.id, column="公司", op="contains",
                              value="样例科技有限公司", aggregate="count")
    assert "请严格原样使用" in out
    assert ctx.rejections["entity"] == 1
    assert ctx.stats_calls == []


def test_table_stats_passes_substring_entity_and_records_caliber(tmp_path):
    """合法简称 → 放行；成功调用留痕（供终答口径落地）。"""
    s, ctx = _ctx(tmp_path)
    table = _seed_company_table(s)
    ctx.question = "「样例信息有限公司」的岗位按学历要求分组"
    out = toolbox.table_stats(ctx, table_id=table.id, column="公司", op="contains",
                              value="样例信息", aggregate="count")
    assert ctx.rejections == {}
    assert "1" in out
    # 探针取**整条口径**（不是筛选值）：筛选值太短，容易在答案里偶然命中而**漏判**
    assert ctx.stats_calls and ctx.stats_calls[0]["probe"] == "按「公司 contains 样例信息」筛选"


def test_table_stats_forces_entity_on_second_mismatch(tmp_path):
    """第二次仍用错名 → **强制替换**成问题里的实体名（spec §3.3 修订：不再放行）。

    修订依据（2026-10-16 评测实测）：软兜底让 q-0036 的错答落地了——模型被明确打回后
    仍坚持用「样例科技有限公司」，还在答案里把「样例信息有限公司」说成"另一家公司"。
    """
    s, ctx = _ctx(tmp_path)
    table = _seed_company_table(s)
    ctx.question = "「样例信息有限公司」的岗位按学历要求分组"

    first = toolbox.table_stats(ctx, table_id=table.id, column="公司", op="contains",
                                value="样例科技有限公司", aggregate="count")
    assert "请严格原样使用" in first          # 第一次：打回

    toolbox.table_stats(ctx, table_id=table.id, column="公司", op="contains",
                        value="样例科技有限公司", aggregate="count")

    assert ctx.stats_calls[-1]["probe"] == "按「公司 contains 样例信息有限公司」筛选"


def test_table_stats_tolerates_numeric_filter_value(tmp_path):
    """数值筛选很常见（「90 分以上」）——不得因值是**数字**而不是字符串而炸掉整次工具调用。"""
    s, ctx = _ctx(tmp_path)
    table = _seed_grade_table(s)
    ctx.question = "「成绩总表」里数学 90 分以上的有几个？"

    out = toolbox.table_stats(ctx, table_id=table.id, column="数学", op=">=", value=90,
                              aggregate="count")

    assert "参数不合法" not in out
    assert "1 条" in out          # 甲 90 分


def test_table_stats_output_includes_filter_line(tmp_path):
    """工具输出带筛选条件行——模型据此在答案里复述口径（口径可见）。"""
    s, ctx = _ctx(tmp_path)
    table = _seed_company_table(s)
    out = toolbox.table_stats(ctx, table_id=table.id, column="公司", op="contains",
                              value="样例信息", aggregate="count")
    assert "筛选条件：公司 contains 样例信息" in out


def test_table_stats_output_has_no_filter_line_without_filter(tmp_path):
    """无筛选 → 输出不多这一行（零影响）。"""
    s, ctx = _ctx(tmp_path)
    table = _seed_company_table(s)
    out = toolbox.table_stats(ctx, table_id=table.id, aggregate="count")
    assert "筛选条件" not in out
