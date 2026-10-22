"""问答层主循环（qa.py）：材料两级检索 / 五工具 / 隐私闸 / 轮数上限 / 回退开关。"""
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import answer, llm, qa, retrieval
from app.memory import bitemporal, chunks, documents, fts, materials, store, vector
from app.models.db import Image, init_db
from app.models.schemas import ExtractedFact

MODEL = "BAAI/bge-small-zh-v1.5"


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=512)
    return Session(engine)


def _settings(tmp_path, **kw):
    return Settings(_env_file=None, data_dir=tmp_path, embedding_model=MODEL,
                    rerank_engine="off", **kw)


def _completion(content=None, tool_calls=None):
    message = type("Msg", (), {"content": content, "tool_calls": tool_calls})()
    return type("Resp", (), {"choices": [type("Choice", (), {"message": message})()]})()


def _tool_call(id_, name, args):
    fn = type("Fn", (), {"name": name, "arguments": json.dumps(args)})()
    return type("TC", (), {"id": id_, "function": fn})()


class ScriptedClient:
    """按脚本返回响应并记录每次调用；脚本元素可为响应对象或要抛出的异常。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                item = outer.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _seed_doc(s, title="实验报告", text="第三组学习率 0.001", allow_cloud=1, parsed_text=None):
    doc = documents.insert_document(
        s, kind="word", status="done", title=title,
        allow_cloud=allow_cloud, parsed_text=parsed_text if parsed_text is not None else text,
    )
    s.flush()
    c = documents.insert_chunk(s, document_id=doc.id, seq=0, text=text, parent_text=text)
    s.flush()
    chunks.index_chunk(s, c.id, [0.1] * 512, text)
    materials.upsert_for_document(s, doc, model=MODEL)
    s.commit()
    return doc, c


def _patch_embed(monkeypatch):
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)


def test_system_prompt_defines_inclusive_yishang():
    """「N 以上」的包含口径写进系统提示词。

    评测实测（会话 26）：同一句「数学 90 分以上」模型时而用 >=（对）、时而用 >（差 1 人判错）
    ——提示词里没有口径定义，模型自己猜。
    """
    assert "以上」含 N" in qa._SYSTEM_PROMPT


def test_system_prompt_requires_explanation_before_marker():
    """拒答要先给线索：不足以回答时必须先说明材料里有什么/缺什么，再另起一行输出标记。

    真实现场（会话 26 v2 复跑）：旧提示词写「只输出：[信息不足]」，模型照做 → 用户只收到
    兜底文案「根据已有内容无法回答该问题。」——库里有「存了体检报告」的记录，却什么都不给。
    """
    assert "先说明材料里有什么" in qa._SYSTEM_PROMPT
    assert "只输出：" not in qa._SYSTEM_PROMPT


def test_no_materials_returns_fixed_answer(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    resp = qa.run(s, "不存在的记忆", _settings(tmp_path))
    assert resp.answer == "没有找到相关记录"
    assert resp.sources == [] and resp.documents == []


def test_run_answers_without_tools(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    doc, _ = _seed_doc(s)
    client = ScriptedClient([_completion(content="学习率是 0.001")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    resp = qa.run(s, "学习率多少", _settings(tmp_path))
    assert resp.answer == "学习率是 0.001"
    assert len(client.calls) == 1 and "tools" in client.calls[0]
    assert [d.document_id for d in resp.documents] == [doc.id]  # 溯源 = 命中片段


def test_char_count_question_uses_measure_text_tool(tmp_path, monkeypatch):
    """「统计字数」误判回归的根治形态：问题交模型，精确计数由 measure_text 工具出。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    doc, _ = _seed_doc(s, title="论文", text="正文一二三四五", parsed_text="一二三四五")
    client = ScriptedClient([
        _completion(tool_calls=[_tool_call(
            "t1", "measure_text", {"doc_id": doc.id, "metric": "chars"}
        )]),
        _completion(content="这篇论文共有 5 个字。"),
    ])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    resp = qa.run(s, "统计一下这篇论文有多少个字", _settings(tmp_path))
    assert resp.answer == "这篇论文共有 5 个字。"
    second_messages = client.calls[1]["messages"]
    tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
    assert tool_msgs and "字数：5 个字符" in tool_msgs[0]["content"]


def test_rounds_exhausted_forces_final_answer(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    same_call = [_tool_call("t1", "list_files", {})]
    client = ScriptedClient([
        _completion(tool_calls=same_call),
        _completion(tool_calls=same_call),
        _completion(content="最终答案"),
    ])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    resp = qa.run(s, "随便问问", _settings(tmp_path, qa_max_rounds=2))
    assert resp.answer == "最终答案"
    assert len(client.calls) == 3
    assert client.calls[2].get("tool_choice") == "none"  # 轮数耗尽 → 禁用工具拿终答


def test_gate_masks_sensitive_doc_context(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s, text="联系人手机号 13800000001", allow_cloud=0)
    client = ScriptedClient([_completion(content="ok")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    qa.run(s, "联系人", _settings(tmp_path))
    user_content = client.calls[0]["messages"][1]["content"]
    assert "13800000001" not in user_content
    assert "[[SECRET]]" in user_content


def test_gate_allows_cloud_doc_verbatim(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s, text="联系人手机号 13800000001", allow_cloud=1)
    client = ScriptedClient([_completion(content="ok")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    qa.run(s, "联系人", _settings(tmp_path))
    user_content = client.calls[0]["messages"][1]["content"]
    assert "13800000001" in user_content


def test_record_raw_masked_in_context(tmp_path, monkeypatch):
    """记录侧一律过闸（记录无 allow_cloud 字段；敏感值只本地存的大原则）。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    raw = store.insert_raw(s, kind="text", content="我的手机号是 13800000001")
    s.flush()
    fact = store.insert_fact(
        s,
        ExtractedFact(type="contact", subject_key="手机号", summary="我的手机号", confidence=0.9),
        raw.id,
    )
    s.flush()
    fts.index_fact(s, fact.id, "我的手机号")
    vector.index_fact(s, fact.id, [0.1] * 512)
    materials.upsert_for_record(s, raw, title=fact.summary, model=MODEL)
    s.commit()

    client = ScriptedClient([_completion(content="ok")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    qa.run(s, "我的手机号", _settings(tmp_path))

    user_content = client.calls[0]["messages"][1]["content"]
    assert "13800000001" not in user_content
    assert "[[SECRET]]" in user_content


def test_table_stats_tool_flow(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    doc, _ = _seed_doc(s, title="成绩总表", text="课程学分")
    table = documents.insert_table(
        s, document_id=doc.id,
        headers=json.dumps(["课程", "学分"], ensure_ascii=False),
        rows=json.dumps([["体育", "0.5"], ["高数", "3.0"]], ensure_ascii=False),
    )
    s.commit()
    client = ScriptedClient([
        _completion(tool_calls=[_tool_call("t1", "table_stats", {
            "table_id": table.id, "column": "学分", "op": "<", "value": "1", "aggregate": "none",
        })]),
        _completion(content="只有体育课学分少于 1。"),
    ])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    resp = qa.run(s, "学分少于1的课有哪些", _settings(tmp_path))
    # 统计类回答由代码补口径尾行（spec 2026-10-16-stats-param-guard §3.4/G3）——故这里不再逐字相等
    assert resp.answer.startswith("只有体育课学分少于 1。")
    assert "统计口径：按「学分 < 1」筛选" in resp.answer
    first_user = client.calls[0]["messages"][1]["content"]
    assert f"表 {table.id}" in first_user  # 表格清单（id + 表头）进了上下文
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    assert "体育" in tool_msgs[0]["content"] and "高数" not in tool_msgs[0]["content"]
    assert [d.document_id for d in resp.documents].count(doc.id) >= 1  # 表格所在文档可溯源


def test_deleted_doc_not_visible(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    doc, _ = _seed_doc(s)
    doc.deleted_at = store.utcnow()
    s.commit()
    resp = qa.run(s, "学习率", _settings(tmp_path))
    assert resp.answer == "没有找到相关记录"


def test_details_filtered_to_hit_materials(tmp_path, monkeypatch):
    """第二级检索只回命中材料内的细节：两份文档同文本都被索引，只传 doc_a 时不得混入 doc_b。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    doc_a, chunk_a = _seed_doc(s, title="甲", text="学习率 0.001")
    doc_b, chunk_b = _seed_doc(s, title="乙", text="学习率 0.001")
    fact_ids, chunk_ids = retrieval.search_details_in_materials(
        s, ["学习率"], record_raw_ids=[], doc_ids=[doc_a.id], top_k=5, model=MODEL,
    )
    assert chunk_ids == [chunk_a.id]
    assert fact_ids == []


def _seed_state_record(s, content, summary, subject_key="身份证:位置"):
    """按真实整理链路落一条状态类记录：双时态 upsert + 索引 + 材料（同 subject_key 自动取代旧值）。"""
    raw = store.insert_raw(s, kind="text", content=content)
    s.flush()
    fact = bitemporal.upsert(
        s,
        ExtractedFact(type="item_location", subject_key=subject_key, summary=summary,
                      confidence=0.9),
        raw.id,
    )
    s.flush()
    fts.index_fact(s, fact.id, f"{summary} {content}")
    vector.index_fact(s, fact.id, [0.1] * 512)
    materials.upsert_for_record(s, raw, title=fact.summary, model=MODEL)
    s.commit()
    return raw, fact


def _section(user_content, header):
    """取某个「## 」二级段的文本（到下一段为止）。"""
    out, capturing = [], False
    for line in user_content.splitlines():
        if line.startswith("## "):
            capturing = line == header
            continue
        if capturing:
            out.append(line)
    return "\n".join(out)


def test_superseded_fact_not_in_qa_context(tmp_path, monkeypatch):
    """读侧双时态：已被取代的旧值不再进检索明细；材料清单里的旧记录行带「已被覆盖」标注。

    场景（复刻真库）：记 A「身份证放第一柜」→ 问 → 记 B「身份证放背包夹层」（同 subject_key，
    写侧 upsert 已把 A 置 valid_to、superseded_by 指向 B）→ 再问。A 的索引行仍在（账本留档、
    双时态不清理索引），修复前经检索原样进上下文 → 模型看到两条矛盾记录判「无法判断」。
    """
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    client = ScriptedClient([_completion(content="第一次回答"), _completion(content="第二次回答")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    _seed_state_record(s, "身份证我放在书房第一柜了", "身份证放在书房第一柜")
    qa.run(s, "身份证在哪", _settings(tmp_path))
    _, fact_b = _seed_state_record(s, "身份证改放背包夹层了", "身份证放在背包夹层")
    qa.run(s, "身份证在哪", _settings(tmp_path))

    ctx = client.calls[1]["messages"][1]["content"]
    details = _section(ctx, "## 相关内容（检索命中）")
    assert "- 记忆：" in details              # 明细段确实有事实命中（不是空跑）
    assert "身份证放在书房第一柜" not in details  # 已被取代的旧值不再进明细
    assert "身份证放在背包夹层" in details         # 当前有效值仍在

    mats = _section(ctx, "## 材料清单")
    old_line = next(line for line in mats.splitlines() if "身份证放在书房第一柜" in line)
    new_line = next(line for line in mats.splitlines() if "身份证放在背包夹层" in line)
    assert old_line.endswith("（已被更晚记录覆盖）")   # 旧记录材料行带标注（读侧标注，不删记录）
    assert "（已被更晚记录覆盖）" not in new_line      # 当前记录行不带
    assert fact_b.superseded_by is None


def test_material_annotation_skips_raw_only_and_mixed_facts(tmp_path, monkeypatch):
    """标注口径边界：raw 无事实（raw_only）不标；事实有活有死（混合）不标（v1 局限）。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)

    # raw_only：有材料、无事实（纯照片类记录；无事实可判覆盖）
    raw_only = store.insert_raw(s, kind="photo", content="一张照片的记录")
    s.flush()
    materials.upsert_for_record(s, raw_only, title=None, model=MODEL)

    # 混合：同一条记录上「活的事件」+「已被取代的状态事实」→ 不标
    raw_mixed = store.insert_raw(s, kind="text", content="买了鸡蛋，顺便把身份证放在第一柜")
    s.flush()
    store.insert_fact(
        s, ExtractedFact(type="consumption", summary="买了鸡蛋", amount=10.0,
                         currency="CNY", confidence=0.9), raw_mixed.id)
    s.flush()
    bitemporal.upsert(s, ExtractedFact(type="item_location", subject_key="身份证:位置",
                                       summary="身份证放在第一柜", confidence=0.9), raw_mixed.id)
    s.flush()
    bitemporal.upsert(s, ExtractedFact(type="item_location", subject_key="身份证:位置",
                                       summary="身份证放在第二柜", confidence=0.9), raw_mixed.id)
    s.flush()
    materials.upsert_for_record(s, raw_mixed, title=None, model=MODEL)
    s.commit()

    client = ScriptedClient([_completion(content="答")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    qa.run(s, "身份证", _settings(tmp_path))

    mats = _section(client.calls[0]["messages"][1]["content"], "## 材料清单")
    only_line = next(line for line in mats.splitlines() if "一张照片的记录" in line)
    mixed_line = next(line for line in mats.splitlines() if "买了鸡蛋" in line)
    assert "（已被更晚记录覆盖）" not in only_line   # raw_only 不标
    assert "（已被更晚记录覆盖）" not in mixed_line  # 混合事实不标



def test_answer_dispatches_by_engine(tmp_path, monkeypatch):
    s = _session(tmp_path)
    monkeypatch.setattr(answer, "answer_legacy", lambda s_, q, st: "LEGACY")
    monkeypatch.setattr(qa, "run", lambda s_, q, st: "NEW")
    assert answer.answer(s, "q", Settings(_env_file=None, qa_engine="legacy")) == "LEGACY"
    assert answer.answer(s, "q", Settings(_env_file=None)) == "NEW"  # 默认 tool_loop


def test_loop_failure_falls_back_to_legacy(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    monkeypatch.setattr(llm, "build_client", lambda settings: ScriptedClient([RuntimeError("boom")]))
    monkeypatch.setattr(answer, "answer_legacy", lambda s_, q, st: "FALLBACK")
    assert answer.answer(s, "学习率", _settings(tmp_path)) == "FALLBACK"


def test_insufficient_marker_becomes_readable(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    client = ScriptedClient([_completion(content="[信息不足]")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    resp = qa.run(s, "无关问题", _settings(tmp_path))
    assert resp.answer == "根据已有内容无法回答该问题。"


def test_explained_insufficient_keeps_explanation(tmp_path, monkeypatch):
    """「解释 + 标记」的终答保留解释、剥掉标记；只有整段就是标记才换兜底文案。

    真实案例（会话 26 评测排查）：模型答「记录里只有…没有写细节」并附标记，
    旧逻辑把整段替换成「根据已有内容无法回答该问题。」——用户能用的信息全丢。
    """
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    client = ScriptedClient([_completion(
        content="记录里只有一条「今天参加了公司的团建」，没有写具体做了什么。\n\n[信息不足]")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    resp = qa.run(s, "无关问题", _settings(tmp_path))
    assert "[信息不足]" not in resp.answer
    assert "今天参加了公司的团建" in resp.answer


def test_user_content_has_today_line(tmp_path, monkeypatch):
    """上下文首行给「今天」——此前模型不知道今天几号，时间类回答全受影响。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    client = ScriptedClient([_completion(content="答")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    qa.run(s, "学习率", _settings(tmp_path))
    content = client.calls[0]["messages"][1]["content"]
    first = content.splitlines()[0]
    assert first.startswith("今天：")
    assert "周" in first
    from app.memory.store import to_local, utcnow

    assert to_local(utcnow()).strftime("%Y-%m-%d") in first


def test_system_prompt_mentions_memory_stats():
    assert "memory_stats" in qa._SYSTEM_PROMPT
    assert "this_month" in qa._SYSTEM_PROMPT
    assert "type=consumption" in qa._SYSTEM_PROMPT  # 「几笔消费」口径（v3 实测：不带 type 数出 32 笔）


def test_used_facts_join_provenance(tmp_path, monkeypatch):
    """工具读过的明细 fact 并入溯源——回答里的数字能点回记录。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    raw = store.insert_raw(s, kind="text", content="午饭 花费32元")
    s.flush()
    fact = store.insert_fact(
        s, ExtractedFact(type="consumption", summary="午饭", amount=32.0, currency="CNY",
                         confidence=0.9), raw.id)
    s.commit()
    client = ScriptedClient([
        _completion(tool_calls=[_tool_call("t1", "memory_stats", {"period": "this_month"})]),
        _completion(content="这个月花了 32 元"),
    ])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    resp = qa.run(s, "这个月花了多少钱", _settings(tmp_path))
    assert any(src.fact_id == fact.id for src in resp.sources)
    assert "32" in resp.answer


def test_record_hit_pulls_group_document_into_scope(tmp_path, monkeypatch):
    """组关系传递：只命中「记录」材料时，同组文档的分块与图片同样进入上下文。

    真实案例：记录「我微信头像」命中，但照片本体（同组文档）里的图必须能被看到。
    """
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    raw = store.insert_raw(s, kind="photo", content="我微信头像")
    s.flush()
    fact = store.insert_fact(
        s, ExtractedFact(type="other", summary="微信头像", confidence=0.9), raw.id
    )
    s.flush()
    doc = documents.insert_document(
        s, kind="photo", title="avatar.jpg", file_path="files/a.jpg",
        status="done", allow_cloud=1, raw_id=raw.id,
    )
    s.flush()
    c = documents.insert_chunk(
        s, document_id=doc.id, seq=0, text="豆包AI生成", parent_text="豆包AI生成"
    )
    s.flush()
    chunks.index_chunk(s, c.id, [0.1] * 512, "豆包AI生成")
    s.add(Image(id="img1", document_id=doc.id, file_path="files/a.jpg"))
    record_material = materials.upsert_for_record(s, raw, title=fact.summary, model=MODEL)
    s.commit()

    # 材料层只回「记录」这一条（模拟文档材料没进 top-8 的现实场景）
    monkeypatch.setattr(retrieval, "search_materials_multi", lambda *a, **k: [record_material.id])

    client = ScriptedClient([_completion(content="你的头像是……")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    resp = qa.run(s, "我微信头像长什么样", _settings(tmp_path))

    user_content = client.calls[0]["messages"][1]["content"]
    assert "可用图片" in user_content and "img1" in user_content  # 图片随组进入清单
    assert "豆包AI生成" in user_content                            # 同组分块也进了细节
    assert resp.answer == "你的头像是……"


def test_provenance_sources_carry_attachment_doc_ids(tmp_path, monkeypatch):
    """回答的来源要能点开原件：Source 带 raw_id + 该记录的未删附件 doc_ids（前端拼 /raw 链接用）。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s)
    raw = store.insert_raw(s, kind="text", content="体检报告 存了")
    s.flush()
    doc = documents.insert_document(
        s, kind="word", title="体检报告.pdf", file_path="files/t.pdf", status="done",
        raw_id=raw.id)
    # 同一记录的第二份附件已软删：不进 doc_ids（列出来的话前端点开必 404）
    doc_deleted = documents.insert_document(
        s, kind="word", title="体检报告-旧版.pdf", file_path="files/t-old.pdf", status="done",
        raw_id=raw.id, deleted_at=store.utcnow())
    fact = store.insert_fact(
        s, ExtractedFact(type="health", summary="存了体检报告", confidence=0.9), raw.id)
    s.flush()
    fts.index_fact(s, fact.id, "存了体检报告")
    vector.index_fact(s, fact.id, [0.1] * 512)
    materials.upsert_for_record(s, raw, title=None, model=MODEL)  # 材料层命中是二级检索的前提
    s.commit()
    client = ScriptedClient([_completion(
        tool_calls=[_tool_call("t1", "read_document", {"doc_id": doc.id})]),
        _completion(content="已存体检报告")])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)
    resp = qa.run(s, "体检报告在哪", _settings(tmp_path))
    src = next(x for x in resp.sources if x.fact_id == fact.id)
    assert src.raw_id == raw.id
    assert src.doc_ids == [doc.id]
    assert doc_deleted.id not in src.doc_ids  # 软删排除：已删附件不列进来源


# --- 口径落地（spec 2026-10-16-stats-param-guard §3.4）----------------------
# 追加在本文件：复用既有 _session/_settings/_completion/_tool_call/ScriptedClient 夹具。


def test_stats_caliber_appended_when_answer_omits_it(tmp_path, monkeypatch):
    """调过统计、而终答没写口径 → 代码在末尾补一行（口径一定可见）。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    doc, _ = _seed_doc(s, title="成绩总表", text="课程学分")
    table = documents.insert_table(
        s, document_id=doc.id,
        headers=json.dumps(["课程", "学分"], ensure_ascii=False),
        rows=json.dumps([["体育", "0.5"], ["高数", "3.0"]], ensure_ascii=False),
    )
    s.commit()
    client = ScriptedClient([
        _completion(tool_calls=[_tool_call("t1", "table_stats", {
            "table_id": table.id, "column": "学分", "op": "<", "value": "1", "aggregate": "none",
        })]),
        _completion(content="只有体育课学分少于 1。"),   # 答案里没提筛选口径
    ])
    monkeypatch.setattr(llm, "build_client", lambda settings: client)

    resp = qa.run(s, "学分少于1的课有哪些", _settings(tmp_path))

    assert resp.answer.startswith("只有体育课学分少于 1。")
    assert "统计口径：按「学分 < 1」筛选" in resp.answer


def test_run_leaves_answer_untouched_without_stats(tmp_path, monkeypatch):
    """一轮下来没调统计 → 终答逐字不变（零打扰）。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    _seed_doc(s, title="实验报告", text="第三组学习率 0.001")
    s.commit()
    monkeypatch.setattr(llm, "build_client", lambda settings: ScriptedClient([
        _completion(content="第三组学习率是 0.001。"),
    ]))

    resp = qa.run(s, "第三组学习率是多少", _settings(tmp_path))

    assert resp.answer == "第三组学习率是 0.001。"


def test_answer_engine_legacy_has_no_caliber(tmp_path, monkeypatch):
    """旧管线（legacy）不追加口径——钉住 Review Focus #5：口径只在 qa.run 里落地。"""
    _patch_embed(monkeypatch)
    s = _session(tmp_path)
    monkeypatch.setattr(llm, "build_client", lambda settings: ScriptedClient([
        _completion(content="qa"),                     # router.classify 的一次调用
    ]))

    resp = answer.answer(s, "这个月花了多少钱", _settings(tmp_path, qa_engine="legacy"))

    assert "没有找到相关记录" in resp.answer
    assert "统计口径" not in resp.answer
