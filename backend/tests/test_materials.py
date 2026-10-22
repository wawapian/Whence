"""材料层：每份材料（记录/文档各一行）入库时生成一条可检索索引（摘要规则 / 幂等 / 截断兜底 / 检索）。"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app import deps
from app.api.files import process_document
from app.config import Settings
from app.core import extraction, llm, queue, retrieval, trash
from app.main import create_app
from app.memory import documents, materials, store
from app.models.db import Document, Material, RawEvidence, init_db
from app.models.schemas import ExtractedFact

MODEL = "BAAI/bge-small-zh-v1.5"


def _session(tmp_path, monkeypatch, dim: int = 4) -> Session:
    """直连会话（照 test_chunk_index 的轻量骨架）；embedding 一律替身，不加载真模型。"""
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * dim)
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine, dim=dim)
    return Session(engine)


def test_record_upsert_creates_visible_material(tmp_path, monkeypatch):
    """记录入库（POST /api/memories）→ materials 1 行，material_fts/material_vec 索引行就位。"""
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model=MODEL
    )
    monkeypatch.setattr(
        extraction, "extract",
        lambda client, model, text: [ExtractedFact(
            type="item_location", subject_key="身份证:位置", summary="身份证放书房", confidence=0.9
        )],
    )
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)

    def _submit(fn, *args, **kwargs):  # 记录整理内联执行（accept-fast 的整理在后台）
        lane = kwargs.pop("lane", "doc")  # 车道参数由队列消费，不进任务函数
        if lane == "record":
            return fn(*args, **kwargs) or "test-inline"
        return "test-noop"

    monkeypatch.setattr(queue, "submit", _submit)
    resp = TestClient(app).post("/api/memories", data={"text": "身份证放在书房左边第二个抽屉"})
    assert resp.status_code == 200

    s = _session(tmp_path, monkeypatch, dim=512)  # 同库另开会话读回（init_db 装 vec 扩展钩子）
    assert s.query(Material).count() == 1
    material = s.query(Material).one()
    raw = s.query(RawEvidence).one()
    assert (material.source_type, material.source_id) == ("record", raw.id)
    assert material.title == "身份证放书房"  # title 由调用方传 fact 摘要
    assert material.summary == "身份证放在书房左边第二个抽屉"
    assert material.vec_rowid is not None
    assert s.execute(
        text("SELECT content FROM material_fts WHERE material_id = :m"), {"m": material.id}
    ).scalar() == "身份证放书房 身份证放在书房左边第二个抽屉"  # 索引内容 = title + 摘要
    assert s.execute(
        text("SELECT count(*) FROM material_vec WHERE rowid = :r"), {"r": material.vec_rowid}
    ).scalar() == 1


def test_document_intake_creates_material(tmp_path, monkeypatch):
    """文档入库（process_document）→ materials 1 行；无 summary/caption 时摘要 = 正文前 200 字。"""
    settings = Settings(data_dir=tmp_path, embedding_model=MODEL)
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.txt").write_text("查重报告 相似度 15%", encoding="utf-8")
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)

    factory = deps._session_factory(settings)
    with factory() as s:
        doc = documents.insert_document(
            s, kind="text", title="a.txt", file_path="files/a.txt", status="processing"
        )
        s.commit()
        doc_id = doc.id

    process_document(settings, doc_id, "files/a.txt", "off", 0, 0)

    with factory() as s:
        assert s.get(Document, doc_id).status == "done"
        assert s.query(Material).count() == 1
        material = s.query(Material).one()
        assert (material.source_type, material.source_id) == ("document", doc_id)
        assert material.title == "a.txt"
        assert material.summary == "查重报告 相似度 15%"
        assert material.vec_rowid is not None
        assert s.execute(
            text("SELECT count(*) FROM material_fts WHERE material_id = :m"), {"m": material.id}
        ).scalar() == 1


def test_table_only_document_skips_cloud_summary(tmp_path, monkeypatch):
    """表格类文档（正文只剩表格摘要行）不调云总结——退化输入只会换回抱怨文本。

    会话 26 复跑实测：云总结对「表格（34行×4列：姓名、语文、数学、英语）」仍回出抱怨
    变体（「无法处理：…请粘贴表格中的具体数据」），短语封堵法漏网；改为结构化判定，
    材料摘要由表结构构造（班名/列名/取值进检索）。
    """
    from app.core import cloud
    from app.core.ingest import pipeline as pipeline_mod
    from app.core.ingest.models import ParsedDocument, TextChunk
    from app.core.tables import ParsedTable

    settings = Settings(data_dir=tmp_path, embedding_model=MODEL)
    (tmp_path / "files").mkdir()
    monkeypatch.setattr(llm, "embed", lambda q, model=None: [0.1] * 512)
    calls: list[str] = []
    monkeypatch.setattr(cloud, "summarize", lambda c, m, t: calls.append(t) or "云端总结")

    def _parsed(text: str) -> ParsedDocument:
        return ParsedDocument(kind="other", title=None, full_text=text, headings=[],
                              blocks=[], engine_used="t", ocr_used=False, needs_ocr=None)

    def _chunk(text: str) -> TextChunk:
        return TextChunk(text=text, heading_path="表格", page=None, parent_text="", seq=0)

    prose = pipeline_mod.IngestResult(
        parsed=_parsed("查重报告 相似度 15%"), chunks=[_chunk("查重报告 相似度 15%")])
    table_only = pipeline_mod.IngestResult(
        parsed=_parsed("表格（1行×4列：姓名、语文、数学、英语）"),
        chunks=[_chunk("表头：姓名 | 语文 | 数学 | 英语\n朱若曦 | 61 | 93 | 87")],
        tables=[ParsedTable(headers=["姓名", "语文", "数学", "英语"],
                            rows=[["朱若曦", "61", "93", "87"]],
                            heading="三年级四班（东校区）期末成绩表")],
    )
    monkeypatch.setattr(pipeline_mod, "ingest",
                        lambda path, **kw: table_only if str(path).endswith(".xlsx") else prose)

    factory = deps._session_factory(settings)
    with factory() as s:
        prose_doc = documents.insert_document(
            s, kind="text", title="a.txt", file_path="files/a.txt", status="processing")
        table_doc = documents.insert_document(
            s, kind="text", title="grades_007.xlsx", file_path="files/x.xlsx",
            status="processing")
        s.commit()
        prose_id, table_id = prose_doc.id, table_doc.id

    process_document(settings, prose_id, "files/a.txt", "off", 1, 0)
    process_document(settings, table_id, "files/x.xlsx", "off", 1, 0)

    with factory() as s:
        assert calls == ["查重报告 相似度 15%"]        # 只有散文类文档调了云总结
        assert s.get(Document, prose_id).summary == "云端总结"
        table_material = s.query(Material).filter(Material.source_id == table_id).one()
        assert "三年级四班（东校区）" in table_material.summary
        assert "数学" in table_material.summary


def test_document_upsert_summary_priority(tmp_path, monkeypatch):
    """文档摘要优先级：云端 summary → 第一张图 caption → 正文前 200 字 → 标题。"""
    s = _session(tmp_path, monkeypatch)

    def _doc(**fields) -> Document:
        d = documents.insert_document(s, kind="text", status="done", **fields)
        s.flush()
        return d

    with_summary = _doc(title="文档一", summary="云端总结", parsed_text="正文首段")
    with_caption = _doc(title="文档二", parsed_text="正文首段")
    documents.insert_image(
        s, document_id=with_caption.id, page=1, ocr_text="", caption="图中是成绩总表"
    )
    s.flush()
    with_text = _doc(title="文档三", parsed_text="正" * 300)
    with_nothing = _doc(title="文档四")

    m1 = materials.upsert_for_document(s, with_summary, model=MODEL)
    m2 = materials.upsert_for_document(s, with_caption, model=MODEL)
    m3 = materials.upsert_for_document(s, with_text, model=MODEL)
    m4 = materials.upsert_for_document(s, with_nothing, model=MODEL)
    s.commit()

    assert m1.summary == "云端总结"
    assert m2.summary == "图中是成绩总表"
    assert m3.summary == "正" * 200
    assert m4.summary == "文档四"  # 全空兜底到标题
    assert [m.title for m in (m1, m2, m3, m4)] == ["文档一", "文档二", "文档三", "文档四"]


def test_document_material_uses_table_text_when_summary_missing(tmp_path, monkeypatch):
    """表类文档：无云摘要时材料摘要改由结构化表构造——表标题/列名/取值才进得了检索。

    真实案例（会话 26 评测排查）：成绩表正文只剩「表格（N行×M列…）」，云总结回抱怨文本
    被质量门丢弃后，材料层若仍用退化正文，问题照样召不回。
    """
    import json

    s = _session(tmp_path, monkeypatch)
    doc = documents.insert_document(s, kind="text", status="done", title="grades_007.xlsx")
    s.flush()
    documents.insert_table(
        s, document_id=doc.id, heading="三年级四班（东校区）期末成绩表",
        headers=json.dumps(["姓名", "语文", "数学", "英语"], ensure_ascii=False),
        rows=json.dumps([["朱若曦", "61", "93", "87"], ["郭嘉怡", "66", "88", "90"]],
                        ensure_ascii=False),
    )
    s.flush()
    m = materials.upsert_for_document(s, doc, model=MODEL)
    s.commit()
    assert "三年级四班（东校区）" in m.summary   # 表标题（班名）
    assert "数学" in m.summary                  # 列名
    assert "朱若曦" in m.summary                # 取值样本
    assert len(m.summary) <= materials.SUMMARY_MAX


def test_document_material_prepends_table_values_even_with_summary(tmp_path, monkeypatch):
    """表类文档：材料摘要**前置表的取值**（哪怕已有云摘要）——按公司名/班名提问才命得中（v3 实测抓错表）。"""
    import json

    s = _session(tmp_path, monkeypatch)
    doc = documents.insert_document(s, kind="text", status="done", title="jobs_006.xlsx",
                                    summary="**摘要**：表格列出15家公司的岗位、月薪范围和学历要求")
    s.flush()
    documents.insert_table(
        s, document_id=doc.id, heading="岗位信息汇总表",
        headers=json.dumps(["公司", "岗位", "月薪范围", "学历要求"], ensure_ascii=False),
        rows=json.dumps([["样例科技有限公司", "调度员", "10-20K", "本科"],
                         ["松间检测有限公司", "注塑工", "8-15K", "大专"]], ensure_ascii=False),
    )
    s.flush()
    m = materials.upsert_for_document(s, doc, model=MODEL)
    s.commit()
    assert "样例科技有限公司" in m.summary     # 取值前置（检索信号）
    assert "表格列出15家公司" in m.summary      # 云摘要仍在（截断内）
    assert len(m.summary) <= materials.SUMMARY_MAX


def test_material_summary_truncated_and_title_fallback(tmp_path, monkeypatch):
    """超长原话截 200；记录/文档全空也不缺条目（title 兜底、条目仍存在）。"""
    s = _session(tmp_path, monkeypatch)

    long_raw = store.insert_raw(s, kind="text", content="长" * 300)
    empty_raw = store.insert_raw(s, kind="text", content=None)
    empty_doc = documents.insert_document(s, kind="text", title="空文档", status="done")
    s.flush()

    m_long = materials.upsert_for_record(s, long_raw, title=None, model=MODEL)
    m_empty = materials.upsert_for_record(s, empty_raw, title=None, model=MODEL)
    m_doc = materials.upsert_for_document(s, empty_doc, model=MODEL)
    s.commit()

    assert m_long.summary == "长" * 200
    assert m_long.title == "长" * 40  # title 缺省 → 原话前 40 字
    assert (m_empty.title, m_empty.summary) == ("记录", "")  # 全空：标题兜底
    assert m_doc.summary == "空文档"  # 文档全空：摘要兜底到标题
    assert s.query(Material).count() == 3  # 条目永不缺失


def test_upsert_is_idempotent_no_duplicate_index_rows(tmp_path, monkeypatch):
    """同源重复入库：materials 恒一行，FTS/向量行先删旧再重建，不产生重复索引行。"""
    s = _session(tmp_path, monkeypatch)

    raw = store.insert_raw(s, kind="text", content="身份证放在书房")
    doc = documents.insert_document(
        s, kind="text", title="成绩总表", parsed_text="数学 95", status="done"
    )
    s.flush()
    materials.upsert_for_record(s, raw, title="旧摘要", model=MODEL)
    materials.upsert_for_document(s, doc, model=MODEL)
    s.commit()

    materials.upsert_for_record(s, raw, title="新摘要", model=MODEL)
    materials.upsert_for_document(s, doc, model=MODEL)
    s.commit()

    assert s.query(Material).count() == 2
    assert s.execute(text("SELECT count(*) FROM material_fts")).scalar() == 2
    assert s.execute(text("SELECT count(*) FROM material_vec")).scalar() == 2
    record_material = s.query(Material).filter_by(source_type="record").one()
    assert record_material.title == "新摘要"
    assert s.execute(
        text("SELECT count(*) FROM material_fts WHERE material_id = :m AND content LIKE :pat"),
        {"m": record_material.id, "pat": "%新摘要%"},
    ).scalar() == 1  # 旧索引行已删，留下的是重建后的新内容
    assert record_material.vec_rowid is not None
    assert s.execute(
        text("SELECT count(*) FROM material_vec WHERE rowid = :r"), {"r": record_material.vec_rowid}
    ).scalar() == 1


def test_material_search_finds_target(tmp_path, monkeypatch):
    """3 条材料查其一：FTS 与向量两路 top1 均命中目标材料。"""
    s = _session(tmp_path, monkeypatch)

    def fake_embed(text, model=None):  # 目标材料向量贴近查询向量，其余远离
        return [0.9] * 4 if "身份证" in text else [0.1] * 4

    monkeypatch.setattr(llm, "embed", fake_embed)
    r1 = store.insert_raw(s, kind="text", content="身份证放在书房抽屉")
    r2 = store.insert_raw(s, kind="text", content="今天买了鸡蛋和牛奶")
    r3 = store.insert_raw(s, kind="text", content="周五下午开会讨论预算")
    s.flush()
    m1 = materials.upsert_for_record(s, r1, title=None, model=MODEL)
    m2 = materials.upsert_for_record(s, r2, title=None, model=MODEL)
    m3 = materials.upsert_for_record(s, r3, title=None, model=MODEL)
    s.commit()

    assert materials.fts_search(s, "身份证", k=5) == [m1.id]  # 关键词路只有目标命中
    assert materials.vector_search(s, [0.9] * 4, k=5)[0] == m1.id  # 向量路 top1
    assert set(materials.vector_search(s, [0.1] * 4, k=5)) == {m1.id, m2.id, m3.id}  # 三条都在索引里


def test_material_search_excludes_deleted_sources(tmp_path, monkeypatch):
    """软删文档/记录 → 各自材料查不到、半边状态按源各自判断；源行不存在的残留材料查不到也不报错。"""
    s = _session(tmp_path, monkeypatch)

    raw = store.insert_raw(s, kind="text", content="身份证放在书房抽屉")
    doc = documents.insert_document(
        s, kind="text", title="成绩总表", parsed_text="数学 95 分", status="done"
    )
    s.flush()
    m_rec = materials.upsert_for_record(s, raw, title=None, model=MODEL)
    m_doc = materials.upsert_for_document(s, doc, model=MODEL)
    # 边界：源对象被彻底删除（purge 已清账，理论不应残留）→ 查不到即可，不报错
    orphan = Material(id=store.new_id(), source_type="document", source_id="已不存在的源")
    s.add(orphan)
    s.flush()
    materials.index_material(s, orphan.id, [0.1] * 4, "孤儿材料占位")
    s.commit()

    # 基线：软删前都可命中（证明后续是「过滤」而非查询本身失效）
    assert m_rec.id in materials.fts_search(s, "身份证", k=5)
    assert m_doc.id in materials.fts_search(s, "成绩总表", k=5)
    assert {m_rec.id, m_doc.id} <= set(materials.vector_search(s, [0.1] * 4, k=5))
    assert orphan.id not in materials.fts_search(s, "孤儿材料", k=5)
    assert orphan.id not in materials.vector_search(s, [0.1] * 4, k=5)

    trash.soft_delete_document(s, doc.id)  # 只删文档侧：记录侧材料照常命中（半边按源各自判断）
    s.commit()
    assert materials.fts_search(s, "成绩总表", k=5) == []
    assert m_doc.id not in materials.vector_search(s, [0.1] * 4, k=5)
    assert m_rec.id in materials.fts_search(s, "身份证", k=5)
    assert m_rec.id in materials.vector_search(s, [0.1] * 4, k=5)

    trash.soft_delete_record(s, raw.id)
    s.commit()
    assert materials.fts_search(s, "身份证", k=5) == []
    assert m_rec.id not in materials.vector_search(s, [0.1] * 4, k=5)


def test_search_materials_multi_fuses_queries(tmp_path, monkeypatch):
    """多查询 RRF 融合：变体查询把另一条材料也带进结果，返回去重、不炸。"""
    s = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(materials, "vector_search", lambda *a, **k: [])  # 隔离向量路，只测 FTS 融合

    r1 = store.insert_raw(s, kind="text", content="第三组学习率 0.001")
    r2 = store.insert_raw(s, kind="text", content="今天买了鸡蛋")
    s.flush()
    m1 = materials.upsert_for_record(s, r1, title=None, model=MODEL)
    m2 = materials.upsert_for_record(s, r2, title=None, model=MODEL)
    s.commit()

    assert retrieval.search_materials_multi(s, ["买了鸡蛋"], top_k=5) == [m2.id]  # 单查询只命中 m2
    fused = retrieval.search_materials_multi(s, ["买了鸡蛋", "学习率"], top_k=5)
    assert set(fused) == {m1.id, m2.id}  # 变体把 m1 也带进来
    assert len(fused) == len(set(fused))  # 去重
