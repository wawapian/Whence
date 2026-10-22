import io

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings
from app.core import extraction, llm, parsing, queue
from app.main import create_app
from app.models.db import Document, Fact, RawEvidence, Secret
from app.models.schemas import ExtractedFact


def _client(tmp_path, monkeypatch, queue_recorder=None):
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5"
    )

    def fake_extract(client, model, text):
        return [ExtractedFact(
            type="item_location", subject_key="身份证:位置",
            summary="身份证放书房", confidence=0.9,
        )]

    monkeypatch.setattr(extraction, "extract", fake_extract)
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)

    # 记录车道内联执行（accept-fast 后整理在后台，测试里当场跑完好断言 DB）；
    # doc 车道照旧禁掉，防残留后台线程在 monkeypatch 撤销后调真实 llm.embed（加载模型/联网）
    def _submit(fn, *args, **kwargs):
        lane = kwargs.pop("lane", "doc")  # 车道参数由队列消费，不进任务函数
        if lane == "record":
            if queue_recorder is not None:
                queue_recorder.append((fn.__name__, "record"))
                return "test-record"
            return fn(*args, **kwargs) or "test-inline"
        return "test-noop"

    monkeypatch.setattr(queue, "submit", _submit)
    return TestClient(app)


def test_record_text(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post("/api/memories", data={"text": "身份证放在书房左边第二个抽屉"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "accepted"  # accept-fast：整理在后台（测试内联执行）

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with Session(engine) as s:
        assert s.query(RawEvidence).count() == 1
        assert s.query(Fact).count() == 1
        raw = s.query(RawEvidence).one()
        assert raw.content == "身份证放在书房左边第二个抽屉"  # 原话原文入库


def test_record_file_with_text(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("receipt.jpg", io.BytesIO(b"fake-image"), "image/jpeg")}
    resp = client.post("/api/memories", data={"text": "这是消费单"}, files=files)
    assert resp.status_code == 200

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with Session(engine) as s:
        raw = s.query(RawEvidence).first()
        assert raw.kind == "photo"
        assert raw.file_path is not None
        assert (tmp_path / raw.file_path).exists()
        assert raw.content == "这是消费单"  # 原话原文入库
        assert s.query(Fact).count() == 1  # 文字也抽了 FACT


def test_record_file_only(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("contract.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")}
    resp = client.post("/api/memories", files=files)
    assert resp.status_code == 200
    assert resp.json()["state"] == "accepted"

    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    with Session(engine) as s:
        raw = s.query(RawEvidence).first()
        assert raw.kind == "document"
        assert s.query(Fact).count() == 0  # 无文字且解析不出正文 → 整理后不建事实


def test_record_document_with_text_keeps_user_text(tmp_path, monkeypatch):
    """「文档+文字」：RAW 存用户原话（原话优先，不被解析正文顶掉），事实索引同源。"""
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")
    files = {"file": ("f.docx", io.BytesIO(b"fake-docx"),
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    resp = client.post("/api/memories", data={"text": "这是我大三上的查重报告"}, files=files)
    assert resp.status_code == 200

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        raw = s.query(RawEvidence).one()
        assert raw.content == "这是我大三上的查重报告"  # 原话优先，正文不再顶掉原话
        fact = s.query(Fact).one()
        indexed = s.execute(
            text("SELECT content FROM memory_fts WHERE fact_id = :f"), {"f": fact.id}
        ).scalar()
        assert "这是我大三上的查重报告" in indexed  # 索引文本用原话
        assert "相似度" not in indexed               # 解析正文不进该条事实的索引（它走文档分块）


def test_record_document_without_text_keeps_parsed_body(tmp_path, monkeypatch):
    """「文档+无文字」：RAW 回落文档解析正文（回归：原行为不变）。"""
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")
    files = {"file": ("f.docx", io.BytesIO(b"fake-docx"),
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    resp = client.post("/api/memories", files=files)
    assert resp.status_code == 200

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        raw = s.query(RawEvidence).one()
        assert raw.content == "查重报告 相似度 15%"


def test_record_multiple_files_grouped(tmp_path, monkeypatch):
    """多附件成组：1 图片 + 1 docx → 1 条 raw（kind=document）+ 2 个 Document（同一 raw_id、各自 file_path）。"""
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")
    seen = {}

    def capture_extract(client_, model, text_):
        seen["desc"] = text_
        return [ExtractedFact(type="other", summary="查重报告", confidence=0.9)]

    monkeypatch.setattr(extraction, "extract", capture_extract)
    files = [
        ("files", ("a.jpg", io.BytesIO(b"fake-image"), "image/jpeg")),
        ("files", ("b.docx", io.BytesIO(b"fake-docx"),
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ]
    resp = client.post("/api/memories", files=files)
    assert resp.status_code == 200

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        raw = s.query(RawEvidence).one()
        assert raw.kind == "document"  # 含非图片 → document
        assert raw.file_path is not None
        assert raw.content == "查重报告 相似度 15%"
        docs = s.query(Document).all()
        assert len(docs) == 2
        assert {d.raw_id for d in docs} == {raw.id}  # 同一条记录（组）
        assert len({d.file_path for d in docs}) == 2  # 各附件独立路径
        assert all((tmp_path / d.file_path).exists() for d in docs)
        assert {d.title for d in docs} == {"a.jpg", "b.docx"}
    assert seen["desc"] == "上传了 2 个文件：a.jpg、b.docx"


def test_record_single_file_desc_unchanged(tmp_path, monkeypatch):
    """单附件（走旧 `file` 键）抽取文案逐字不变：`上传了文档：{filename}`（回归）。"""
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(parsing, "parse_document", lambda p: "查重报告 相似度 15%")
    seen = {}

    def capture_extract(client_, model, text_):
        seen["desc"] = text_
        return [ExtractedFact(type="other", summary="查重报告", confidence=0.9)]

    monkeypatch.setattr(extraction, "extract", capture_extract)
    files = {"file": ("f.docx", io.BytesIO(b"fake-docx"),
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    resp = client.post("/api/memories", files=files)
    assert resp.status_code == 200

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        docs = s.query(Document).all()
        assert len(docs) == 1
        assert docs[0].raw_id == s.query(RawEvidence).one().id
    assert seen["desc"] == "上传了文档：f.docx"


def test_record_one_bad_attachment_does_not_break_others(tmp_path, monkeypatch):
    """单份解析抛错：该文件仍建 Document，raw.content 只含成功解析的文本，接口 200。"""
    client = _client(tmp_path, monkeypatch)

    def flaky_parse(path):
        if path.suffix == ".pdf":
            raise RuntimeError("corrupted pdf")
        return "查重报告 相似度 15%"

    monkeypatch.setattr(parsing, "parse_document", flaky_parse)
    files = [
        ("files", ("bad.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")),
        ("files", ("good.docx", io.BytesIO(b"fake-docx"),
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ]
    resp = client.post("/api/memories", files=files)
    assert resp.status_code == 200

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        raw = s.query(RawEvidence).one()
        assert raw.content == "查重报告 相似度 15%"  # 只含成功解析的文本
        assert s.query(Document).count() == 2        # 坏文件也照建 Document


def test_two_sensitive_records_no_placeholder_collision(tmp_path, monkeypatch):
    """回归：同库第二条含敏感值的记录不得再撞 secrets 占位符主键（修复前整条 500）。

    同时断言生产链路把 secrets 关联到 fact（`fact_id` 非空），彻底删除时清得到。
    """
    client = _client(tmp_path, monkeypatch)
    r1 = client.post("/api/memories", data={"text": "我的手机号是 13800000001"})
    r2 = client.post("/api/memories", data={"text": "备用手机号 13900000002"})
    assert r1.status_code == 200 and r2.status_code == 200

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        rows = s.query(Secret).all()
        assert len(rows) == 2
        assert len({r.placeholder for r in rows}) == 2  # 占位符跨记录唯一
        assert all(r.fact_id for r in rows)             # 已关联 fact


def test_accept_fast_returns_before_consolidation(tmp_path, monkeypatch):
    """accept-fast：请求只落盘入库即返回（整理任务仅入队、未执行）——批量上传不再排队等待。"""
    submitted = []
    client = _client(tmp_path, monkeypatch, queue_recorder=submitted)
    resp = client.post("/api/memories", data={"text": "身份证放在书房"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "accepted" and body["raw_id"]
    assert ("consolidate_record", "record") in submitted  # 整理任务已入队（未执行）

    with Session(create_engine(f"sqlite:///{tmp_path}/mem.db")) as s:
        assert s.query(RawEvidence).count() == 1  # 同步段：RAW 已落库
        assert s.query(Fact).count() == 0         # 整理未执行：事实尚不存在


def test_timeline_shows_raw_only_record(tmp_path, monkeypatch):
    """无事实的纯记录（纯照片等）也要出现在时间线——「我存了东西就要看得见」。

    真实案例：微信头像照片（无文字）存完在时间线隐形；现在以「（照片）文件名」条目呈现。
    """
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("avatar.jpg", io.BytesIO(b"\xff\xd8\xff\xe0fake"), "image/jpeg")}
    resp = client.post("/api/memories", files=files)
    assert resp.status_code == 200
    raw_id = resp.json()["raw_id"]

    (item,) = client.get("/api/memories").json()
    assert item["fact_id"] is None                # 纯照片记录：无事实
    assert item["raw_id"] == raw_id               # 但有 raw，删除/展示都可用
    assert item["type"] == "photo"
    assert item["summary"] == "（照片）avatar.jpg"
