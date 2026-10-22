import time

from fastapi.testclient import TestClient

from app import deps
from app.config import Settings
from app.core import cloud, llm
from app.core.ingest import pipeline as pipeline_mod
from app.core.ingest.models import ParsedDocument, TextChunk
from app.main import create_app


def test_summarize_calls_llm(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    FakeClient.instance.calls += 1
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "摘要"})()})()]})()
    FakeClient.instance = FakeClient()
    monkeypatch.setattr(llm, "build_client", lambda s: FakeClient.instance)
    assert cloud.summarize(FakeClient.instance, "m", "正文") == "摘要"
    assert FakeClient.instance.calls == 1


def test_summarize_rejects_prompt_echo_refusal(monkeypatch):
    """模型收到退化输入（只剩标题+行列数）会回「请提供文本内容…」——这类抱怨不得当摘要入库。

    真实案例（会话 26 评测排查）：xlsx 正文被表格摘要替换后送云总结，模型回抱怨文本，
    原样入库成 doc.summary → 材料层零信号 → 该文档的问题召不回。
    """
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {
                        "content": "好的，请提供需要处理的文本内容（即表格中的具体数据或文字），"
                                   "我会据此进行去噪并提炼要点，输出简洁摘要。"})()})()]})()
    out = cloud.summarize(FakeClient(), "m", "表格（36行×4列：三年级四班（东校区）期末成绩表、、、）")
    assert out == ""


def test_summarize_keeps_summary_merely_mentioning_weihanzhi(monkeypatch):
    """终审：拒绝变体「未包含」收窄——合法摘要里出现「未包含」（如"本表未包含薪资列"）不得整条丢弃。"""
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {
                        "content": "**摘要**：本表列了岗位与学历，未包含薪资信息。"})()})()]})()
    out = cloud.summarize(FakeClient(), "m", "表结构说明")
    assert "未包含薪资信息" in out


def test_describe_image_calls_vision_llm(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    FakeClient.instance.calls.append(kwargs)
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "查重报告 15%"})()})()]})()
    FakeClient.instance = FakeClient()
    out = cloud.describe_image(FakeClient.instance, "qwen-vl-plus", b"\xff\xd8\xff\xe0fakejpeg", "这张图讲了什么？")
    assert out == "查重报告 15%"
    assert len(FakeClient.instance.calls) == 1
    kwargs = FakeClient.instance.calls[0]
    assert kwargs["model"] == "qwen-vl-plus"
    content = kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "这张图讲了什么？"}
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def _describe_image_url(image_bytes):
    class FakeClient:
        def __init__(self):
            self.calls = []
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    FakeClient.instance.calls.append(kwargs)
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()
    FakeClient.instance = FakeClient()
    cloud.describe_image(FakeClient.instance, "qwen-vl-plus", image_bytes, "q")
    return FakeClient.instance.calls[0]["messages"][0]["content"][1]["image_url"]["url"]


def test_describe_image_png_mime(monkeypatch):
    url = _describe_image_url(b"\x89PNG\r\n\x1a\nfakepng")
    assert url.startswith("data:image/png;base64,")


def test_describe_image_jpeg_mime(monkeypatch):
    url = _describe_image_url(b"\xff\xd8\xff\xe0fakejpeg")
    assert url.startswith("data:image/jpeg;base64,")


def test_describe_image_unknown_defaults_jpeg(monkeypatch):
    url = _describe_image_url(b"\x00\x01\x02not-an-image")
    assert url.startswith("data:image/jpeg;base64,")


def test_vlm_settings_defaults():
    s = Settings(_env_file=None)
    assert s.vlm_model == ""
    assert s.vlm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert s.vlm_api_key == ""


def _client(tmp_path, monkeypatch):
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        data_dir=tmp_path, embedding_model="BAAI/bge-small-zh-v1.5")
    monkeypatch.setattr(llm, "embed", lambda text, model=None: [0.1] * 512)
    fake = pipeline_mod.IngestResult(
        parsed=ParsedDocument(kind="word", title="t", full_text="正文",
                              headings=[], blocks=[], engine_used="python-docx"),
        chunks=[TextChunk("块", None, None, "块", 0)],
    )
    monkeypatch.setattr(pipeline_mod, "ingest", lambda *a, **k: fake)
    return TestClient(app)


def test_files_no_cloud_no_llm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cloud, "summarize", lambda c, m, t: calls.append(1) or "S")
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("a.docx", b"fake", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    r = client.post("/api/files", files=files, data={"allow_cloud": "0", "multimodal": "0", "ocr": "auto"})
    did = r.json()["document_id"]
    for _ in range(50):
        time.sleep(0.1)
        d = client.get(f"/api/files/{did}").json()
        if d["status"] == "done":
            break
    assert calls == []           # 不上传云 → 不调 LLM


def test_files_allow_cloud_calls_llm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cloud, "summarize", lambda c, m, t: calls.append(1) or "S")
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("a.docx", b"fake", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    r = client.post("/api/files", files=files, data={"allow_cloud": "1", "multimodal": "0", "ocr": "auto"})
    did = r.json()["document_id"]
    # status=done 在云总结之前写入；必须等到 summarize 真正执行并回写 summary，
    # 否则固定 5 秒轮询会在慢机器上抢先断言（既有竞态）。
    for _ in range(100):
        time.sleep(0.1)
        d = client.get(f"/api/files/{did}").json()
        if calls and d.get("summary") == "S":
            break
    assert len(calls) == 1       # 上传云 → 调一次 LLM
    assert d["summary"] == "S"


def test_files_cloud_failure_keeps_done(tmp_path, monkeypatch):
    """云总结失败不阻塞：本地解析/分块已落库，文档仍 done，summary 为空。"""
    def _boom(client, model, text):
        raise RuntimeError("cloud down")

    monkeypatch.setattr(cloud, "summarize", _boom)
    client = _client(tmp_path, monkeypatch)
    files = {"file": ("a.docx", b"fake", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    r = client.post("/api/files", files=files, data={"allow_cloud": "1", "multimodal": "0", "ocr": "auto"})
    did = r.json()["document_id"]
    for _ in range(50):
        time.sleep(0.1)
        d = client.get(f"/api/files/{did}").json()
        if d["status"] == "done":
            break
    assert d["status"] == "done"
    assert d["summary"] is None
