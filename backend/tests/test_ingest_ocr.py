from app.core.ingest import ocr


def test_rapidocr_recognize(monkeypatch):
    class Fake:
        def __call__(self, img_bytes):
            return [[[0, 0, 1, 1], "38 元", 0.99]]

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", lambda: Fake())
    assert ocr.RapidOcrEngine().recognize(b"img") == "38 元"


def test_rapidocr_recognize_tuple_format(monkeypatch):
    """rapidocr_onnxruntime 2.x 返回 (items, elapse) 二元组，应取第一个元素拼接文本。"""
    class Fake:
        def __call__(self, img_bytes):
            return ([[[0, 0, 1, 1], "查重报告", 0.99]], [0.123])

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", lambda: Fake())
    assert ocr.RapidOcrEngine().recognize(b"img") == "查重报告"


def test_rapidocr_recognize_none_items_when_no_text(monkeypatch):
    """无文字的图片（风景照等）：rapidocr 2.x 返回 (None, elapse)——必须回空串而不是 TypeError。

    真实案例：草莓照片（12MB、无文字）→ 拿 None 迭代抛 TypeError → 整条照片解析判 error。
    """
    class Fake:
        def __call__(self, img_bytes):
            return (None, [0.1])

    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", lambda: Fake())
    assert ocr.RapidOcrEngine().recognize(b"img") == ""


def test_get_ocr_engine():
    assert ocr.get_ocr_engine("off") is None
    assert isinstance(ocr.get_ocr_engine("rapidocr"), ocr.RapidOcrEngine)


def test_parse_image_registers_self_as_image_ref(tmp_path, monkeypatch):
    """随手拍修复：图片解析结果包含照片本体（blob + OCR 文本 + 位置=整图）。"""
    from app.core.ingest import pipeline

    class FakeEngine:
        def recognize(self, data):
            return "38 元"

    monkeypatch.setattr(ocr, "get_ocr_engine", lambda name: FakeEngine())
    png = tmp_path / "bill.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    result = pipeline.ingest(png, ocr_name="rapidocr")
    assert result.parsed.full_text == "38 元"
    assert len(result.parsed.images) == 1
    ref = result.parsed.images[0]
    assert ref.blob == b"\x89PNG\r\n\x1a\nfake"
    assert ref.ocr_text == "38 元"
    assert ref.position == "整图"


def test_parse_image_with_ocr_off_still_registers_self(tmp_path, monkeypatch):
    """OCR 关闭只是没有文字，照片本体仍登记（供 caption / 看图可达）。"""
    from app.core.ingest import pipeline

    png = tmp_path / "photo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    result = pipeline.ingest(png, ocr_name="off")
    assert result.parsed.full_text == ""
    assert len(result.parsed.images) == 1
    assert result.parsed.images[0].ocr_text == ""
