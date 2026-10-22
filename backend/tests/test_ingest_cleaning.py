from app.core.ingest import cleaning


def test_clean_text():
    assert cleaning.clean_text("a  b\r\n") == "a b"


def test_strip_repeated_lines():
    lines = ["页眉", "正文A", "页眉", "正文B", "页眉"]
    assert cleaning.strip_repeated_lines(lines) == ["正文A", "正文B"]


def test_detect_garbled():
    assert cleaning.detect_garbled("���") is True
    assert cleaning.detect_garbled("正常中文") is False
