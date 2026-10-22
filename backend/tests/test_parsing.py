"""记录附件回填解析（parsing.parse_document）：中文 PDF 不走样 + 乱码质量门。

背景（会话 26 评测排查）：回填走 pypdf 时，无 ToUnicode 映射的中文 PDF 会被吐成
内容流乱码（实测 7588 字符），乱码原样进记录内容/事实索引/材料摘要——污染检索。
"""

import re

from app.core import parsing


def _make_cjk_pdf(tmp_path):
    """fitz + china-s 生成中文 PDF（不带 ToUnicode 映射——真实世界常见形态）。"""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "基于多模态检索的个人知识管理方法研究",
                     fontname="china-s", fontsize=14)
    path = tmp_path / "paper.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_parse_pdf_cjk_text_not_mojibake(tmp_path):
    text = parsing.parse_document(_make_cjk_pdf(tmp_path))
    assert "个人知识管理" in text
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]", text)


def test_garbage_output_is_dropped(tmp_path, monkeypatch):
    """解析器吐乱码（控制字符占比高）→ 视为解析不出返回空，不把乱码写进记录与索引。"""
    monkeypatch.setattr(
        parsing, "_pdf_text",
        lambda p: 'WúN\x8eY\x1aj!`\x01hÀ}"v\x84N*Nºwå\x8bÆ{¡t\x06e¹lÕx\x14zvÿ\x08{,')
    assert parsing.parse_document(tmp_path / "x.pdf") == ""


def test_normal_text_passes_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(parsing, "_pdf_text", lambda p: "体检报告 存于书房 2026-10-14")
    assert parsing.parse_document(tmp_path / "x.pdf") == "体检报告 存于书房 2026-10-14"
