"""文件格式检测：magic bytes 优先，扩展名兜底。"""
from pathlib import Path

import filetype

_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic",
}
_WORD_EXTS = {".docx", ".doc"}
_TEXT_EXTS = {".txt", ".md", ".markdown", ".text"}


def _from_magic(path: Path) -> str | None:
    """用 filetype 检测真实类型，映射到五类；识别不出返回 None。"""
    guessed = filetype.guess(str(path))
    if guessed is None:
        return None
    ext = (guessed.extension or "").lower()
    mime = (guessed.mime or "").lower()
    if ext in _WORD_EXTS:
        return "word"
    if ext == "pdf":
        return "pdf"
    if mime.startswith("image/") or ext in _IMAGE_EXTS:
        return "image"
    if mime.startswith("text/") or ext in _TEXT_EXTS:
        return "text"
    return None


def _from_suffix(path: Path) -> str:
    """按扩展名兜底。"""
    ext = path.suffix.lower()
    if ext in _WORD_EXTS:
        return "word"
    if ext == ".pdf":
        return "pdf"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _TEXT_EXTS:
        return "text"
    return "other"


def detect_format(path: Path) -> str:
    """返回 "word" | "pdf" | "image" | "text" | "other"，magic bytes 优先。"""
    return _from_magic(path) or _from_suffix(path)
