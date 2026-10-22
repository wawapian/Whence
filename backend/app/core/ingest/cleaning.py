"""文本清洗：空白规范化、重复行去除、乱码检测。"""
import re
from collections import Counter


def clean_text(text: str) -> str:
    """把任意连续空白(空格/制表/回车/换行)折叠为单个空格，并去除首尾空白。"""
    return re.sub(r"\s+", " ", text).strip()


def strip_repeated_lines(lines: list[str]) -> list[str]:
    """去掉(stripped 后)出现 ≥2 次的行(页眉页脚)，保持剩余行原顺序。"""
    counts = Counter(line.strip() for line in lines)
    return [line for line in lines if counts[line.strip()] < 2]


def detect_garbled(text: str) -> bool:
    """U+FFFD 替换字符占比超过阈值(>0.1)则判定为乱码。"""
    if not text:
        return False
    return text.count("\ufffd") / len(text) > 0.1
