"""分块策略路由器：auto / structural / recursive / sliding。"""
import re

from app.core.ingest.models import ParsedDocument, TextChunk
from app.core.tables import ParsedTable, table_summary

# 句末标点（recursive 第二层切分符）
_SENTENCE_END = "。！？；"


def chunk_document(
    doc: ParsedDocument,
    strategy: str = "auto",
    max_chars: int = 600,
    overlap: int = 60,
) -> list[TextChunk]:
    """把 ParsedDocument 按指定策略切成 TextChunk 列表（seq 为 0 起序号）。

    auto：headings 非空走 structural，否则走 recursive。
    semantic / proposition 本轮仅声明，抛 NotImplementedError。
    """
    if strategy == "auto":
        strategy = "structural" if doc.headings else "recursive"

    if strategy == "structural":
        return _chunk_structural(doc, max_chars, overlap)
    if strategy == "recursive":
        return _chunk_recursive(doc, max_chars)
    if strategy == "sliding":
        return _chunk_sliding(doc, max_chars, overlap)
    if strategy in ("semantic", "proposition"):
        raise NotImplementedError(f"strategy={strategy} 本轮仅声明，暂未实现")
    raise ValueError(f"未知分块策略: {strategy}")


def _concat_text(doc: ParsedDocument) -> str:
    """取全文文本：full_text 为空时回退到 blocks 拼接。"""
    if doc.full_text:
        return doc.full_text
    return "\n".join(b.text for b in doc.blocks if b.text)


def _chunk_structural(doc: ParsedDocument, max_chars: int, overlap: int) -> list[TextChunk]:
    """按 heading_path 分组为小节；整节 ≤ max_chars 一块，否则按段落切 + overlap。"""
    # 按出现顺序分组；heading_path=None 的块归入上一个已见小节（或顶层 None 小节）
    order: list[str | None] = []
    groups: dict[str | None, list[str]] = {}
    active: str | None = None
    for b in doc.blocks:
        if b.heading_path is not None:
            active = b.heading_path
        key = active
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(b.text)

    chunks: list[TextChunk] = []
    for key in order:
        section_text = "\n".join(groups[key])
        if not section_text:
            continue
        texts = [section_text] if len(section_text) <= max_chars else _split_structural(
            section_text, max_chars, overlap
        )
        for text in texts:
            chunks.append(
                TextChunk(text=text, heading_path=key, page=None, parent_text=section_text, seq=len(chunks))
            )
    return chunks


def _chunk_recursive(doc: ParsedDocument, max_chars: int) -> list[TextChunk]:
    """空行 → 句末标点 → 字符逐级切分（无重叠）。"""
    text = _concat_text(doc)
    texts = _split_recursive(text, max_chars)
    return [
        TextChunk(text=t, heading_path=None, page=None, parent_text=text, seq=i)
        for i, t in enumerate(texts)
    ]


def _chunk_sliding(doc: ParsedDocument, max_chars: int, overlap: int) -> list[TextChunk]:
    """定长 max_chars 窗口滑动，步长 = max_chars - overlap。"""
    text = _concat_text(doc)
    texts = _split_window(text, max_chars, overlap)
    return [
        TextChunk(text=t, heading_path=None, page=None, parent_text=text, seq=i)
        for i, t in enumerate(texts)
    ]


def _split_structural(text: str, max_chars: int, overlap: int) -> list[str]:
    """按段落边界切块，超长段落用滑动窗口切分（带 overlap）。"""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n"):
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_window(para, max_chars, overlap))
        elif not current:
            current = para
        elif len(current) + 1 + len(para) <= max_chars:
            current = current + "\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _split_recursive(text: str, max_chars: int) -> list[str]:
    """recursive 切分：空行 → 句末标点 → 字符，逐级切到 ≤ max_chars。"""
    result: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        if len(para) <= max_chars:
            result.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= max_chars:
                result.append(sent)
            else:
                result.extend(_split_chars(sent, max_chars))
    return result


def _split_sentences(text: str) -> list[str]:
    """按句末标点切分，标点保留在句尾。"""
    parts: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


def _split_chars(text: str, max_chars: int) -> list[str]:
    """按固定 max_chars 字符切分（无重叠）。"""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _split_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """定长滑动窗口切分，步长 = max_chars - overlap（相邻块重叠 overlap 字符）。"""
    if not text:
        return []
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def chunk_tables(tables: list[ParsedTable], max_chars: int = 600) -> list[TextChunk]:
    """表格按行组切成文本块：块首为表头行，块内累计长度 ≤ max_chars（至少含一行，单行超长不切）。

    无数据行的表跳过。seq 一律填 0——由 ingest 统一重编为正文块之后的连续序号。
    """
    chunks: list[TextChunk] = []
    for table in tables:
        if not table.rows:
            continue
        header_line = "表头：" + " | ".join(table.headers)
        lines: list[str] = []
        size = len(header_line)
        for row in table.rows:
            line = " | ".join(cell for cell in row if cell)
            if lines and size + 1 + len(line) > max_chars:
                chunks.append(_table_chunk(header_line, lines, table))
                lines, size = [], len(header_line)
            lines.append(line)
            size += 1 + len(line)
        if lines:
            chunks.append(_table_chunk(header_line, lines, table))
    return chunks


def _table_chunk(header_line: str, lines: list[str], table: ParsedTable) -> TextChunk:
    """表格块：表头行 + 一组数据行，parent_text 为整表摘要、heading_path 为所在小节。"""
    return TextChunk(
        text="\n".join([header_line, *lines]),
        heading_path=table.heading or "表格",
        page=None,
        parent_text=table_summary(table),
        seq=0,
    )
