"""Markdown → blocks/headings 解析：供 pandoc（docx）与 docling（pdf）复用。"""
from app.core.ingest.models import Block, Heading, ParsedDocument


def is_table_separator(line: str) -> bool:
    """判断是否为 markdown 表格分隔行（如 |---|---|、|:---:|---|）。"""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(set(c) <= {"-", ":", " "} and "-" in c for c in cells)


def parse_markdown(
    markdown_text: str,
    *,
    kind: str,
    engine_used: str,
    ocr_used: bool = False,
    needs_ocr: bool | None = None,
) -> ParsedDocument:
    """将 markdown 解析为标题/表格/段落 blocks（kind/engine_used 由调用方指定）。"""
    headings: list[Heading] = []
    blocks: list[Block] = []
    title: str | None = None
    heading_stack: list[str] = []

    def _heading_path() -> str | None:
        return " > ".join(heading_stack) if heading_stack else None

    lines = markdown_text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()

        # 标题（#、##、…）
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if text:
                if title is None and level == 1:
                    title = text
                headings.append(Heading(level=level, text=text))
                del heading_stack[level - 1:]
                heading_stack.append(text)
                blocks.append(
                    Block(type="heading", text=text, heading_path=_heading_path(), page=None)
                )
            i += 1
            continue

        # 表格：`| a | b |` 后跟 `|---|` 分隔行
        if stripped.startswith("|") and i + 1 < n and is_table_separator(lines[i + 1]):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(
                Block(type="table", text="\n".join(table_lines), heading_path=_heading_path(), page=None)
            )
            continue

        # 段落：连续非空行合并为一段
        if stripped:
            para_lines = []
            while i < n and lines[i].strip() and not lines[i].strip().startswith("#"):
                para_lines.append(lines[i].strip())
                i += 1
            blocks.append(
                Block(type="paragraph", text=" ".join(para_lines), heading_path=_heading_path(), page=None)
            )
        else:
            i += 1

    return ParsedDocument(
        kind=kind,
        title=title,
        full_text=markdown_text,
        headings=headings,
        blocks=blocks,
        engine_used=engine_used,
        ocr_used=ocr_used,
        needs_ocr=needs_ocr,
    )
