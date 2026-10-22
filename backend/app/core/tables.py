"""Markdown 管道表格解析 + 原始网格通用归一化：供统计分析管线复用。"""
import re
from dataclasses import dataclass


@dataclass
class ParsedTable:
    headers: list[str]
    rows: list[list[str]]
    heading: str | None = None
    suspect: bool = False


def _split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown_table(text: str) -> ParsedTable | None:
    """解析 markdown 中第一个管道表格；无表格返回 None。"""
    # 函数内导入：ingest 包 __init__ 会导入 pipeline，而 pipeline 依赖本模块，
    # 模块级导入在「先导入 tables」时成环
    from app.core.ingest.markdown import is_table_separator

    lines = text.splitlines()
    n = len(lines)
    for i in range(n - 1):
        if lines[i].strip().startswith("|") and is_table_separator(lines[i + 1]):
            table_lines = []
            j = i
            while j < n and lines[j].strip().startswith("|"):
                table_lines.append(lines[j])
                j += 1
            headers = _split_cells(table_lines[0])
            width = len(headers)
            rows = []
            for raw in table_lines[2:]:  # 跳过表头与分隔行
                cells = _split_cells(raw)[:width]
                cells += [""] * (width - len(cells))
                rows.append(cells)
            return ParsedTable(headers=headers, rows=rows)
    return None


def table_summary(table: ParsedTable) -> str:
    """表格摘要（行列数 + 表头），供索引/回答提示使用。"""
    return f"表格（{len(table.rows)}行×{len(table.headers)}列：{'、'.join(table.headers)}）"


# 原始网格归一化（docx 直读 / PDF 坐标法共用）
_CJK_BLANK_RE = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")
_BLANK_RUN_RE = re.compile(r"\s+")
_GLUED_CELL_RE = re.compile(r"^\D+\s+\d+(?:\.\d+)?$")


def _clean_cell(text: str) -> str:
    """单元格清洗：合并格重复段落去重、CJK 字符间空白删除；
    数字/英文词间空格保留（`40\\n60` → `40 60`，绝不拼接改变语义）。"""
    parts = [p.strip() for p in text.split("\n")]
    parts = [p for p in parts if p]
    if parts and all(p == parts[0] for p in parts):
        parts = parts[:1]
    s = " ".join(parts)
    s = _CJK_BLANK_RE.sub("", s)
    return _BLANK_RUN_RE.sub(" ", s).strip()


def _split_subtables(rows: list[list[str]]) -> list[list[list[str]]]:
    """按重复出现的表头列把并排子表切成多段；只看表头行（第 0 行），无重复 → 单段。"""
    header = rows[0]
    cuts = [i for i, h in enumerate(header) if i > 0 and h and h == header[0]]
    bounds = [0, *cuts, len(header)]
    return [[row[start:end] for row in rows] for start, end in zip(bounds, bounds[1:])]


def _drop_empty_columns(block: list[list[str]]) -> list[list[str]]:
    """去掉表头与全部数据行都为空的全空列。"""
    headers, *rows = block
    cols = [j for j in range(len(headers)) if headers[j] or any(r[j] for r in rows)]
    return [[row[j] for j in cols] for row in block]


def _is_suspect(headers: list[str], rows: list[list[str]]) -> bool:
    """解析可疑标记：① 表头有空名或重名；② 含粘格（如 `通识必修 2.0`）的数据行 ≥2 行且占数据行 ≥10%。"""
    if any(not h for h in headers) or len(set(headers)) < len(headers):
        return True
    glued_rows = sum(1 for row in rows if any(_GLUED_CELL_RE.match(c) for c in row))
    return glued_rows >= 2 and glued_rows * 10 >= len(rows)


def split_title_row(grid: list[list[str]]) -> tuple[str | None, list[list[str]]]:
    """拆出表格上方的「标题行」：首行只有 1 个非空格、第二行 ≥2 个非空 → `(标题, 其余行)`。

    真实形态（会话 26 评测实测）：xlsx 里 A1 直接打字当标题（未合并），第二行才是真表头
    ——不拆的话标题会被 `normalize_grid` 当表头，列名全丢（成绩表按列筛选统计做不了）。
    已知代价：真表头恰好只填一格、且下一行多格非空的表会被误当标题——少见，且这种表
    本就带空表头（`_is_suspect` 会标记），标题也进 heading 不丢。
    """
    if len(grid) < 2:
        return None, grid
    first_cells = [c for c in grid[0] if c and c.strip()]
    second_cells = [c for c in grid[1] if c and c.strip()]
    if len(first_cells) == 1 and len(second_cells) >= 2:
        return first_cells[0].strip(), grid[1:]
    return None, grid


def normalize_grid(grid: list[list[str]]) -> list[ParsedTable]:
    """原始表格网格（第 0 行为表头）→ 干净结构化表；空网格/只有表头/清洗后无数据行 → `[]`。

    前置：输入须为矩形网格、单元格为 str（补齐等宽与 None 转换由来源方负责）。
    """
    cleaned = [[_clean_cell(cell) for cell in row] for row in grid]
    cleaned = [row for row in cleaned if any(row)]
    if len(cleaned) < 2:
        return []
    tables = []
    for block in _split_subtables(cleaned):
        data_rows = [row for row in block[1:] if any(row)]  # 段内再过滤：该行可能只在别的子表段非空
        headers, *rows = _drop_empty_columns([block[0], *data_rows])
        if not headers or not rows:
            continue
        tables.append(ParsedTable(headers=headers, rows=rows,
                                  suspect=_is_suspect(headers, rows)))
    return tables
