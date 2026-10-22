"""PDF 坐标法表格提取：PyMuPDF 的矢量线检测 + 文字坐标直读 + **表样自检**，不猜、不走视觉模型。

表样自检（2026-10-13）：检测器的原始输出会把双栏参考文献列表、矢量结构图框误判成"表"
（真实案例：一篇 26 页综述检出 34 张"表"，其中大部分是这两类，且被判表的区域会从正文
剔除 —— 17% 正文因此不在解析全文里）。自检只放行"像真表"的网格；**不像表的一律不产出、
也不剔除该区域**（正文保住）。判据保守：只打"段落感/图内含长文"的表，短格子真表不受影响。
"""
import logging
from pathlib import Path

import pymupdf

from app.core.tables import ParsedTable, normalize_grid

logger = logging.getLogger(__name__)

# 表样自检阈值（保守；不要为个别案例随意放宽——放宽会连带正文剔除）
_HEADER_AVG_MAX = 24   # 表头单元格平均长度上限（更长 = 像正文句子/图内标注）
_HEADER_CELL_MAX = 60  # 表头单格长度上限
_BODY_MEDIAN_MAX = 30  # 全表非空单元格长度中位数上限（段落感的"表" = 正文被切进来了）


def _is_likely_table(grid: list[list[str]]) -> bool:
    """表样自检：网格像不像真表（至少 2×2；表头短而像表头；单元格非段落状）。"""
    rows = [[str(c).strip() for c in row] for row in grid]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return False  # 单行/单列"表"（文献列表碎片等）没有统计价值
    if max(len(r) for r in rows) < 2:
        return False
    header = rows[0]
    if not header:
        return False
    if sum(len(c) for c in header) / len(header) > _HEADER_AVG_MAX:
        return False
    if any(len(c) > _HEADER_CELL_MAX for c in header):
        return False
    lens = sorted(len(c) for r in rows for c in r if c)
    if not lens:
        return False
    if lens[len(lens) // 2] > _BODY_MEDIAN_MAX:
        return False
    return True


def _page_tables(page, page_no: int) -> list:
    """单页表格检测：该页异常仅 warning 并跳过，不影响其余页。"""
    try:
        return page.find_tables().tables
    except Exception:
        logger.warning("PDF 表格检测失败（第 %d 页，已跳过）", page_no + 1, exc_info=True)
        return []


def _grid_of(table) -> list[list[str]]:
    """Table.extract() → 矩形 str 网格（None 转 ""、短行补齐），满足 normalize_grid 前置。"""
    raw = table.extract()
    grid = [[cell or "" for cell in row] for row in raw]
    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid]


def extract_pdf_tables(path: Path) -> list[ParsedTable]:
    """PDF 内嵌表格（第 0 行为表头）→ 结构化表列表；打不开文件/无表/单表异常 → 跳过（不抛）。

    表样自检不过的"表"不产出（与 `table_regions` 同一判据：该区域也会被保留在正文里）。
    """
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        logger.warning("PDF 表格提取：打不开文件 %s", path, exc_info=True)
        return []
    tables: list[ParsedTable] = []
    skipped = 0
    try:
        for page_no, page in enumerate(doc):
            for t in _page_tables(page, page_no):
                try:
                    grid = _grid_of(t)
                    if not _is_likely_table(grid):
                        skipped += 1
                        continue
                    tables.extend(normalize_grid(grid))
                except Exception:
                    logger.warning("PDF 表格提取：第 %d 页某表解析失败（已跳过）", page_no + 1, exc_info=True)
        if skipped:
            logger.info("PDF 表格自检：%s 未过检 %d 张（不产出、不剔正文）", path.name, skipped)
    finally:
        doc.close()
    return tables


def table_regions(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """各页**过检**表格的 bbox：页序号（0 起）→ [bbox, ...]；打不开文件 → {}（不抛）。

    与 `extract_pdf_tables` 用同一套表样自检——未过检的区域不进剔除名单，文字留在正文里。
    """
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        logger.warning("PDF 表格区域：打不开文件 %s", path, exc_info=True)
        return {}
    regions: dict[int, list[tuple[float, float, float, float]]] = {}
    try:
        for page_no, page in enumerate(doc):
            boxes = []
            for t in _page_tables(page, page_no):
                try:
                    if not _is_likely_table(_grid_of(t)):
                        continue
                    boxes.append(tuple(t.bbox))
                except Exception:
                    logger.warning("PDF 表格区域：第 %d 页某表 bbox 读取失败（已跳过）", page_no + 1, exc_info=True)
            if boxes:
                regions[page_no] = boxes
    finally:
        doc.close()
    return regions
