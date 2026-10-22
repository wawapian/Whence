"""Excel 解析：Docling MsExcel 后端直读（每 sheet 一页、自动表格检测、合并格处理）。

- `.xlsx`/`.xlsm` 归本模块；老 `.xls` 不收（归未知格式如实标注 `unsupported`，需要再说）。
- 结构化表：docling 逻辑单元格按起始坐标摊平成网格 → `normalize_grid`
  （与 docx/PDF 同一套归一化：合并格值置首个列位、其余列位置空）。
- v1 只取文本 + 表格；内嵌图表/图片不处理（docling 默认也不渲染图表图）。
- 公式只认保存时的缓存值（openpyxl 生成、Excel 没存过的表公式格为空）——与其他解析器同类边界。
"""
from pathlib import Path

from app.core.ingest.markdown import parse_markdown
from app.core.ingest.models import ParsedDocument
from app.core.tables import normalize_grid, split_title_row

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}

# 惰性导入：首次调用时才加载 docling（重）；转换器实例复用
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.backend_options import MsExcelBackendOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, ExcelFormatOption

        _converter = DocumentConverter(
            format_options={
                # 单格「表」（如某 sheet 只有一行说明）按文本处理，避免 1×1 假表混进表格清单
                InputFormat.XLSX: ExcelFormatOption(
                    backend_options=MsExcelBackendOptions(treat_singleton_as_text=True)
                )
            }
        )
    return _converter


def _grid_from_table(table) -> list[list[str]]:
    """docling 表格 → 等宽字符串网格。

    - 横向合并：值置首个列位，其余列位置空（同 docx 规则；避免触发并排子表切分）；
    - 纵向合并（如企业名/简介跨多行）：**值向下填充到覆盖的每一行**——否则续行的
      岗位/薪资会失去归属。真实案例：岗位信息表一家公司 6 行岗位（公司名合并格只出现
      一次），不填充则「这家公司有哪些岗位」只能查到第一条。
    """
    n_rows, n_cols = table.data.num_rows, table.data.num_cols
    grid = [[""] * n_cols for _ in range(n_rows)]
    for cell in table.data.table_cells:
        r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            continue
        row_span = getattr(cell, "row_span", 1) or 1
        for rr in range(r, min(r + row_span, n_rows)):
            grid[rr][c] = cell.text
    return grid


def parse_excel_docling(path: Path) -> ParsedDocument:
    """Docling 转换 Excel → markdown（blocks/headings/full_text）+ 结构化表（normalize_grid）。"""
    doc = _get_converter().convert(str(path)).document
    parsed = parse_markdown(
        doc.export_to_markdown(), kind="excel", engine_used="docling-xlsx"
    )
    for table in doc.tables:
        title, grid = split_title_row(_grid_from_table(table))  # 未合并的标题行不吞真表头
        for parsed_table in normalize_grid(grid):
            if title and not parsed_table.heading:
                parsed_table.heading = title
            parsed.tables.append(parsed_table)
    return parsed
