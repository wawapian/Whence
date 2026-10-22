"""Ingestion 管道编排：识别报告 → 选链解析 → 结构化表格 → 清洗 → 分块。"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.ingest import chunking, cleaning
from app.core.ingest.markdown import is_table_separator
from app.core.ingest.models import ParsedDocument, TextChunk
from app.core.ingest.processor import IngestContext
from app.core.ingest.processors import build_chain
from app.core.ingest.report import build_report
from app.core.tables import ParsedTable, parse_markdown_table, table_summary

logger = logging.getLogger(__name__)

# 表块文本超过该长度视为长表；表块一律替换为一行摘要后远低于该值（防巨型表被分块切碎）
TABLE_SUMMARY_THRESHOLD = 2000

# 表格摘要行（`table_summary` 的产出形态）；表格类文档的正文可能只剩这些行
_TABLE_SUMMARY_LINE_RE = re.compile(r"^表格（\d+行×\d+列：.*）$")


def is_table_only_text(text: str) -> bool:
    """正文是否只剩表格摘要行（表格类文档）——调用方据此跳过云总结。

    这类文档的云总结输入是退化文本（只有表名与行列数），模型会回抱怨/请求
    （「请提供文本内容…」），入库即毒化材料索引（会话 26 评测实测）。空文本不算
    「只剩表格」（扫描件/无正文文档另有无正文兜底）。
    """
    if not text.strip():
        return False
    return all(not line.strip() or _TABLE_SUMMARY_LINE_RE.match(line.strip())
               for line in text.splitlines())


@dataclass
class IngestResult:
    """ingest 的产出：解析结果 + 分块列表 + 结构化表格。"""

    parsed: ParsedDocument
    chunks: list[TextChunk]
    tables: list[ParsedTable] = field(default_factory=list)


def _pipe_table_texts(text: str) -> list[str]:
    """`text` 中出现的 markdown 管道表原文（按出现顺序），保留原文片段以便在原地替换。"""
    lines = text.splitlines(keepends=True)
    tables: list[str] = []
    i, n = 0, len(lines)
    while i + 1 < n:
        if not (lines[i].strip().startswith("|") and is_table_separator(lines[i + 1])):
            i += 1
            continue
        j = i
        while j < n and lines[j].strip().startswith("|"):
            j += 1
        tables.append("".join(lines[i:j]).rstrip("\r\n"))
        i = j
    return tables


def structify_tables(parsed: ParsedDocument) -> list[ParsedTable]:
    """结构化表优先：`parsed.tables` 非空直接采用；为空才从表块兜底解析（heading=块的 heading_path）。

    与结构化表的对应关系只在**确有依据**时按出现顺序配对（见 `paired`）：此时表块文本替换为
    摘要、`full_text` 中的管道表原文同步替换——表格原文不进检索索引（structural 与 recursive
    分块路径一致，后者用 full_text）。
    对不上时（并排子表切分、零产出表的兜底原文等）**两条路径都保留原文**：不按序错配，也不用
    「（表格见结构化数据）」占位——占位句会失真（宣称「见结构化数据」，但该表可能零产出、
    本无结构化数据），且会把零产出表的兜底原文从索引里抹掉。
    """
    result = list(parsed.tables)
    if not result:
        for block in parsed.blocks:
            if block.type != "table":
                continue
            parsed_table = parse_markdown_table(block.text)
            if parsed_table is None:
                continue
            parsed_table.heading = block.heading_path
            result.append(parsed_table)
    if not result:
        return result

    table_blocks = [b for b in parsed.blocks if b.type == "table"]
    raw_blocks = [b for b in table_blocks if parse_markdown_table(b.text) is not None]
    raw_texts = _pipe_table_texts(parsed.full_text)
    # 一一对应确有依据：① 全部表块都已由来源换成摘要（pandoc/docx/docling），full_text 里残留的
    # 管道表与结构化表数量相等；② 全部表块仍是管道表原文且与结构化表数量相等。
    # 形态/数量对不上（并排子表切分、零产出表的兜底原文等）→ 不按序错配，一律保留原文
    paired = (not raw_blocks and len(raw_texts) == len(result)) or (
        len(raw_blocks) == len(table_blocks) == len(result)
    )

    if not paired:
        # 未配对：两条替换都跳过——占位句会失真（宣称「见结构化数据」，但该表无结构化数据），
        # 且会把零产出表的兜底原文从索引里抹掉；宁可保留原文，也不装作读全了
        logger.debug(
            "表格与结构化表未配对（结构化 %d / 表块 %d / 管道表 %d）：保留原文，不做替换",
            len(result), len(table_blocks), len(raw_texts),
        )
        return result

    for i, raw_table in enumerate(raw_texts):
        summary = table_summary(result[i]) if i < len(result) else "（表格见结构化数据）"
        parsed.full_text = parsed.full_text.replace(raw_table, summary)

    for i, block in enumerate(raw_blocks):
        block.text = table_summary(result[i])
    return result


def ingest(
    path: Path,
    *,
    ocr_name: str = "rapidocr",
    chunk_strategy: str = "auto",
    chunk_max: int = 600,
    chunk_overlap: int = 60,
    advanced_engine: str = "off",
) -> IngestResult:
    """build_report → build_chain 选链 → 单元解析 → 清洗 → 分块；表格行组块追加在正文块之后（seq 连续）。

    解析异常降级为错误结果，文本编码失败则上抛。
    """
    path = Path(path)
    report = build_report(path)
    chain = build_chain(report)
    try:
        parsed = chain[0].run(
            path, report, IngestContext(ocr_name=ocr_name, advanced_engine=advanced_engine)
        )
    except UnicodeDecodeError:
        raise  # 文本编码无法识别：上抛让上层把文档标记为 failed
    except Exception:
        return IngestResult(
            parsed=ParsedDocument(
                kind=report.kind,
                title=None,
                full_text="",
                headings=[],
                blocks=[],
                engine_used="error",
                ocr_used=False,
                needs_ocr=None,
            ),
            chunks=[],
        )

    # 必须先于 clean_text：管道表扫描依赖换行，clean_text 会把 full_text 折叠成单行
    parsed_tables = structify_tables(parsed)
    parsed.full_text = cleaning.clean_text(parsed.full_text)
    chunks = chunking.chunk_document(
        parsed, strategy=chunk_strategy, max_chars=chunk_max, overlap=chunk_overlap
    )
    # 表格行组块追加在正文块之后：process_document 既有循环自动入库 + 建 FTS/向量索引
    table_chunks = chunking.chunk_tables(parsed_tables)
    for i, c in enumerate(table_chunks):
        c.seq = len(chunks) + i
    chunks += table_chunks
    return IngestResult(parsed=parsed, chunks=chunks, tables=parsed_tables)
