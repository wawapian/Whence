"""SQLite FTS5 全文检索封装（trigram 分词，支持中文子串匹配）。"""
import logging
import re

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def fts_query(q: str) -> str:
    """把用户输入转成安全的 FTS5 MATCH 查询：逐词加引号、OR 连接。

    - 每个空白分隔的原词**整体加引号**：引号内 FTS5 语法字符（, : ( ) " - *
      等）全部被中和，且保留 node.js、C++ 这类词内符号，原样可命中；
    - 同时把词内按文字片段拆出的各部分也加引号放进 OR：应对「，」等无空格
      粘连的中文词（如 message_runtime,配置 → 配置 也能独立命中）；
    - 词间 OR 连接保证长句/自然语言问题也能命中（AND 要求全部词都出现）。
    """
    terms: list[str] = []
    for tok in q.split():
        quoted = '"' + tok.replace('"', '""') + '"'
        terms.append(quoted)
        for part in re.findall(r"\w+", tok):
            if part != tok:
                terms.append('"' + part + '"')
    if not terms:
        return '""'
    return " OR ".join(dict.fromkeys(terms))


def index_fact(session: Session, fact_id: str, content: str):
    session.execute(
        text("INSERT INTO memory_fts(content, fact_id) VALUES (:content, :fact_id)"),
        {"content": content, "fact_id": fact_id},
    )


def search(session: Session, q: str, k: int = 10) -> list[str]:
    try:
        # memory_fts 保持原名不别名：FTS5 的 MATCH 语法要求左操作数是 fts 表本身。
        # LEFT JOIN 只排除「记录已进回收站」的 fact，不改变其余检索语义（无 fact 的
        # 孤儿索引行照旧命中）。
        rows = session.execute(
            text(
                "SELECT memory_fts.fact_id FROM memory_fts "
                "LEFT JOIN facts f ON f.id = memory_fts.fact_id "
                "LEFT JOIN raw_evidence r ON r.id = f.raw_id "
                "WHERE memory_fts MATCH :q AND r.deleted_at IS NULL "
                "ORDER BY rank LIMIT :k"
            ),
            {"q": fts_query(q), "k": k},
        ).fetchall()
    except Exception:
        # FTS 失败只降级为无关键词命中，不打断向量检索主链路
        logger.warning("memory_fts 查询失败（已降级为纯向量检索）", exc_info=True)
        return []
    return [r[0] for r in rows]
