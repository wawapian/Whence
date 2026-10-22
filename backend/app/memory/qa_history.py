"""提问历史（`qa_history` 表）与记忆库版本号（`store_meta`）读写原语。

本模块是叶子模块（只依赖 `app.models.db`）：`app.memory.store` / `documents` /
`app.core.trash` 的写入路径都要调用 `bump()`，反向依赖它们会成环，故不 import 它们。

版本号语义：**只有记忆库写入才 +1**（记录/文档等落库、删除/恢复/清空）；
提问、落历史行、缓存命中都不 +1。会话原语均不 commit，由调用方提交。
"""
import json
import uuid
from datetime import timedelta

import numpy as np
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from app.models.db import QaHistory, StoreMeta, utcnow

_VERSION_KEY = "store_version"


def cacheable_answer(answer: str, execution: dict | None = None) -> bool:
    """Failure history remains available but is never a successful answer cache.

    Text guards also exclude historical rows created before structured status existed.
    """
    execution = execution or {}
    if execution.get("status", "ok") != "ok":
        return False
    if not answer or not answer.strip():
        return False
    if answer.strip() in {"没有找到相关记录", "根据已有内容无法回答该问题。"}:
        return False
    if any(marker in answer for marker in (
        "（回答合成失败，", "全文总结未完成", "全文总结的最终汇总未完成", "（全文过长，已截断", "[信息不足]",
    )):
        return False
    reading = execution.get("reading")
    if reading and (not reading.get("complete") or execution.get("summary_version") != 1):
        return False
    return True


def record(
    session: Session,
    *,
    question: str,
    answer: str,
    sources: list,
    documents: list,
    question_vec: list[float],
    resolved_question: str | None = None,
    from_cache: bool = False,
    execution: dict | None = None,
) -> QaHistory:
    """落一行提问历史，返回已 add 的对象（不 commit，由调用方提交）。

    `sources`/`documents` 存 JSON 字符串（须为可序列化结构，pydantic 对象先 model_dump()）；
    `store_version` 取写入时的当前版本号——缓存有效性 = 行版本 == 当前版本。
    """
    row = QaHistory(
        id=uuid.uuid4().hex,  # uuid4 hex，同 store.new_id
        question=question,
        resolved_question=resolved_question,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False),
        documents=json.dumps(documents, ensure_ascii=False),
        question_vec=pack_vec(question_vec),
        store_version=current_version(session),
        from_cache=int(from_cache),
        execution=json.dumps(execution or {}, ensure_ascii=False),
    )
    session.add(row)
    return row


def list_rows(session: Session, *, limit: int = 50, offset: int = 0) -> list[QaHistory]:
    """按 `created_at` 倒序取历史行（新的在前）。

    同微秒写入的行（本机时钟粒度 1µs，连续插入可撞）用隐式 rowid 兜底——SQLite 普通表的
    rowid 随插入递增且已存行稳定，仅作同时间戳的次序键：「后写的在前」不随查询计划漂移。
    """
    return list(
        session.scalars(
            select(QaHistory)
            .order_by(QaHistory.created_at.desc(), text("rowid DESC"))
            .limit(limit)
            .offset(offset)
        )
    )


def clear_all(session: Session) -> int:
    """清空全部提问历史，返回删除条数（不 commit，由调用方提交）。"""
    return session.execute(delete(QaHistory)).rowcount


def lookup(
    session: Session,
    vec: list[float],
    *,
    threshold: float,
    top_k: int,
    ttl_hours: int,
    require_full_summary: bool = False,
    required_document_id: str | None = None,
) -> QaHistory | None:
    """缓存查找：有效行（版本一致 ∧ TTL 内）中取最近 `top_k` 行，余弦最高者 ≥ 阈值才返回。

    无候选 / 低于阈值 / 版本不符 → None（调用方静默走管线，用户无感）。
    TTL 判定走 ORM datetime 绑定（同 `core.trash.sweep_expired`）：`created_at` 读回是
    naive-UTC，同格式字符串可比——勿手拼 isoformat。本地嵌入已归一化，点积即余弦；
    `unpack_vec` 只读视图仅参与点积、不改。
    """
    cutoff = utcnow() - timedelta(hours=ttl_hours)
    rows = session.scalars(
        select(QaHistory)
        .where(
            QaHistory.store_version == current_version(session),
            QaHistory.created_at >= cutoff,
        )
        .order_by(QaHistory.created_at.desc(), text("rowid DESC"))
        .limit(top_k)
    ).all()

    query = np.asarray(vec, dtype=np.float32)
    best, best_score = None, float("-inf")
    for row in rows:
        try:
            execution = json.loads(row.execution or "{}")
        except (TypeError, ValueError):
            continue
        if not cacheable_answer(row.answer, execution):
            continue
        if require_full_summary and not (execution.get("summary_version") == 1 and
                                        execution.get("reading", {}).get("complete")):
            continue
        if required_document_id and execution.get("reading", {}).get("document_id") != required_document_id:
            continue
        score = float(np.dot(unpack_vec(row.question_vec), query))
        if score > best_score:
            best, best_score = row, score
    if best is None or best_score < threshold:
        return None
    return best


def current_version(session: Session) -> int:
    """当前记忆库版本号（无记录 = 初值 0）。

    SQL 直读而非 ORM 取对象：绕开身份映射缓存，`bump()` 的 UPSERT 立即可见。
    """
    value = session.execute(
        text("SELECT value FROM store_meta WHERE key = :k"), {"k": _VERSION_KEY}
    ).scalar()
    return int(value) if value is not None else 0


def bump(session: Session) -> None:
    """记忆库版本号 +1（记录/文档等落库、删除/恢复/清空时调用；不 commit）。

    单条 UPSERT 语句，而非 ORM 读改写：后台车道多线程并发写库时读改写会丢更新——
    丢更新 = 旧缓存行保持「有效」，端出过期答案。
    """
    session.execute(
        text(
            "INSERT INTO store_meta(key, value) VALUES(:k, '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(store_meta.value AS INTEGER) + 1"
        ),
        {"k": _VERSION_KEY},
    )


def pack_vec(vec: list[float]) -> bytes:
    """问题向量 → float32 BLOB（本地嵌入已归一化，比对时点积即余弦）。"""
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vec(blob: bytes) -> np.ndarray:
    """BLOB → float32 向量（`np.frombuffer` 只读视图；不复制）。"""
    return np.frombuffer(blob, dtype=np.float32)
