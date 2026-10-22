"""RAW/FACT 读写 + 上传文件落盘。"""
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.memory import qa_history
from app.models.db import Fact, RawEvidence
from app.models.schemas import ExtractedFact


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_local(dt: datetime | None, tz=None) -> datetime | None:
    """把库里读回的时间转成本机时区（展示/导出用，带偏移输出）；tz 可注入（测试）。

    SQLite 读回的时间无时区（值按 UTC 存）——直接显示会差一个时区
    （用户实测：北京时间下午一点记录，界面上显示凌晨五点）。展示层一律经本函数转换。
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)  # tz=None → 本机时区


def insert_raw(
    session: Session,
    kind: str,
    content: str | None = None,
    file_path: str | None = None,
) -> RawEvidence:
    raw = RawEvidence(id=new_id(), kind=kind, content=content, file_path=file_path)
    session.add(raw)
    qa_history.bump(session)
    return raw


def insert_fact(session: Session, fact: ExtractedFact, raw_id: str, state: str = "confirmed") -> Fact:
    fields = {
        "place": fact.place,
        "amount": fact.amount,
        "currency": fact.currency,
        "people": fact.people,
        "items": fact.items,
        "datetime_hint": fact.datetime_hint,
        "is_reminder": fact.is_reminder,
    }
    f = Fact(
        id=new_id(),
        subject_key=fact.subject_key,
        type=fact.type,
        summary=fact.summary,
        fields=fields,
        confidence=fact.confidence,
        state=state,
        valid_from=utcnow(),
        valid_to=None,
        raw_id=raw_id,
    )
    session.add(f)
    qa_history.bump(session)
    return f


def classify_kind(content_type: str | None) -> str:
    if content_type and content_type.startswith("image/"):
        return "photo"
    return "document"


def save_upload(upload: UploadFile, data_dir: Path) -> str:
    """保存上传文件到 `data_dir/files/`，返回相对 data_dir 的路径。"""
    files_dir = data_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename or "").suffix
    dest = files_dir / f"{new_id()}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest.relative_to(data_dir))


def resolve_data_file(data_dir: Path, rel_path: str | None) -> Path:
    """库内相对路径 → data_dir 内的绝对路径；非法/越界/不存在 → `ValueError`。

    原件出口与**未来鉴权/数据隔离的唯一路径判定点**：空路径、路径穿越（`..`、绝对路径）、
    解析后落在 `data_dir` 之外、盘上不存在——一律拒绝（调用方转 404，绝不吐内容）。
    """
    if not rel_path or not str(rel_path).strip():
        raise ValueError("空文件路径")
    base = Path(data_dir).resolve()
    candidate = (base / str(rel_path)).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("路径越出 data_dir")
    if not candidate.is_file():
        raise ValueError("文件不存在")
    return candidate


def fact_search_text(session: Session, fact_id: str) -> str:
    fact = session.get(Fact, fact_id)
    if fact is None:
        return ""
    raw = session.get(RawEvidence, fact.raw_id) if fact.raw_id else None
    content = (raw.content or "") if raw else ""
    return f"{fact.summary or ''} {content[:2000]}".strip()
