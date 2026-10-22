"""敏感值本地规则检测 + 打码（敏感值只本地存、密文落库，原文永不出门）。"""
import re
import uuid
from collections.abc import Iterable
from pathlib import Path

from cryptography.fernet import Fernet
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models.db import Secret

# 先匹配更具体的身份证（18 位），再银行卡（13~19 位）、手机号（11 位）、密码。
# 边界用「数字视界」而非 \b：汉字算 \w，「手机号是139…」「卡号6214…已核对」这类紧贴写法
# 用 \b 会失配、漏打码（会话 26 评测实测；lookaround 保证汉字相邻能中、长数字串不半截匹配）。
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_CARD_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_PASSWORD_RE = re.compile(r"(密码\s*[是为：:]\s*)([A-Za-z0-9]{4,20})")


def mask_text(text: str, session: Session, fernet: Fernet) -> tuple[str, dict[str, str]]:
    """检出敏感值 → 替换为占位符 → 密文写 secrets 表。

    返回 `(打码后文本, {占位符: 真实值})`。同一次调用内同一真实值复用同一占位符；
    占位符跨调用全局唯一（带随机短串）——占位符是 secrets 主键，若每次调用从 1
    重新数，同库第二条含敏感值的记录会撞主键（IntegrityError，整条请求 500）。
    """
    mapping: dict[str, str] = {}
    seen: dict[str, str] = {}

    def mask(value: str) -> str:
        if value in seen:
            return seen[value]
        placeholder = f"[[SECRET_{uuid.uuid4().hex[:8]}]]"
        seen[value] = placeholder
        mapping[placeholder] = value
        session.add(Secret(placeholder=placeholder, value=fernet.encrypt(value.encode()).decode()))
        return placeholder

    text = _ID_RE.sub(lambda m: mask(m.group(0)), text)
    text = _CARD_RE.sub(lambda m: mask(m.group(0)), text)
    text = _PHONE_RE.sub(lambda m: mask(m.group(0)), text)
    text = _PASSWORD_RE.sub(lambda m: m.group(1) + mask(m.group(2)), text)

    return text, mapping


_OUTBOUND = "[[SECRET]]"


def mask_outbound(text: str) -> str:
    """出库打码（查询/工具出库用）：与 mask_text 同一套正则，替换为 `[[SECRET]]`、不落库。

    查询期没有回填/清账需求，故不建 secrets 行；占位不带随机串——它只是「此处已打码」标记。
    空串/None 原样返回。
    """
    if not text:
        return text
    text = _ID_RE.sub(_OUTBOUND, text)
    text = _CARD_RE.sub(_OUTBOUND, text)
    text = _PHONE_RE.sub(_OUTBOUND, text)
    text = _PASSWORD_RE.sub(lambda m: m.group(1) + _OUTBOUND, text)
    return text


def link_secrets_to_fact(session: Session, placeholders: Iterable[str], fact_id: str) -> None:
    """把本次打码写入的 secrets 行关联到 fact_id。

    打码发生在建 fact 之前，行插入时 `fact_id` 只能为空；fact 建好后由调用方补关联，
    彻底删除时 `DELETE FROM secrets WHERE fact_id = …` 才清得到加密副本。
    """
    placeholders = list(placeholders)
    if not placeholders:
        return
    session.execute(
        update(Secret).where(Secret.placeholder.in_(placeholders)).values(fact_id=fact_id)
    )


def load_fernet(data_dir: Path) -> Fernet:
    """从 `data_dir/secret.key` 加载 Fernet 密钥，不存在则生成。"""
    key_path = data_dir / "secret.key"
    if key_path.exists():
        key = key_path.read_bytes()
    else:
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key)
    return Fernet(key)
