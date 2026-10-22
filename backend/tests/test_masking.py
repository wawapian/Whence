from cryptography.fernet import Fernet
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.masking import link_secrets_to_fact, load_fernet, mask_outbound, mask_text
from app.models.db import Secret, init_db


def _mk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/mem.db")
    init_db(engine)
    return Session(engine), Fernet(Fernet.generate_key())


def test_mask_password(tmp_path):
    session, fernet = _mk(tmp_path)
    masked, mapping = mask_text("我的银行卡密码是 123456", session, fernet)
    assert "123456" not in masked
    assert "[[" in masked
    assert list(mapping.values()) == ["123456"]


def test_mask_id_and_card(tmp_path):
    session, fernet = _mk(tmp_path)
    masked, _ = mask_text("身份证 110101199001010000 卡号 6222020000000000", session, fernet)
    assert "110101199001010000" not in masked
    assert "6222020000000000" not in masked


def test_no_sensitive_untouched(tmp_path):
    session, fernet = _mk(tmp_path)
    masked, mapping = mask_text("明天下午三点去医院复诊", session, fernet)
    assert masked == "明天下午三点去医院复诊"
    assert mapping == {}


def test_secret_stored_encrypted(tmp_path):
    session, fernet = _mk(tmp_path)
    mask_text("我的银行卡密码是 123456", session, fernet)
    session.commit()
    secret = session.scalars(select(Secret)).one()
    assert secret.value != "123456"  # 落库的是密文
    assert fernet.decrypt(secret.value.encode()).decode() == "123456"


def test_load_fernet_reuses_key(tmp_path):
    f1 = load_fernet(tmp_path)
    f2 = load_fernet(tmp_path)
    assert (tmp_path / "secret.key").exists()
    token = f1.encrypt(b"x")
    assert f2.decrypt(token) == b"x"


def test_placeholders_unique_across_calls(tmp_path):
    """回归：同库第二次打码不得撞 `secrets.placeholder` 主键。

    修复前占位符每次调用都从 `[[SECRET_1]]` 重新数 → 第二条含敏感值的记录
    `IntegrityError`（整条请求 500）。修复后占位符跨调用唯一。
    """
    session, fernet = _mk(tmp_path)
    _, map1 = mask_text("我的手机号 13800000001", session, fernet)
    session.commit()
    _, map2 = mask_text("备用手机号 13900000002", session, fernet)
    session.commit()  # 修复前这里抛 IntegrityError

    assert set(map1) != set(map2)
    assert session.query(Secret).count() == 2


def test_link_secrets_to_fact_fills_fact_id(tmp_path):
    """打码建行时 fact 还不存在（fact_id 为空），link 后补上关联——purge 才清得到。"""
    session, fernet = _mk(tmp_path)
    _, mapping = mask_text("我的银行卡密码是 123456", session, fernet)
    session.commit()
    assert session.scalars(select(Secret)).one().fact_id is None

    link_secrets_to_fact(session, mapping, "fact-1")
    session.commit()
    assert session.scalars(select(Secret)).one().fact_id == "fact-1"


def test_mask_outbound_replaces_without_persisting():
    """出库打码（查询期）：同一套正则替换为 [[SECRET]]、不落库、不影响密码前缀文字。"""
    out = mask_outbound("我的手机号 13800000001，密码是 abcd1234")
    assert "13800000001" not in out and "abcd1234" not in out
    assert "[[SECRET]]" in out
    assert "密码是 " in out  # 密码前缀保留（与 mask_text 同规则）
    assert mask_outbound("") == ""
    assert mask_outbound("没有敏感值") == "没有敏感值"


def test_masks_number_glued_to_chinese(tmp_path):
    # 汉字紧贴数字（「手机号是139…」）——\b 边界会失配（汉字算 \w），必须用数字 lookaround（会话 26 发现）
    session, fernet = _mk(tmp_path)
    masked, mapping = mask_text("我的手机号是13800000001，身份证号110101199001010000", session, fernet)
    assert "13800000001" not in masked
    assert "110101199001010000" not in masked
    assert sorted(mapping.values()) == ["110101199001010000", "13800000001"]


def test_masks_number_between_chinese_chars(tmp_path):
    session, fernet = _mk(tmp_path)
    masked, _ = mask_text("卡号6222020000000000已核对", session, fernet)
    assert "6222020000000000" not in masked


def test_long_digit_run_not_partially_masked(tmp_path):
    # 20 位长数字串不是有效证件/卡号，不得半截匹配（lookaround 的回归护栏）
    session, fernet = _mk(tmp_path)
    masked, mapping = mask_text("订单12345678900000000000", session, fernet)
    assert masked == "订单12345678900000000000"
    assert mapping == {}


def test_landline_not_masked(tmp_path):
    # 12 位座机号（0 开头）：既非手机也非卡号，保持不误伤
    session, fernet = _mk(tmp_path)
    _, mapping = mask_text("物业电话057100000000", session, fernet)
    assert mapping == {}


def test_mask_outbound_glued(tmp_path):
    assert "[[SECRET]]" in mask_outbound("手机号是13800000001")
