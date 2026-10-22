"""按数据空间保存模型偏好；Key 加密落盘，读取接口永不返回密钥。"""
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.core.masking import load_fernet
from app.core.model_catalog import CATALOG, policy, validate_parameters

_lock = threading.RLock()
_filename = "model-settings.json"
_keys = ("llm_api_key", "vlm_api_key")


class ModelPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_provider: str | None = Field(None, pattern=r"^(deepseek|kimi|qwen|openai|anthropic|custom)$")
    llm_base_url: str | None = Field(None, min_length=1, max_length=500)
    llm_model: str | None = Field(None, min_length=1, max_length=160)
    llm_api_key: SecretStr | None = Field(None, max_length=500)
    clear_llm_api_key: bool = False
    vlm_base_url: str | None = Field(None, min_length=1, max_length=500)
    vlm_model: str | None = Field(None, max_length=160)
    vlm_api_key: SecretStr | None = Field(None, max_length=500)
    clear_vlm_api_key: bool = False
    llm_temperature: float | None = Field(None, ge=0, le=2)
    llm_max_tokens: int | None = Field(None, ge=128, le=16384)
    llm_timeout_s: float | None = Field(None, ge=1, le=120)
    qa_max_rounds: int | None = Field(None, ge=1, le=8)

    @field_validator("llm_model", "vlm_model")
    @classmethod
    def clean_model(cls, value, info):
        if value is not None:
            value = value.strip()
            if info.field_name == "llm_model" and not value:
                raise ValueError("模型名称不能为空")
        return value

    @field_validator("llm_api_key", "vlm_api_key")
    @classmethod
    def clean_key(cls, value):
        if value is not None and any(ord(char) < 32 or ord(char) > 126 for char in value.get_secret_value()):
            raise ValueError("Key 含不支持的字符")
        return value


def _read(data_dir):
    path = Path(data_dir) / _filename
    try:
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(saved, dict):
            raise ValueError("设置格式错误")
        return saved
    except (OSError, ValueError) as error:
        raise HTTPException(503, "模型设置无法读取，请重新保存或恢复默认设置") from error


def apply(settings):
    with _lock:
        saved = _read(settings.data_dir)
        values = {key: value for key, value in saved.items() if key in ModelPreferences.model_fields
                  and key not in _keys and not key.startswith("clear_")}
        for key in _keys:
            if key in saved:
                try:
                    values[key] = load_fernet(settings.data_dir).decrypt(saved[key].encode()).decode() if saved[key] else ""
                except Exception as error:
                    raise HTTPException(503, "模型 Key 无法解密，请重新填写或恢复默认设置") from error
        return settings.model_copy(update=values)


def validate_url(value, settings):
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise HTTPException(422, "模型服务地址的端口不正确") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise HTTPException(422, "请输入不含用户名、密码、查询参数的 HTTP(S) Base URL")


def save(settings, payload):
    with _lock:
        saved = _read(settings.data_dir)
        updates = payload.model_dump(exclude_unset=True, exclude=set(_keys) | {"clear_" + k for k in _keys})
        for key in ("llm_base_url", "vlm_base_url"):
            if key in updates and updates[key] is not None:
                updates[key] = updates[key].strip().rstrip("/")
                validate_url(updates[key], settings)
                if updates[key] != getattr(settings, key).rstrip("/"):
                    # 改服务地址时不转发旧 Key，也不继承服务器的 Key。
                    saved[key.replace("base_url", "api_key")] = ""
        for key, value in updates.items():
            if value is None and key not in {"llm_temperature", "llm_max_tokens"}:
                continue
            saved[key] = value
        candidate = settings.model_copy(update={key: value for key, value in saved.items()
                                               if key not in _keys})
        validate_parameters(candidate)
        for key in _keys:
            value = getattr(payload, key)
            if getattr(payload, "clear_" + key):
                saved[key] = ""
            elif value is not None and value.get_secret_value().strip():
                saved[key] = load_fernet(settings.data_dir).encrypt(value.get_secret_value().strip().encode()).decode()
        folder = Path(settings.data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / (".model-settings-" + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, folder / _filename)
        finally:
            temporary.unlink(missing_ok=True)


def reset(settings):
    with _lock:
        (Path(settings.data_dir) / _filename).unlink(missing_ok=True)


def public(settings):
    from app.core.cloud import vlm_key
    return {
        key: getattr(settings, key) for key in (
            "llm_provider", "llm_base_url", "llm_model", "vlm_base_url", "vlm_model",
            "llm_temperature", "llm_max_tokens", "llm_timeout_s", "qa_max_rounds")
    } | {"llm_key_configured": bool(settings.llm_api_key), "vlm_key_configured": bool(vlm_key(settings)),
         "scope": "local",
         "customized": bool(_read(settings.data_dir)),
         "providers": CATALOG, "parameter_policy": policy(settings.llm_model, settings.llm_base_url),
         "effective_timeout_s": settings.llm_timeout_s,
         "allowed_hosts": []}


def configure_client(client, settings):
    original_create = client.chat.completions.create

    def create(*args, **kwargs):
        rules = policy(kwargs.get("model", settings.llm_model), client.base_url)
        if rules["blocked"]:
            raise HTTPException(422, rules["note"])
        if rules["temperature_supported"]:
            if settings.llm_temperature is not None:
                kwargs.setdefault("temperature", settings.llm_temperature)
        else:
            kwargs.pop("temperature", None)
        if rules["extra_body"]:
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **rules["extra_body"]}
        # Use the provider's actual token parameter; never send conflicting aliases.
        supplied = kwargs.pop("max_tokens", None)
        completion = kwargs.pop("max_completion_tokens", None)
        limit = completion if completion is not None else supplied
        if settings.llm_max_tokens is not None:
            limit = min(limit, settings.llm_max_tokens) if limit is not None else settings.llm_max_tokens
        if limit is not None:
            kwargs[rules["token_field"]] = limit
        return original_create(*args, **kwargs)

    client.chat.completions.create = create
    if hasattr(client, "with_options"):
        original_copy = client.with_options

        def with_options(**kwargs):
            if isinstance(kwargs.get("timeout"), (int, float)):
                kwargs["timeout"] = min(kwargs["timeout"], settings.llm_timeout_s)
            return configure_client(original_copy(**kwargs), settings)

        client.with_options = with_options
    return client
