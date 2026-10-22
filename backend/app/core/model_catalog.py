"""Official Chat API presets, reviewed 2026-10-21; account access still varies."""
from urllib.parse import urlsplit
import re

CATALOG = {
    "deepseek": {"name": "DeepSeek", "base_url": "https://api.deepseek.com", "models": ["deepseek-flash", "deepseek-v4-pro"], "docs": "https://api-docs.deepseek.com/", "note": "使用非思考模式，温度可调。思考模式下温度无效。"},
    "kimi": {"name": "Kimi", "base_url": "https://api.moonshot.cn/v1", "models": ["kimi-k3", "kimi-k2.6"], "docs": "https://platform.kimi.com/docs/guide/kimi-k3-quickstart", "note": "K3 始终思考，温度固定为 1；K2.6 使用非思考模式，温度固定为 0.6。"},
    "qwen": {"name": "通义千问", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "models": ["qwen3.8-flash", "qwen3.8-max", "qwen-plus"], "docs": "https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions", "note": "使用非思考模式。Key 与服务地域必须一致；工作空间专属地址可在本机版填写。"},
    "custom": {"name": "自定义兼容服务", "base_url": "", "models": [], "docs": "", "note": "需支持 Chat Completions 和工具调用；参数是否支持以该服务文档为准。"},
}


def policy(model, base_url):
    """Use the actual destination and requested model, including separate VLM calls."""
    host = urlsplit(str(base_url)).hostname
    model = model.lower()
    result = {"temperature_supported": True, "temperature_max": 2, "temperature_exclusive": False,
              "token_field": "max_tokens", "extra_body": {}, "blocked": False, "note": ""}
    if host == "api.openai.com":
        result["token_field"] = "max_completion_tokens"
        if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
            result.update(temperature_supported=False, note="当前推理模式不发送温度参数。")
        if model.startswith("gpt-6"):
            result.update(blocked=True, note="GPT-6 系列的工具调用需 Responses API，当前工作台尚未适配，请选择 GPT-4.1 系列。")
    elif host == "api.anthropic.com":
        result["temperature_max"] = 1
        legacy = bool(re.match(r"^claude-(3(?:-|$)|(?:opus|sonnet|haiku)-4-(?:[0156])(?:-|$))", model))
        result.update(temperature_supported=legacy, note="新版 Claude 温度固定，不发送温度参数；兼容接口支持基础工具调用。")
    elif host in {"api.moonshot.cn", "api.moonshot.ai"}:
        if model.startswith("kimi-k3"):
            result.update(temperature_supported=False, token_field="max_completion_tokens", note="Kimi K3 始终思考，温度固定为 1，不发送温度参数。")
        elif model.startswith(("kimi-k2.5", "kimi-k2.6", "kimi-k2.7")):
            result.update(temperature_supported=False, extra_body={"thinking": {"type": "disabled"}}, note="该 Kimi 模型使用非思考模式和服务商固定温度，不发送自定义温度。")
    elif host == "api.deepseek.com":
        if model.startswith(("deepseek-flash", "deepseek-v4")):
            result.update(extra_body={"thinking": {"type": "disabled"}}, note="当前使用非思考模式，温度可调。")
        elif model == "deepseek-reasoner":
            result.update(temperature_supported=False, note="此模型的思考模式不使用温度。")
    elif host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}:
        result.update(temperature_exclusive=True)
        if model.startswith(("qwen3", "qwen-plus", "qwen-flash")):
            result["extra_body"] = {"enable_thinking": False}
    return result


def validate_parameters(settings):
    from fastapi import HTTPException
    rules = policy(settings.llm_model, settings.llm_base_url)
    if rules["blocked"]:
        raise HTTPException(422, rules["note"])
    value = settings.llm_temperature
    if value is not None:
        if not rules["temperature_supported"]:
            raise HTTPException(422, rules["note"] + " 请清空 Temperature。")
        if value > rules["temperature_max"] or (rules["temperature_exclusive"] and value == rules["temperature_max"]):
            raise HTTPException(422, "该模型的 Temperature 超出服务商支持范围")
    return rules
