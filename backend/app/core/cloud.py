"""云 LLM 能力：摘要/提纯与多模态（VLM）。"""
import base64
from urllib.parse import urlsplit

from openai import OpenAI
from app.core.model_preferences import configure_client


# 云总结的「拒绝/复读」特征：输入退化（如正文只剩表格标题与行列数说明）时，模型不提炼，
# 反而要求用户补文本——这类输出不是摘要，入库会毒化材料索引（会话 26 评测实测：
# 76 条材料 7 条中招，其中一条使目标文档在材料层零信号、问题召不回）。命中即弃用。
_SUMMARY_REFUSAL_HINTS = ("请提供", "请补充", "请粘贴", "我会据此", "我将为您", "我会帮",
                          "无法处理", "无法提炼")


def summarize(client: OpenAI, model: str, text: str) -> str:
    """把清洗后文本发给云 LLM 做去噪 + 要点提炼，返回摘要（拒绝/复读文本 → 空串）。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是文档去噪与要点提炼助手。"},
            {"role": "user", "content": f"请对以下文本做去噪并提炼要点，输出简洁摘要：\n\n{text}"},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    if any(hint in content for hint in _SUMMARY_REFUSAL_HINTS):
        return ""
    return content


def vlm_key(settings):
    return settings.vlm_api_key or (settings.llm_api_key if settings.llm_provider == "qwen"
        and (urlsplit(settings.llm_base_url).scheme, urlsplit(settings.llm_base_url).netloc)
        == (urlsplit(settings.vlm_base_url).scheme, urlsplit(settings.vlm_base_url).netloc) else "")


def build_vlm_client(settings) -> OpenAI:
    """VLM 客户端：vlm_api_key 优先；仅 provider 为 qwen 时可回退 llm_api_key。"""
    options = {}
    options["timeout"] = settings.llm_timeout_s
    return configure_client(OpenAI(
        base_url=settings.vlm_base_url,
        api_key=vlm_key(settings) or "dummy",
        **options,
    ), settings.model_copy(update={"llm_temperature": None}))


def describe_image(client: OpenAI, model: str, image_bytes: bytes, question: str) -> str:
    """用云 VLM 理解图片内容，返回对 question 的文字回答。"""
    if image_bytes[:3] == b"\xff\xd8\xff":
        mime = "data:image/jpeg;base64,"
    elif image_bytes[:4] == b"\x89PNG":
        mime = "data:image/png;base64,"
    else:
        mime = "data:image/jpeg;base64,"
    data_uri = mime + base64.b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content
