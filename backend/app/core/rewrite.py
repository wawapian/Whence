"""查询改写：多查询扩展（分级触发）。"""
import logging
import re

from app.config import Settings

logger = logging.getLogger(__name__)

MAX_VARIANTS = 3

_EXPAND_PROMPT = (
    "把下面的问题改写成 2-3 条适合全文检索的简洁查询，"
    "从不同措辞角度覆盖，保留关键实体、术语、文件名等。"
    "每条一行，不要编号，不要回答或解释问题本身。\n\n"
    "问题：{question}"
)


def should_expand(question: str, settings: Settings) -> bool:
    """分级触发：仅 auto 模式、且问题长度达标时才做多查询扩展。"""
    return (
        settings.query_rewrite == "auto"
        and len(question.strip()) >= settings.query_rewrite_min_chars
    )


def expand_queries(client, model: str, question: str) -> list[str]:
    """LLM 一次生成最多 3 条检索变体；调用/解析失败返回 []（调用方降级为原问题）。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _EXPAND_PROMPT.format(question=question)}],
        )
        content = resp.choices[0].message.content or ""
    except Exception:
        logger.warning("查询扩展 LLM 调用失败（降级为原问题检索）", exc_info=True)
        return []
    variants = _parse_variants(content)
    if not variants:
        logger.warning("查询扩展输出无法解析（降级为原问题检索）：%r", content[:200])
    return variants


def _parse_variants(text: str) -> list[str]:
    """逐行解析：去编号/项目符号、去空行、去重、上限 MAX_VARIANTS 条。"""
    variants: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+\s*[.、)）:：])\s*", "", line).strip()
        if line and line not in variants:
            variants.append(line)
    return variants[:MAX_VARIANTS]


def retrieval_queries(question: str, settings: Settings, client) -> list[str]:
    """检索用查询列表：不触发/失败 → [原问题]；触发 → [原问题, ...变体]（原问题保底）。"""
    if not should_expand(question, settings):
        return [question]
    return [question] + expand_queries(client, settings.llm_model, question)
