"""instructor 结构化抽取。"""
import instructor
from openai import OpenAI

from app.models.schemas import ExtractedFact, ExtractionResult

# 一条记录可含多个陈述（发现#2：位置类陈述曾被整条漏抽）→ 逐条拆解、状态类不得遗漏、按出现顺序
_PROMPT = (
    "分析下面这段文本，把它拆解为**全部独立事实**并逐条结构化抽取（只输出符合 schema 的 JSON）：\n"
    "- 一句话可含多个事实：每个独立事实单独一条，不得合并、不得遗漏；\n"
    "- 状态类陈述（位置／偏好／联系方式等）必须逐条抽出，不得遗漏；\n"
    "- 按事实在原文中出现的顺序输出；\n"
    "- 每条遵循 schema 规则（状态类必填 subject_key，如'身份证:位置'）。\n"
    "文本：\n{text}"
)


def extract(client: OpenAI, model: str, text: str) -> list[ExtractedFact]:
    """把一段文本抽取为**全部独立事实**（按出现顺序）；模型零产出时返回空列表。"""
    iclient = instructor.from_openai(client)
    result = iclient.chat.completions.create(
        model=model,
        response_model=ExtractionResult,
        messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
    )
    return result.facts
