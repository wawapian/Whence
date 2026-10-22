"""查询路由：LLM 意图判定，失败退回精确规则。

意图 = 处理方式（怎么处理），不是主题（处理什么）：
- summarize      → 总结路径（读整篇文档做摘要）
- image_query    → 图片理解（查询时 VLM 看图）
- analysis_query → 表格条件统计（LLM 抽条件 → 代码筛选/聚合 → LLM 组答）
- qa             → 默认检索问答

未来新增处理方式（提醒等）时：枚举加一项 + 分类提示词加一行 + answer.py 接一条分支。
"""
import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class QueryKind(str, Enum):
    summarize = "summarize"
    image_query = "image_query"
    analysis_query = "analysis_query"
    qa = "qa"


# 规则兜底（仅 LLM 失败时使用）：精确模式，不用「是什么」这类宽词
_SUMMARIZE_KEYWORDS = (
    "总结", "概括", "概述", "讲了什么", "讲了啥", "主要内容", "大意", "写了什么", "写了啥", "简介",
)
_IMAGE_KEYWORDS = ("图", "图片", "图表", "截图", "照片", "这张")

_CLASSIFY_PROMPT = (
    "判断下面这个问题的类型，只输出一个类别名（summarize / image_query / analysis_query / qa），不要输出其他任何内容。\n\n"
    "summarize：要求总结、概括某份文档或资料的全文内容\n"
    "image_query：询问图片、图表、截图、照片里的内容（如颜色、图形、图中数据）\n"
    "analysis_query：要求对数据做条件筛选/计数/求和/对比统计的问题（如「少于/超过 X 的有哪些」「一共多少」「总计」）\n"
    "qa：其他一切（查找、回忆、询问具体事实）\n\n"
    "问题：{question}"
)


def mentions_images(question: str) -> bool:
    """问题是否提到图片类内容（用于总结路径附加图片描述）。"""
    return any(k in question for k in _IMAGE_KEYWORDS)


def route(question: str) -> QueryKind:
    """关键词兜底判定（LLM 不可用时）：仅识别最明确的两类，其余默认 qa。

    analysis_query 为 LLM 专属（规则无法可靠识别统计诉求）：LLM 判定失败时
    这类问题落 qa = 降级普通问答。
    """
    if any(k in question for k in _SUMMARIZE_KEYWORDS):
        return QueryKind.summarize
    if mentions_images(question):
        return QueryKind.image_query
    return QueryKind.qa


def classify(question: str, client, model: str) -> QueryKind:
    """LLM 意图判定；调用失败/输出无法解析 → 退回 route() 规则兜底。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(question=question)}],
        )
        content = resp.choices[0].message.content or ""
    except Exception:
        logger.warning("意图判定 LLM 调用失败（退回规则兜底）", exc_info=True)
        return route(question)
    kind = _parse_kind(content)
    if kind is None:
        logger.warning("意图判定输出无法解析（退回规则兜底）：%r", content[:120])
        return route(question)
    return kind


def _parse_kind(text: str) -> QueryKind | None:
    """从 LLM 输出中提取类别名（容忍夹带说明文字）；提取不到返回 None。"""
    match = re.search(r"\b(analysis_query|summarize|image_query|qa)\b", text.lower())
    return QueryKind(match.group(1)) if match else None
