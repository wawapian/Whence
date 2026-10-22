"""结构化抽取 schema 与记忆类型清单。"""
from typing import List, Optional

from pydantic import BaseModel, Field

# 状态类（会变化的事实，触发双时态：新值覆盖、旧值留档）
STATE_TYPES = [
    "item_location",  # 物品位置
    "residence",      # 住址
    "contact",        # 联系方式
    "employer",       # 工作单位
    "bank_card",      # 银行卡
    "health",         # 健康状态
    "preference",     # 偏好
    "relationship",   # 关系
    "account",        # 账号
    "vehicle",        # 车辆
    "pet",            # 宠物
]

# 事件类（一次性发生，只增不减）
EVENT_TYPES = [
    "consumption",  # 消费
    "trip",         # 行程
    "todo",         # 待办
    "schedule",     # 日程
    "meeting",      # 会议
    "idea",         # 想法
    "other",        # 其他
]

ALL_TYPES = STATE_TYPES + EVENT_TYPES


class ExtractedFact(BaseModel):
    """LLM 结构化抽取结果。"""

    subject_key: Optional[str] = Field(
        None, description="规范化主语，如'身份证:位置'。状态类记忆必填"
    )
    type: str = Field(..., description=" / ".join(ALL_TYPES))
    summary: str = Field(..., description="一句话摘要")
    place: Optional[str] = None
    amount: Optional[float] = Field(
        None,
        description="金额（明确的金钱数额，如「花了 32 元」的 32）；"
                    "距离、时长、次数、数量等非金钱数字一律留空",
    )
    currency: Optional[str] = Field(
        None, description="金额的币种（如 CNY、USD）；未写明币种时留空"
    )
    people: List[str] = Field(default_factory=list)
    items: List[str] = Field(default_factory=list)
    datetime_hint: Optional[str] = Field(
        None, description="发生/截止时间，如'明天下午三点'或'下周三前'"
    )
    is_reminder: bool = Field(False, description="是否含提醒意图")
    confidence: float = Field(..., ge=0, le=1, description="抽取置信度")


class ExtractionResult(BaseModel):
    """一次抽取的全部事实：一条记录可含多个独立事实（按原文出现顺序）。"""

    facts: List[ExtractedFact]


class Source(BaseModel):
    """回答的溯源条目。"""

    fact_id: str
    summary: str | None = None
    raw: str | None = None
    file: str | None = None
    raw_id: str | None = None            # 所属记录（前端按它关联附件/删除入口）
    doc_ids: list[str] = Field(default_factory=list)  # 未删附件文档 id（点开原件用）


class DocumentSource(BaseModel):
    """回答引用的文档片段。"""

    chunk_id: str
    document_id: str
    title: str | None = None
    heading_path: str | None = None
    page: int | None = None
    text: str


class QueryResponse(BaseModel):
    """问答响应。"""

    answer: str
    sources: List[Source] = Field(default_factory=list)
    documents: List[DocumentSource] = Field(default_factory=list)
    from_cache: bool = False              # 答案缓存命中（返回的是存下的答案）
    resolved_question: str | None = None  # 实际用于管线的问题（无 history 恒 None）
    status: str = "ok"  # ok / not_found / degraded / partial
    execution: dict = Field(default_factory=dict)  # Code-generated reading coverage, files and tool trace.
