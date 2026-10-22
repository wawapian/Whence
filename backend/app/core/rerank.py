"""重排：本地 llama-server / 云 LLM / 关闭。"""
import instructor
from pydantic import BaseModel

from app.core import llama_client

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


# 送服务前**整对（查询 + 正文）**的字符预算：对齐旧 torch 版 CrossEncoder 的
# max_length=512 口径。必须截断——llama.cpp 与 torch 不同，**它不截断**：超长输入
# 直接 500（2026-10-16 实测：5 字查询 + 800 字正文通过、+1200 字挂；本库有 6079 字的图注块）。
# 注意预算约束的是**整对**，不是每段各一份——所以正文的额度 = 预算 − 查询长度。
# 历史：曾把 256 当提速杠杆，A/B 归因后撤回——截断才动头名证据（512 截 17.6% 块，256 截 43.2%）。
_MAX_PAIR_CHARS = 512
_MIN_DOC_CHARS = 128  # 查询很长时给正文留的最小额度（否则等于对空串打分）


def local_rerank(query: str, candidates: list[str], model: str = DEFAULT_RERANK_MODEL) -> list[int]:
    """本地重排（经 llama-server）：返回按相关性**降序**的候选索引。

    `model` 保留为标识参数（同 `llm.embed`）——模型装在服务端，不在调用点切换。
    """
    if not candidates:
        return []
    head = query[: _MAX_PAIR_CHARS - _MIN_DOC_CHARS]
    budget = max(_MAX_PAIR_CHARS - len(head), _MIN_DOC_CHARS)
    scores = llama_client.rerank_pairs(head, [c[:budget] for c in candidates])
    return sorted(range(len(candidates)), key=lambda i: -scores[i])


class _Ranking(BaseModel):
    ranking: list[int]


def llm_rerank(client, model: str, query: str, candidates: list[str]) -> list[int]:
    if not candidates:
        return []
    numbered = "\n".join(f"[{i}] {c[:200]}" for i, c in enumerate(candidates))
    iclient = instructor.from_openai(client)
    resp = iclient.chat.completions.create(
        model=model,
        response_model=_Ranking,
        messages=[{"role": "user", "content": f"按与问题的相关性从高到低给候选排序，返回全部索引：\n问题：{query}\n候选：\n{numbered}"}],
    )
    order = []
    seen = set()
    for i in resp.ranking:
        if 0 <= i < len(candidates) and i not in seen:
            order.append(i)
            seen.add(i)
    order += [i for i in range(len(candidates)) if i not in seen]
    return order


def rerank(query: str, candidates: list[str], *, engine: str = "local",
           model: str = DEFAULT_RERANK_MODEL, client=None, llm_model: str = "") -> list[int]:
    if not candidates:
        return []
    if engine == "off":
        return list(range(len(candidates)))
    if engine == "llm":
        if client is None:
            return list(range(len(candidates)))  # 无 client 降级不重排
        return llm_rerank(client, llm_model, query, candidates)
    if engine == "local":
        return local_rerank(query, candidates, model)
    raise ValueError(f"unknown rerank_engine: {engine}")
