"""LLM / embedding 客户端抽象：云 LLM 走 OpenAI 兼容客户端，本地向量走 llama-server。"""
from openai import OpenAI

from app.config import Settings
from app.core import llama_client
from app.core.model_preferences import configure_client

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DIM_BY_MODEL = {"BAAI/bge-m3": 1024, "BAAI/bge-small-zh-v1.5": 512}


def build_client(settings: Settings) -> OpenAI:
    # 空 key 用占位符，构造不炸；真实调用时才需要有效 key
    options = {}
    options["timeout"] = settings.llm_timeout_s
    return configure_client(OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "dummy",
                                   **options), settings)


def embed(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """文本 → 向量：经 llama-server（规格见 spec §3.1）。

    `model` 保留为标识参数——一个 llama-server 进程只装一个模型，切换模型 = 换服务，
    不在调用点表达（13 处调用点一律传 `settings.embedding_model`，行为与旧版一致）。
    """
    return llama_client.embed_texts([text])[0]


def embedding_dim(model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """模型维度：**纯查表**——换引擎后不再为未知模型加载权重（表外即抛，见 spec §3.4）。"""
    if model in _DIM_BY_MODEL:
        return _DIM_BY_MODEL[model]
    raise ValueError(f"模型 {model} 的维度未登记，请加进 _DIM_BY_MODEL")
