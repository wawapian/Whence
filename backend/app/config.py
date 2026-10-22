from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.core.model_catalog import CATALOG

# provider -> 默认 base_url / model
PROVIDER_PRESETS = {name: {"base_url": item["base_url"], "model": item["models"][0]}
                    for name, item in CATALOG.items() if item["models"]}


class Settings(BaseSettings):
    """应用配置：优先级 环境变量 > .env > 默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 云 LLM（OpenAI 兼容）
    llm_provider: str = "deepseek"
    llm_base_url: str = ""  # 空则按 provider 预设自动填
    llm_api_key: str = ""
    llm_model: str = ""  # 空则按 provider 预设自动填
    llm_temperature: float | None = Field(None, ge=0, le=2)
    llm_max_tokens: int | None = Field(None, ge=128, le=16384)
    llm_timeout_s: float = Field(60, ge=1, le=120)

    # 云 VLM（多模态，OpenAI 兼容）
    vlm_model: str = ""  # 空 = 关闭图片理解
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_api_key: str = ""

    # 本地 embedding
    embedding_model: str = "BAAI/bge-m3"

    # Local model launcher: paths are relative to backend unless absolute.
    embedding_model_file: Path = Path("models/bge-m3-Q8_0.gguf")
    rerank_model_file: Path = Path("models/bge-reranker-base-Q8_0.gguf")
    llama_server_binary: Path = Path("llama/llama-server.exe")
    llama_threads: int = Field(8, ge=1)
    llama_embed_context: int = Field(8192, ge=512)

    # 重排
    rerank_engine: str = "local"
    rerank_model: str = "BAAI/bge-reranker-base"

    # 本地推理服务（llama-server）：一个进程只装一个模型，故向量与重排各起一个实例。
    # 端口 8094/8095 刻意避开 8084——本机 Prism 项目已占用该端口。
    llama_embed_url: str = "http://127.0.0.1:8094"
    llama_rerank_url: str = "http://127.0.0.1:8095"
    llama_timeout_s: float = 30.0  # 单次请求超时（秒）
    llama_warmup_timeout_s: float = 60.0  # 启动预热轮询就绪上限（秒；超时只 warning、不阻塞启动）

    # 启动预热：lifespan 起后台 daemon 线程预载本地模型（失败只记日志、不阻塞启动；WARMUP=false 关闭）
    warmup: bool = True

    # 查询改写（多查询扩展）
    query_rewrite: str = "off"  # off / auto
    query_rewrite_min_chars: int = 15  # auto 模式：问题长度 ≥ 此值才扩展

    # 问答层（spec §5）：tool_loop = 模型主导主循环 + 五工具；legacy = 旧 4 管线（回退开关）
    qa_engine: str = "tool_loop"
    qa_max_rounds: int = 4  # 工具调用轮数上限 N（spec 已决；随实测调）

    # 答案缓存（批三 spec §4）：off 不查不记 / log 只记「若开会否命中」日志（行为同 off）/ on 命中直答
    answer_cache: str = "off"
    # 余弦命中阈值：T8 标定取 0.95 → 2026-10-16 **上调 0.98**。
    # 起因：评测实测「我这个月一共花了多少钱？」与「上个月我一共花了多少钱？」余弦 **0.9749**，
    # 旧值会让**语义相反**的问题互相命中（q-0031 拿到了 q-0029 的答案，逐字相同）；
    # 而原句复问余弦 = 1.0，0.98 不影响「重复问秒回」这个加速器的价值。
    # 注意：标定脚本（scripts/calibrate_answer_cache.py）的 14 组冻结题对里**没有**这类样本，
    # 所以它仍建议 0.95——以本条实测为准，勿据脚本建议回退。
    answer_cache_threshold: float = 0.98
    answer_cache_top_k: int = 200  # 参与比对的最近历史行数上限
    qa_history_ttl_hours: int = 24  # 缓存行有效期（小时）
    qa_history_window: int = 10  # 追问上下文窗口：服务端只取最近 N 组 history

    # 装饰小图降级（保守启发式：极小且均匀纯色块——只影响自动描述/索引，文件与行永久保留）
    image_filter: str = "on"  # on / off

    # 存储路径
    data_dir: Path = Path("./data")

    # 解析 / 分块 / OCR
    ocr_engine: str = "rapidocr"
    ocr_lang: str = "auto"
    advanced_engine: str = "off"
    chunk_strategy: str = "auto"
    chunk_max_chars: int = 600
    chunk_overlap: int = 60

    @model_validator(mode="after")
    def _apply_preset(self) -> "Settings":
        preset = PROVIDER_PRESETS.get(self.llm_provider)
        if preset:
            if not self.llm_base_url:
                self.llm_base_url = preset["base_url"]
            if not self.llm_model:
                self.llm_model = preset["model"]
        return self


def get_settings() -> Settings:
    return Settings()
