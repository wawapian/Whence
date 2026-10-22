"""服务不可用的用户可见语义：503 + 明确文案（不是「没有找到相关记录」）。"""
import asyncio
import json

from app.core.llama_client import LlamaUnavailableError
from app.main import create_app


def test_llama_unavailable_maps_to_503_with_clear_detail():
    app = create_app()
    handler = app.exception_handlers[LlamaUnavailableError]

    response = asyncio.run(handler(None, LlamaUnavailableError("connection refused")))

    assert response.status_code == 503
    detail = json.loads(response.body)["detail"]
    assert "暂时不可用" in detail
    assert "没有找到" not in detail
