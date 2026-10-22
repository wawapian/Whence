from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.config import Settings
from app.core import model_preferences
from app.deps import get_settings

class SecretSafeRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request):
            try:
                return await original(request)
            except RequestValidationError:
                # 默认校验错误带 input 原值；含 Key 的请求只返回通用错误。
                return JSONResponse({"detail": "设置格式不正确，请检查模型名称、服务地址和参数范围"},
                                    status_code=422, headers={"Cache-Control": "private, no-store"})
        return handler


router = APIRouter(prefix="/api/model-settings", tags=["model settings"], route_class=SecretSafeRoute)


def same_origin(request: Request):
    origin = request.headers.get("origin")
    try:
        matches = not origin or (urlsplit(origin).scheme, urlsplit(origin).netloc) == (request.url.scheme, request.url.netloc)
    except ValueError:
        matches = False
    if request.headers.get("sec-fetch-site") == "cross-site" or not matches:
        raise HTTPException(403, "请从本站页面修改模型设置")


@router.get("")
def read_settings(response: Response, settings: Settings = Depends(get_settings)):
    response.headers["Cache-Control"] = "private, no-store"
    return model_preferences.public(settings)


@router.put("", dependencies=[Depends(same_origin)])
def write_settings(payload: model_preferences.ModelPreferences, response: Response,
                   settings: Settings = Depends(get_settings)):
    model_preferences.save(settings, payload)
    response.headers["Cache-Control"] = "private, no-store"
    # 从原始配置重算，避免重复叠加旧偏好；空间身份保持不变。
    base = Settings().model_copy(update={"data_dir": settings.data_dir})
    return model_preferences.public(model_preferences.apply(base))


@router.post("/policy", dependencies=[Depends(same_origin)])
def parameter_policy(payload: model_preferences.ModelPreferences, response: Response,
                     settings: Settings = Depends(get_settings)):
    """Local metadata only: never send credentials or call a provider."""
    base_url = payload.llm_base_url or settings.llm_base_url
    model_preferences.validate_url(base_url, settings)
    response.headers["Cache-Control"] = "private, no-store"
    return model_preferences.policy(payload.llm_model or settings.llm_model, base_url)


@router.delete("", dependencies=[Depends(same_origin)])
def reset_settings(response: Response, settings: Settings = Depends(get_settings)):
    model_preferences.reset(settings)
    base = Settings().model_copy(update={"data_dir": settings.data_dir})
    response.headers["Cache-Control"] = "private, no-store"
    return model_preferences.public(base)
