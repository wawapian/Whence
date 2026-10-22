# syntax=docker/dockerfile:1
# Whence · 来处 —— 后端（FastAPI）+ 前端静态文件
# 模型（gguf）与 llama-server 二进制不进镜像，用 compose 单独起两个推理容器。
FROM python:3.11-slim

# opencv（rapidocr 依赖）需要这些系统库；python:slim 默认没有，缺了会在
# `import cv2` 时报 libxcb.so.1 / libGL.so.1 找不到，后端直接起不来。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 用 uv 按 uv.lock 装依赖（可复现），装进 /app/backend/.venv
# uv cache clean 放在同一层里，避免把下载缓存也打进镜像
RUN pip install --no-cache-dir uv
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen --no-dev --no-install-project && uv cache clean

# 只拷运行需要的代码；frontend 是后端同源托管的静态页
COPY backend/app ./backend/app
COPY backend/scripts ./backend/scripts
COPY frontend ./frontend

ENV DATA_DIR=/data \
    PATH="/app/backend/.venv/bin:$PATH"

EXPOSE 8000

# --app-dir /app/backend：让 app 在 /app/backend 下被 import，与本地 uv 运行方式一致
CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
