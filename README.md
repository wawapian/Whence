# Whence · 来处

## 它能做什么

- **记录**：文字、照片、PDF / Word / Excel / Markdown，原件留在本地。
- **找回**：用自然语言提问，关键词 + 向量 + 重排混合检索，答案附出处。
- **追溯**：顺出处打开原记录或原件；同一件事的旧状态也能查。
- **生活**：时间线、提醒、消费统计、回收站与 Markdown 导出。
- **本地**：单用户、默认只监听回环；身份证、银行卡、手机号、密码等敏感信息在本地打码；附件上云需逐次授权。

## 数据与隐私

- 记录、附件、SQLite 数据库与加密密钥默认在本机 `backend/data/`。
- 敏感信息在写入前于本地打码，密文由本机生成的 `backend/data/secret.key` 加密。
- 只有在你明确授权后，相关内容才会发送给你配置的外部模型服务；应用不上传遥测。

## 准备工作：两个本地模型和 llama.cpp

检索用的两个模型需要你自己下载，放到 `backend/models/`：

| 文件 | 用途 | 官方模型卡 |
| --- | --- | --- |
| `bge-m3-Q8_0.gguf` | 文本向量 | https://huggingface.co/BAAI/bge-m3 |
| `bge-reranker-base-Q8_0.gguf` | 重排 | https://huggingface.co/BAAI/bge-reranker-base |

模型卡提供的是原始权重，需要转成 GGUF 或使用可信的 GGUF 版本，并核对来源、版本、量化方式和许可证。

推理程序用 llama.cpp：从 https://github.com/ggml-org/llama.cpp/releases 下载与系统匹配的构建，把 `llama-server.exe` 及同批 DLL 放进 `backend/llama/`。启动应用时会自动用它拉起嵌入服务（8094）和重排服务（8095）。

## 快速开始（Windows）

1. 安装 Python 3.11+，双击 `setup.bat` 创建 `.venv` 并安装依赖。
2. 按上一节放好两个 GGUF 与 `llama-server.exe`。
3. 双击 `start-workspace.bat`，浏览器打开 http://127.0.0.1:8000/#chat 。
4. 在页面“模型设置”里填写外部模型的服务地址、模型 ID 和 Key。
5. 双击 `stop-workspace.bat` 停止服务。

## 开发者安装

需要 Python 3.11+。在 `backend/` 中用 `uv sync --locked`（含测试依赖），或创建 `.venv` 后 `python -m pip install -e .`。复制 `.env.example` 为 `.env`，按上一节放好模型与 llama.cpp，然后启动。

Linux / macOS：按 llama.cpp server 的文档启动嵌入与重排两个服务，把地址写进 `.env` 的 `LLAMA_EMBED_URL` / `LLAMA_RERANK_URL`，再在 `backend/` 中执行：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

前端由后端同源提供，不需要 Node 或单独构建。

## 目录

```text
frontend/                 完整前端
backend/app/              API、模型客户端、业务与数据处理
backend/tests/            业务与接口测试
backend/scripts/          启动、重建索引和维护工具
backend/ops/              本地模型 Linux 服务参考模板
backend/pyproject.toml     依赖定义
backend/uv.lock            依赖锁文件
backend/.env.example       配置模板
backend/models/            放本地模型文件
backend/llama/             放 llama.cpp 推理程序
backend/data/              运行数据（首次启动生成）
backend/.runtime/          运行日志
start-workspace.bat        启动
stop-workspace.bat         停止
setup.bat                  安装 Python 依赖
LICENSE                    MIT 许可证
```

应用是本机单用户工具，无账号鉴权；默认只监听回环地址，不应直接暴露到公网。
