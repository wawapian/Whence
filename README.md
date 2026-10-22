# Whence · 来处

> 每一段记忆，都有来处。

把文字、照片和文件留在本地，用一句话找回生活的来龙去脉——每条回答都指回它来自的记录或原件。

英文名 **Whence** 读 /wɛns/，意为“从何处来”；中文名 **来处**。记忆不只要存得住，更要答得清、查得回——这是名字，也是这个项目存在的理由。

本仓库是 Whence · 来处 的本地开源版：完整前后端、模型设置、记录、原件库、历史、时间线、提醒和回收站。这里的“本地”指数据默认保存在本机；文字抽取与问答仍可能调用你配置的外部模型。

## 它能做什么

- **记录**：文字、照片、PDF / Word / Excel / Markdown，原件留在本地。
- **找回**：用自然语言提问，关键词 + 向量 + 重排混合检索，答案附出处。
- **追溯**：顺出处打开原记录或原件；同一件事的旧状态也能查。
- **生活**：时间线、提醒、消费统计、回收站与 Markdown 导出。
- **本地**：单用户、默认只监听回环；身份证、银行卡、手机号、密码等敏感信息在本地打码；附件上云需逐次授权。

## 数据与隐私

- 记录、附件、SQLite 数据库与加密密钥默认在本机 `backend/data/`，不随仓库分发。
- 敏感信息在写入前于本地打码，密文由本机生成的 `backend/data/secret.key` 加密。
- 只有在你明确授权后，相关内容才会发送给你配置的外部模型服务；项目不上传遥测。
- 仓库不包含数据库、`.env`、个人数据、模型权重或推理二进制。

## 快速开始（Windows）

1. 安装 Python 3.11+，双击 `setup.bat` 创建 `.venv` 并安装依赖。
2. 按 [本地模型说明](docs/本地模型.md) 下载两个 GGUF 放入 `backend/models/`，下载 llama.cpp 放入 `backend/llama/`。仓库不附带这些文件。
3. 双击 `start-workspace.bat`，浏览器打开 http://127.0.0.1:8000/#chat 。
4. 在页面“模型设置”里填自己的服务地址、模型 ID 和 Key。
5. 双击 `stop-workspace.bat` 停止服务。

## 开发者安装

需要 Python 3.11+。Windows 可运行 `setup.bat`；也可在 backend 中用 `uv sync --locked`（开发测试含 dev 依赖），或创建 `.venv` 后用 `python -m pip install -e .`。复制 `.env.example` 为 `.env`，按 [本地模型说明](docs/本地模型.md) 放置匹配系统的 GGUF 和 llama.cpp 二进制，然后启动。

Linux / macOS：先独立启动嵌入与重排服务，配置对应地址，在 backend 中执行 `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`。前端由后端同源提供，无需 Node、前端构建或单独启动。

模型权重、llama.cpp 二进制、Python 环境和运行数据都被 `.gitignore` 排除，`backend/models/` 与 `backend/llama/` 只保留说明文件；请按 [本地模型说明](docs/本地模型.md) 自行下载并核对来源、版本、量化与许可证。

## 目录

```text
frontend/                 完整本机前端
backend/app/              API、模型客户端、业务与数据处理
backend/tests/            业务与接口测试
backend/scripts/          启动、重建索引和维护工具
backend/ops/              本地模型 Linux 服务参考模板
backend/pyproject.toml     依赖定义
backend/uv.lock            依赖锁文件
backend/.env.example       无密钥配置模板
backend/models/            本地模型（Git 忽略）
backend/llama/             本地推理程序（Git 忽略）
backend/.venv/             本机 Python 环境（Git 忽略）
backend/data/              首次使用后生成的数据和加密密钥（Git 忽略）
backend/.runtime/          运行日志与进程记录（Git 忽略）
docs/本地模型.md           模型下载、分发、替换及重建索引
start-workspace.bat        启动本机版
stop-workspace.bat         停止本目录服务
setup.bat                  在新电脑安装 Python 依赖
LICENSE                    项目现有 MIT 许可证
```

应用是本机单用户工具，无账号鉴权；默认只监听回环地址，不应直接暴露到公网。

## 仓库边界

只提交源码、文档、LICENSE、配置模板与 `uv.lock`；`.env`、任何 Key、个人数据、数据库、模型权重、二进制、虚拟环境与日志都不入库。请不要在提交中带入本机数据。
