"""Start the existing local stack; all owned logs/cache stay beside the project."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
from urllib.parse import urlsplit
from app.config import Settings

BACKEND = Path(__file__).resolve().parents[1]
RUNTIME = BACKEND / ".runtime"
HTTP = build_opener(ProxyHandler({}))


def ready(url):
    try:
        with HTTP.open(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def port_open(port):
    with socket.socket() as connection:
        connection.settimeout(.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def start(name, arguments, environment):
    with (RUNTIME / f"{name}.log").open("ab") as log:
        return subprocess.Popen(arguments, cwd=BACKEND, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def main(check_only=False, requested_port=None):
    settings = Settings()
    binary = BACKEND / settings.llama_server_binary
    models = [BACKEND / settings.embedding_model_file, BACKEND / settings.rerank_model_file]
    destinations = [urlsplit(settings.llama_embed_url), urlsplit(settings.llama_rerank_url)]
    if any(url.hostname != "127.0.0.1" or url.scheme != "http" or not url.port for url in destinations):
        raise RuntimeError("一键启动仅支持带端口的 http://127.0.0.1 本地模型地址；远端服务请手动启动后端。")
    embed_port, rerank_port = [url.port for url in destinations]
    if not all(item.is_file() for item in [binary, *models]):
        raise RuntimeError("未找到项目原有的本地模型服务或模型文件，请按 README 配置 backend/llama 与 backend/models。")
    if check_only:
        print("启动检查通过：Python、本地模型服务与两个模型文件均已找到。")
        return
    metadata = RUNTIME / "started-processes.json"
    if metadata.is_file():
        try:
            previous_port = int(json.loads(metadata.read_text(encoding="utf-8")).get("port", 0))
            if previous_port and (requested_port is None or requested_port == previous_port) and ready(f"http://127.0.0.1:{previous_port}/api/workspace/overview"):
                print(f"新版记忆空间已在运行：http://127.0.0.1:{previous_port}")
                return
        except (ValueError, OSError, TypeError):
            pass
    # An older checkout may already own :8000. Leave it running and use a free local port.
    candidates = [requested_port] if requested_port is not None else [8000, 8018, 8019, 8020]
    app_port = next((candidate for candidate in candidates if not port_open(candidate)), None)
    if app_port is None:
        raise RuntimeError("可用的本机端口均被占用，请关闭已启动的重复服务后重试。")
    if app_port != 8000 and requested_port is None:
        print(f"8000 端口已有服务，新版将使用 {app_port}，原进程不受影响。")
    RUNTIME.mkdir(exist_ok=True)
    temp = RUNTIME / "tmp"; temp.mkdir(exist_ok=True)
    environment = {**os.environ, "TEMP": str(temp), "TMP": str(temp), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    context, threads = str(settings.llama_embed_context), str(settings.llama_threads)
    services = [
        ("embedding", embed_port, [str(binary), "-m", str(models[0]), "--embeddings", "-c", context, "-b", context, "-ub", context, "-t", threads, "--port", str(embed_port), "--host", "127.0.0.1"]),
        ("reranker", rerank_port, [str(binary), "-m", str(models[1]), "--reranking", "--pooling", "rank", "-t", threads, "--port", str(rerank_port), "--host", "127.0.0.1"]),
    ]
    started = {}
    for name, model_port, command in services:
        if ready(f"http://127.0.0.1:{model_port}/health"):
            print(f"复用已运行的本地模型服务：{model_port}")
        elif port_open(model_port):
            raise RuntimeError(f"端口 {model_port} 已被占用且未就绪，请检查本地模型服务。")
        else:
            process = start(name, command, environment); started[name] = process.pid
            metadata.write_text(json.dumps(started, indent=2), encoding="utf-8")
            print(f"正在后台加载本地模型：{name}")
    process = start("workspace", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(app_port)], environment)
    started["workspace"] = process.pid
    started["port"] = app_port
    metadata.write_text(json.dumps(started, indent=2), encoding="utf-8")
    for _ in range(40):
        if ready(f"http://127.0.0.1:{app_port}/healthz"):
            print(f"记忆空间已启动：http://127.0.0.1:{app_port}")
            print("首次检索前请等待本地模型完成加载。")
            print(f"运行日志：{RUNTIME}")
            return
        if process.poll() is not None:
            raise RuntimeError(f"服务未能启动，请查看 {RUNTIME / 'workspace.log'}")
        time.sleep(.5)
    raise RuntimeError(f"启动等待超时，请查看 {RUNTIME / 'workspace.log'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check paths without starting services or opening data")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", help="Use a specific local application port")
    args = parser.parse_args()
    try:
        main(args.check, args.port)
    except Exception as error:
        print(f"启动未完成：{error}", file=sys.stderr)
        sys.exit(1)
