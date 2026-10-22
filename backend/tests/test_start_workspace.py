"""Launcher must keep the application port separate from model-service ports."""
import json
from types import SimpleNamespace

from scripts import start_workspace


def test_app_port_is_not_overwritten_by_model_ports(tmp_path, monkeypatch):
    for relative in ("llama/llama-server.exe", "models/bge-m3-Q8_0.gguf", "models/bge-reranker-base-Q8_0.gguf"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    monkeypatch.setattr(start_workspace, "BACKEND", tmp_path)
    monkeypatch.setattr(start_workspace, "RUNTIME", tmp_path / ".runtime")
    monkeypatch.setattr(start_workspace, "port_open", lambda port: port == 8000)
    commands = {}

    def fake_start(name, arguments, environment):
        commands[name] = arguments
        return SimpleNamespace(pid=len(commands), poll=lambda: None)

    monkeypatch.setattr(start_workspace, "start", fake_start)
    monkeypatch.setattr(start_workspace, "ready", lambda url: "workspace" in commands and url == "http://127.0.0.1:8018/healthz")
    start_workspace.main()
    assert commands["embedding"][commands["embedding"].index("--port") + 1] == "8094"
    assert commands["reranker"][commands["reranker"].index("--port") + 1] == "8095"
    assert commands["workspace"][commands["workspace"].index("--port") + 1] == "8018"
    metadata = json.loads((tmp_path / ".runtime/started-processes.json").read_text())
    assert metadata["port"] == 8018
