"""The OpenAI-compatible backend against a local fake server (no network needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from derivcheck import check_derivation
from derivcheck.llm import LLMError, get_backend

ANSWER = {
    "assumptions": {"hbar": "positive", "m": "positive"},
    "steps": [
        {"latex": r"p = \hbar k", "lhs": "p", "rhs": "hbar*k", "relation": "definition"},
        {"latex": r"E = \frac{p^2}{2m}", "lhs": "E", "rhs": "p**2/(2*m)", "relation": "given"},
        {"latex": r"= \frac{\hbar^2 k^2}{2m}", "lhs": "E", "rhs": "hbar**2*k**2/(2*m)", "relation": "equals"},
    ],
}


class FakeServer(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        FakeServer.requests.append(json.loads(self.rfile.read(length)))
        content = "<think>reasoning...</think>\n" + json.dumps(ANSWER)
        self._send({"choices": [{"message": {"content": content}}]})


@pytest.fixture
def server(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LMSTUDIO_HOST", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)
    FakeServer.requests = []
    yield httpd
    httpd.shutdown()


def test_lmstudio_backend_end_to_end(server):
    report = check_derivation("E = p^2/2m = hbar^2 k^2/2m, with p = hbar k", "lmstudio")
    assert report.backend == "lmstudio:fake-model"
    assert report.summary.verdict == "consistent" and report.summary.llm_calls == 1
    request = FakeServer.requests[0]
    assert request["model"] == "fake-model" and request["temperature"] == 0
    assert request["messages"][0]["role"] == "system"


def test_unreachable_server_raises(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_HOST", "http://127.0.0.1:9")
    backend = get_backend("lmstudio:any")
    backend.timeout = 2
    with pytest.raises(LLMError):
        backend.complete("system", "user")


def test_backend_specs():
    assert get_backend("offline").is_llm is False
    assert get_backend("ollama:qwen3:32b").name == "ollama:qwen3:32b"
    with pytest.raises(ValueError):
        get_backend("ollama")
    with pytest.raises(ValueError):
        get_backend("gpt5")
