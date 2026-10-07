"""Authenticated checkpoint download for the Cloudflare Container boot path."""

import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

CHECKPOINT_BYTES = b"veya-checkpoint-bytes"
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "hf-space"
    / "space"
    / "fetch_checkpoint.py"
)


def load_fetch_module():
    spec = importlib.util.spec_from_file_location("veya_fetch_checkpoint", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointServer:
    def __init__(self, body: bytes):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parent.authorization = self.headers.get("Authorization")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self.authorization = None
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/model.safetensors"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture()
def checkpoint_server():
    server = CheckpointServer(CHECKPOINT_BYTES)
    yield server
    server.close()


def write_manifest(directory: Path, digest: str) -> Path:
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "base_checkpoint": str(directory / "models"),
                "base_checkpoint_hash": digest,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_download_from_url_sends_bearer_and_atomically_writes(tmp_path, checkpoint_server):
    module = load_fetch_module()
    target = tmp_path / "model.safetensors"
    downloaded = module.download_from_url(checkpoint_server.url, "secret-token", target)
    assert downloaded == target
    assert target.read_bytes() == CHECKPOINT_BYTES
    assert checkpoint_server.authorization == "Bearer secret-token"
    assert not (tmp_path / "model.safetensors.part").exists()


def test_main_rejects_url_download_with_wrong_digest(tmp_path, monkeypatch, checkpoint_server, capsys):
    module = load_fetch_module()
    digest = hashlib.sha256(b"different-bytes").hexdigest()
    manifest = write_manifest(tmp_path, digest)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setenv("VEYA_CHECKPOINT_URL", checkpoint_server.url)
    monkeypatch.setenv("VEYA_CHECKPOINT_TOKEN", "secret-token")
    monkeypatch.delenv("VEYA_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert module.main() == 1
    assert checkpoint_server.authorization == "Bearer secret-token"
    assert not (tmp_path / "models" / "model.safetensors").exists()
    assert not (tmp_path / "models" / "model.safetensors.part").exists()
    captured = capsys.readouterr()
    assert "secret-token" not in captured.out
    assert "secret-token" not in captured.err
