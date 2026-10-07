"""Boot fetch verifies the checkpoint before the model service can see it."""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import importlib.util

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "veya_fetch_release",
    Path(__file__).resolve().parents[1] / "deploy" / "cloudflare" / "fetch_release.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ArtifactError = _MODULE.ArtifactError
assert_no_storage_credentials = _MODULE.assert_no_storage_credentials
fetch_release = _MODULE.fetch_release

BODY = b"veya-checkpoint-bytes"
DIGEST = hashlib.sha256(BODY).hexdigest()


class _Server:
    def __init__(self, body: bytes, status: int = 200, location: str | None = None):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parent.authorization = self.headers.get("Authorization")
                if location:
                    self.send_response(302)
                    self.send_header("Location", location)
                    self.end_headers()
                    return
                self.send_response(status)
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
        return f"http://{host}:{port}/releases/0.3.5-audit-baseline.1/model.safetensors"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture()
def checkpoint_server():
    server = _Server(BODY)
    yield server
    server.close()


def test_fetch_release_verifies_then_renames(tmp_path, checkpoint_server):
    destination = tmp_path / "model.safetensors"
    fetch_release(
        checkpoint_server.url,
        destination,
        DIGEST,
        len(BODY),
        allowed_hosts={"127.0.0.1"},
    )
    assert destination.read_bytes() == BODY
    assert not (tmp_path / "model.safetensors.part").exists()
    assert checkpoint_server.authorization is None


def test_fetch_release_deletes_a_partial_on_hash_mismatch(tmp_path, checkpoint_server):
    destination = tmp_path / "model.safetensors"
    with pytest.raises(ArtifactError):
        fetch_release(
            checkpoint_server.url,
            destination,
            "0" * 64,
            len(BODY),
            allowed_hosts={"127.0.0.1"},
        )
    assert not destination.exists()
    assert not (tmp_path / "model.safetensors.part").exists()


def test_fetch_release_refuses_a_redirect(tmp_path):
    server = _Server(BODY, location="http://example.test/checkpoint")
    try:
        with pytest.raises(ArtifactError):
            fetch_release(
                server.url,
                tmp_path / "model.safetensors",
                DIGEST,
                len(BODY),
                allowed_hosts={"127.0.0.1"},
            )
    finally:
        server.close()
    assert not (tmp_path / "model.safetensors").exists()


def test_fetch_release_refuses_an_unapproved_host(tmp_path):
    with pytest.raises(ArtifactError):
        fetch_release(
            "http://example.test/model.safetensors",
            tmp_path / "model.safetensors",
            DIGEST,
            len(BODY),
        )


def test_storage_credentials_fail_closed():
    with pytest.raises(ArtifactError):
        assert_no_storage_credentials({"R2_SECRET_ACCESS_KEY": "present"})
