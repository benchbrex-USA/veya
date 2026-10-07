"""Upload one checkpoint through the temporary Worker. The token stays in a local file."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PART_SIZE = 16 * 1024 * 1024
EXPECTED = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
WORKERS = 4


def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    candidates = [os.environ.get("SSL_CERT_FILE") or "", "/etc/ssl/cert.pem"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            context.load_verify_locations(candidate)
            return context
    import certifi

    context.load_verify_locations(certifi.where())
    return context


TLS = tls_context()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request(url: str, token: str, method: str, body: bytes | None = None, headers: dict[str, str] | None = None):
    actual = {
        "authorization": f"Bearer {token}",
        "cache-control": "no-store",
        "user-agent": "veya-checkpoint-upload",
    }
    if headers:
        actual.update(headers)
    req = urllib.request.Request(url, data=body, headers=actual, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300, context=TLS) as response:
            payload = response.read()
            return response.status, dict(response.headers), payload
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def request_retry(url: str, token: str, method: str, body: bytes | None = None, headers: dict[str, str] | None = None):
    delay = 1
    for attempt in range(5):
        try:
            status, response_headers, payload = request(url, token, method, body, headers)
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = str(exc)
            if "Bearer" not in reason and "UPLOAD" not in reason:
                print(f"{method} transport error: {reason[:240]}", flush=True)
            status, response_headers, payload = 0, {}, b""
        if status in (200, 201):
            return status, response_headers, payload
        if status not in (0, 429, 500, 502, 503, 504) or attempt == 4:
            detail = payload.decode("utf-8", "replace")[:180]
            if "Bearer" in detail:
                detail = "redacted"
            raise SystemExit(f"{method} failed status={status} bytes={len(payload)} detail={detail}")
        time.sleep(delay)
        delay *= 2
    raise SystemExit("retry exhausted")


def upload_part(base: str, token: str, upload_id: str, path: Path, part_number: int, offset: int, size: int):
    with path.open("rb") as handle:
        handle.seek(offset)
        body = handle.read(size)
    if len(body) != size:
        raise SystemExit(f"short read at part {part_number}")
    query = urllib.parse.urlencode({"uploadId": upload_id, "partNumber": str(part_number)})
    status, _, payload = request_retry(
        f"{base}/mpu/part?{query}",
        token,
        "PUT",
        body,
        {"content-type": "application/octet-stream"},
    )
    parsed = json.loads(payload)
    print(f"part {part_number} status={status} bytes={size}", flush=True)
    return {"partNumber": parsed["partNumber"], "etag": parsed["etag"]}


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: upload.py <workers-url> <token-file> <checkpoint>")
    base = sys.argv[1].rstrip("/")
    token = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
    path = Path(sys.argv[3])
    if not token or len(token) < 32:
        raise SystemExit("upload token file is unusable")
    size = path.stat().st_size
    digest = sha256_file(path)
    manifest = json.loads(Path("artifacts/release/manifest.json").read_text(encoding="utf-8"))
    if digest != EXPECTED or manifest["base_checkpoint_hash"] != EXPECTED:
        raise SystemExit("local checkpoint does not match the manifest")
    print(f"local size={size} sha256={digest}", flush=True)
    status, _, payload = request_retry(f"{base}/mpu/create", token, "POST", b"")
    upload_id = json.loads(payload)["uploadId"]
    print(f"created multipart status={status}", flush=True)
    ranges = []
    offset = 0
    number = 1
    while offset < size:
        chunk = min(PART_SIZE, size - offset)
        ranges.append((number, offset, chunk))
        offset += chunk
        number += 1
    parts = []
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [
                pool.submit(upload_part, base, token, upload_id, path, part_number, part_offset, part_size)
                for part_number, part_offset, part_size in ranges
            ]
            for future in as_completed(futures):
                parts.append(future.result())
        parts.sort(key=lambda item: item["partNumber"])
        status, _, payload = request_retry(
            f"{base}/mpu/complete?{urllib.parse.urlencode({'uploadId': upload_id})}",
            token,
            "POST",
            json.dumps(parts).encode(),
            {"content-type": "application/json"},
        )
        print(f"complete status={status} body={payload.decode()[:200]}", flush=True)
    except Exception:
        request(f"{base}/mpu/abort?{urllib.parse.urlencode({'uploadId': upload_id})}", token, "DELETE")
        raise
    remote = hashlib.sha256()
    remote_size = 0
    verify_req = urllib.request.Request(
        f"{base}/verify-object",
        headers={"authorization": f"Bearer {token}", "user-agent": "veya-checkpoint-upload"},
        method="GET",
    )
    with urllib.request.urlopen(verify_req, timeout=600, context=TLS) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            remote_size += len(chunk)
            remote.update(chunk)
        remote_digest = remote.hexdigest()
    print(f"remote size={remote_size} sha256={remote_digest}", flush=True)
    if remote_size != size or remote_digest != EXPECTED or digest != EXPECTED:
        raise SystemExit("remote object does not match the local checkpoint")
    print("R2_ARTIFACT_GATE=PASS", flush=True)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
