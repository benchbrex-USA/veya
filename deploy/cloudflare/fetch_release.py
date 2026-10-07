"""Stream the production checkpoint from the private R2 hostname and verify it.

The container has no R2 access key. A hash or size mismatch deletes the partial
file and fails closed before the model service starts.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ARTIFACT_HOST = "veya-r2.internal"
FORBIDDEN_ENV = (
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "VEYA_ARTIFACT_TOKEN",
)


class ArtifactError(RuntimeError):
    """The checkpoint is missing, redirected, or does not match the release."""


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ArtifactError("redirect refused")


def assert_no_storage_credentials(environ: dict[str, str] | None = None) -> None:
    source = os.environ if environ is None else environ
    present = [name for name in FORBIDDEN_ENV if source.get(name)]
    if present:
        raise ArtifactError("storage credential is present in the container")


def fetch_release(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
    *,
    allowed_hosts: set[str] | None = None,
) -> None:
    """Download url to destination.part, verify it, then rename it into place."""
    host = urlsplit(url).hostname
    allowed = {ARTIFACT_HOST} if allowed_hosts is None else allowed_hosts
    if host not in allowed or urlsplit(url).scheme != "http":
        raise ArtifactError("checkpoint host is not approved")
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or partial.is_symlink():
        raise ArtifactError("checkpoint path is a symlink")
    opener = urllib.request.build_opener(_RefuseRedirect)
    request = urllib.request.Request(url, method="GET")
    digest = hashlib.sha256()
    size = 0
    try:
        with opener.open(request, timeout=120) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > expected_size:
                    raise ArtifactError("checkpoint is larger than the release")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        actual = digest.hexdigest()
        if size != expected_size or actual != expected_sha256:
            raise ArtifactError("checkpoint does not match the release")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        if partial.exists():
            partial.unlink(missing_ok=True)


def main() -> None:
    assert_no_storage_credentials()
    url = os.environ["VEYA_CHECKPOINT_URL"]
    expected = os.environ["VEYA_CHECKPOINT_SHA256"]
    size = int(os.environ["VEYA_CHECKPOINT_BYTES"])
    destination = Path(os.environ.get("VEYA_MODEL_PATH", "/app/models/production/model.safetensors"))
    fetch_release(url, destination, expected, size)


if __name__ == "__main__":
    main()
