"""Fetch the audited checkpoint into the release tree, then prove its digest.

The 804 MB ``model.safetensors`` is deliberately absent from this Space. The
weights live in the private ``veyadynamics/veya`` model repo, and a Space has
to be public for the Cloudflare edge proxy to reach it without credentials;
committing the checkpoint as Space LFS files would turn a private research
release into a public download. So the container pulls exactly one file at
boot, from a pinned revision, and asserts the SHA-256 the release manifest
already pins.

This duplicates a check on purpose. ``veya.release_bundle.load_bundle`` hashes
the same file before it will load anything and raises ``checkpoint hash does
not match the manifest``. Verifying here first does not weaken that gate -- it
runs afterwards regardless -- it just reports a wrong or truncated download
with both digests and the revision it came from, instead of letting a 3-minute
cold start end in an opaque load failure.

Requires ``VEYA_HF_TOKEN`` or ``HF_TOKEN`` with read access to the checkpoint
repo. Hugging Face injects ``HF_TOKEN`` into every Space automatically, which
is sufficient when the Space and the model repo share an owner.
"""

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "release" / "manifest.json"
FILENAME = "model.safetensors"

DEFAULT_REPO = "veyadynamics/veya"
# The revision recorded in models/production/.cache/huggingface/download/
# model.safetensors.metadata on the machine that produced the audited release,
# whose LFS etag equals the manifest's base_checkpoint_hash. Pinning it means a
# later push to the model repo cannot silently change what this Space serves.
DEFAULT_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

CHUNK = 1024 * 1024


def log(message: str) -> None:
    print(f"[checkpoint] {message}", flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def token() -> str:
    for name in ("VEYA_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HF_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def download_from_url(url: str, credential: str, target: Path) -> Path:
    """Stream an authenticated checkpoint to ``target`` only after the body is complete."""
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {credential}"},
    )
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target


def verify_downloaded(target: Path, expected: str, started: float) -> int:
    elapsed = time.perf_counter() - started
    size_mb = target.stat().st_size / 1e6
    log(f"downloaded {size_mb:.1f} MB in {elapsed:.1f}s; hashing")
    digest = sha256_of(target)
    if digest != expected:
        log(f"SHA-256 MISMATCH: got {digest}, manifest pins {expected}")
        target.unlink()
        return 1
    log(f"verified sha256={digest} in {time.perf_counter() - started:.1f}s total")
    return 0


def main() -> int:
    if not MANIFEST.is_file():
        log(f"release manifest is missing: {MANIFEST}")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = str(manifest.get("base_checkpoint_hash") or "")
    if len(expected) != 64:
        log("manifest does not pin a base_checkpoint_hash; refusing to guess")
        return 1

    model_dir = (ROOT / str(manifest["base_checkpoint"])).resolve()
    target = model_dir / FILENAME
    model_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        digest = sha256_of(target)
        if digest == expected:
            log(f"{FILENAME} already present and verified ({target.stat().st_size / 1e6:.1f} MB)")
            return 0
        log(f"baked {FILENAME} hashes to {digest}, not the pinned {expected}; discarding it")
        target.unlink()

    artifact_url = os.environ.get("VEYA_CHECKPOINT_URL", "").strip()
    if artifact_url:
        credential = os.environ.get("VEYA_CHECKPOINT_TOKEN", "")
        if not credential:
            log("VEYA_CHECKPOINT_URL is set but VEYA_CHECKPOINT_TOKEN is missing")
            return 1
        log("downloading checkpoint from the configured artifact URL")
        started = time.perf_counter()
        try:
            download_from_url(artifact_url, credential, target)
        except Exception as exc:
            log(f"download failed: {type(exc).__name__}")
            partial = target.with_name(target.name + ".part")
            if partial.exists():
                partial.unlink()
            return 1
        return verify_downloaded(target, expected, started)

    credential = token()
    if not credential:
        log("no read token available; set the VEYA_HF_TOKEN Space secret")
        return 1
    # huggingface_hub reads the environment too, so both paths agree even if a
    # future version changes how the explicit argument is threaded through.
    os.environ["HF_TOKEN"] = credential

    repo = os.environ.get("VEYA_CHECKPOINT_REPO", DEFAULT_REPO)
    revision = os.environ.get("VEYA_CHECKPOINT_REVISION", DEFAULT_REVISION)
    log(f"downloading {repo}/{FILENAME} @ {revision[:12]}")
    started = time.perf_counter()
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                filename=FILENAME,
                repo_type="model",
                revision=revision,
                token=credential,
                local_dir=str(model_dir),
                local_files_only=False,
            )
        )
    except Exception as exc:  # network, auth, and missing-revision all land here
        log(f"download failed: {type(exc).__name__}: {exc}")
        return 1

    if downloaded.resolve() != target.resolve():
        shutil.move(str(downloaded), str(target))
    return verify_downloaded(target, expected, started)


if __name__ == "__main__":
    sys.exit(main())
