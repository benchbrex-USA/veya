"""Start the audited Veya model service used by the website."""

import os
from pathlib import Path

import torch

from veya.release_bundle import load_release_manifest
from veya.server import serve


def cpu_threads(environ=os.environ) -> int:
    """Torch intra-op threads: VEYA_TORCH_THREADS when set, else the CPUs this process may use."""
    raw = str(environ.get("VEYA_TORCH_THREADS", "")).strip()
    if raw.isdigit() and int(raw) > 0:
        return min(int(raw), 64)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


# Pin intra-op threads to the CPUs the container actually has (VEYA_TORCH_THREADS overrides).
torch.set_num_threads(cpu_threads())

root = Path(__file__).resolve().parent.parent
manifest = root / "artifacts" / "release" / "manifest.json"
bundle, _ = load_release_manifest(
    str(manifest), device=os.environ.get("VEYA_DEVICE", "cpu"), root=str(root)
)
port = int(os.environ.get("VEYA_MODEL_PORT", "8080"))
server = serve(bundle.agent, host="127.0.0.1", port=port, bundle=bundle)
print(f"Veya audited model service: http://127.0.0.1:{port}", flush=True)
server.serve_forever()
