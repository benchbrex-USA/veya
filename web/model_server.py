"""Start the audited Veya model service used by the website."""

import os
from pathlib import Path

from veya.release_bundle import load_release_manifest
from veya.server import serve

root = Path(__file__).resolve().parent.parent
manifest = root / "artifacts" / "release" / "manifest.json"
bundle, _ = load_release_manifest(
    str(manifest), device=os.environ.get("VEYA_DEVICE", "cpu"), root=str(root)
)
port = int(os.environ.get("VEYA_MODEL_PORT", "8080"))
server = serve(bundle.agent, host="127.0.0.1", port=port, bundle=bundle)
print(f"Veya audited model service: http://127.0.0.1:{port}", flush=True)
server.serve_forever()
