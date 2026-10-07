"""The staging Worker routes to one Cloudflare Container and does not use an external origin.

wrangler.jsonc is the plain `veya` edge proxy built from main (see
tests/test_wrangler_configs.py); the Container staging config is wrangler.staging.jsonc.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "wrangler.staging.jsonc"
ACTIVE = [
    ROOT / "wrangler.staging.jsonc",
    ROOT / "wrangler.production.jsonc",
    ROOT / "worker" / "veya-production.mjs",
    ROOT / "worker" / "production_edge.mjs",
    ROOT / "worker" / "veya-container.mjs",
    ROOT / "worker" / "edge_policy.mjs",
    ROOT / "worker" / "artifact_route.mjs",
    ROOT / "deploy" / "cloudflare" / "Dockerfile",
    ROOT / "deploy" / "cloudflare" / "fetch_release.py",
]


def strip_jsonc(text: str) -> str:
    out = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and text[index : index + 2] == "//":
            newline = text.find("\n", index)
            if newline < 0:
                break
            index = newline
            continue
        out.append(char)
        index += 1
    return "".join(out)


def test_wrangler_binds_one_container_and_has_no_origin():
    config = json.loads(strip_jsonc(CONFIG_PATH.read_text(encoding="utf-8")))
    assert config["name"] == "veya-staging"
    assert config["account_id"] == "8d0a6a64de03f4c9b14184398e75aa6e"
    assert config["main"] == "worker/veya-container.mjs"
    assert "VEYA_ORIGIN" not in config.get("vars", {})
    assert "VEYA_EDGE_TOKEN" not in config.get("vars", {})
    container = config["containers"][0]
    assert container["name"] == "veya-runtime"
    assert container["class_name"] == "VeyaContainer"
    assert container["scheduling_policy"] == "default"
    assert container["image"] == "./deploy/cloudflare/Dockerfile"
    assert container["image_build_context"] == "."
    assert container["instance_type"] == "standard-2"
    assert container["max_instances"] == 1
    assert "images" not in container
    assert config["durable_objects"]["bindings"] == [
        {"name": "VEYA_CONTAINER", "class_name": "VeyaContainer"}
    ]
    assert config["r2_buckets"] == [
        {"binding": "VEYA_ARTIFACTS", "bucket_name": "veya-model-artifacts"}
    ]
    assert config["secrets"]["required"] == ["VEYA_EDGE_TOKEN", "VEYA_DEMO_TOKEN"]
    assert config["ratelimits"][0]["simple"] == {"limit": 20, "period": 60}
    assert config["ratelimits"][1]["simple"] == {"limit": 6, "period": 60}
    assert "VEYA_SOURCE_COMMIT" not in config.get("vars", {})
    assert "VEYA_SOURCE_DIRTY" not in config.get("vars", {})
    assert config["assets"]["directory"] == "./site/dist"
    assert config["assets"]["not_found_handling"] == "single-page-application"
    assert "/api/*" in config["assets"]["run_worker_first"]
    assert len(config["containers"]) == 1
    assert config["exports"]["VeyaContainer"]["container"] == "veya-runtime"
    assert config["exports"]["VeyaContainer"]["storage"] == "sqlite"
    # The staging and production Workers never use the tunnel proxy entrypoint.
    assert config["main"] != "worker/veya-edge.mjs"
    dockerfile = (ROOT / "deploy" / "cloudflare" / "Dockerfile").read_text(encoding="utf-8")
    assert "R2_SECRET_ACCESS_KEY" not in dockerfile
    assert "AWS_SECRET_ACCESS_KEY" not in dockerfile
    assert "COPY models/production/model.safetensors" not in dockerfile
    assert "! -name '*.safetensors'" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.14.0" in dockerfile
    assert dockerfile.index("torch==2.14.0") < dockerfile.index("pip install --no-cache-dir -r requirements.lock")
    assert "COPY web/static/ ./web/static/" in dockerfile
    assert "test -f /app/web/static/site-config.json" in dockerfile
    assert "COPY experts/" not in dockerfile
    assert "COPY quantum/" not in dockerfile
    assert 'CMD ["python", "/app/boot/supervise.py"]' in dockerfile
    assert "sh" not in dockerfile.split("CMD", 1)[1]
    playground = (ROOT / "site" / "src" / "console.tsx").read_text(encoding="utf-8")
    assert 'fetch("/api/infer"' in playground
    assert "/api/playground" not in playground
    assert "VEYA_EDGE_TOKEN" not in playground
    assert "sessionStorage" not in playground
    assert "<project key>" not in (ROOT / "site" / "src" / "marketing.tsx").read_text(encoding="utf-8")
    worker = (ROOT / "worker" / "veya-container.mjs").read_text(encoding="utf-8")
    assert "export { ContainerProxy }" in worker
    assert 'sleepAfter = "8h"' in worker
    assert 'pingEndpoint = "127.0.0.1/readyz"' in worker
    assert "enableInternet = false" in worker
    assert "outboundByHost" in worker
    assert "handleArtifact" not in worker.split("export default", 1)[1]
    headers = (ROOT / "site" / "public" / "_headers").read_text(encoding="utf-8")
    assert "Content-Security-Policy:" in headers
    assert "unsafe-inline" not in headers
    assert "X-Robots-Tag: noindex" in headers
    assert "X-Frame-Options: DENY" in headers
    assert not (ROOT / "site" / "src" / "worker.ts").exists()
    assert not (ROOT / "site" / "wrangler.jsonc").exists()
    assert "not part of the investor deployment" in (ROOT / "site" / "future" / "README.md").read_text(encoding="utf-8")
    gateway = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert "from experts" not in gateway
    assert "from quantum" not in gateway
    assert "SystemOne is not served by this release." in gateway


def test_active_deploy_files_do_not_name_a_tunnel():
    needles = ("tailscale", "tailnet", "taila8b570", "trycloudflare", "cloudflared", "VEYA_ORIGIN")
    for path in ACTIVE:
        text = path.read_text(encoding="utf-8").lower()
        for needle in needles:
            assert needle.lower() not in text, path
