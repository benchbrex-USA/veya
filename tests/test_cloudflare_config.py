"""The Worker routes to one Cloudflare Container and does not use an external origin."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "wrangler.jsonc"
ACTIVE = [
    ROOT / "wrangler.jsonc",
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
    assert config["secrets"]["required"] == ["VEYA_EDGE_TOKEN"]
    assert config["exports"]["VeyaContainer"]["container"] == "veya-runtime"
    assert config["exports"]["VeyaContainer"]["storage"] == "sqlite"
    assert not (ROOT / "worker" / "veya-edge.mjs").exists()
    dockerfile = (ROOT / "deploy" / "cloudflare" / "Dockerfile").read_text(encoding="utf-8")
    assert "R2_SECRET_ACCESS_KEY" not in dockerfile
    assert "AWS_SECRET_ACCESS_KEY" not in dockerfile
    assert "COPY models/production/model.safetensors" not in dockerfile
    assert "! -name '*.safetensors'" in dockerfile
    worker = (ROOT / "worker" / "veya-container.mjs").read_text(encoding="utf-8")
    assert "enableInternet = false" in worker
    assert "outboundByHost" in worker
    assert "handleArtifact" not in worker.split("export default", 1)[1]


def test_active_deploy_files_do_not_name_a_tunnel():
    needles = ("tailscale", "tailnet", "taila8b570", "trycloudflare", "cloudflared", "VEYA_ORIGIN")
    for path in ACTIVE:
        text = path.read_text(encoding="utf-8").lower()
        for needle in needles:
            assert needle.lower() not in text, path
