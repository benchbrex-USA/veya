"""Each Wrangler config maps to exactly one Worker and they cannot collide.

Workers Builds deploys:
  main                       -> `veya`            via `npx wrangler deploy`          (wrangler.jsonc)
  deploy/cloudflare-staging  -> `veya-staging`    via `--config wrangler.staging.jsonc`
  production/veya-dynamics   -> `veya-production` via `--config wrangler.production.jsonc`

If the root config ever names a Container Worker again, a merge to main would
replace the customer edge proxy. These tests make that a CI failure.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "release" / "manifest.json"


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


def load(name):
    return json.loads(strip_jsonc((ROOT / name).read_text(encoding="utf-8")))


def manifest_sha256():
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def test_config_names_map_one_to_one_to_workers():
    names = {
        "wrangler.jsonc": "veya",
        "wrangler.staging.jsonc": "veya-staging",
        "wrangler.production.jsonc": "veya-production",
    }
    for file_name, worker in names.items():
        assert load(file_name)["name"] == worker, file_name
    assert len(set(names.values())) == 3


def test_root_config_is_the_plain_customer_proxy():
    config = load("wrangler.jsonc")
    assert config["main"] == "worker/veya-edge.mjs"
    assert (ROOT / "worker" / "veya-edge.mjs").is_file()
    for key in ("containers", "durable_objects", "exports", "d1_databases", "r2_buckets", "ratelimits", "assets"):
        assert key not in config, "root wrangler.jsonc must not carry %s" % key
    assert "VEYA_MANIFEST_SHA256" not in config.get("vars", {})


def test_container_configs_do_not_share_applications_or_entrypoints():
    staging = load("wrangler.staging.jsonc")
    production = load("wrangler.production.jsonc")
    assert staging["main"] == "worker/veya-container.mjs"
    assert production["main"] == "worker/veya-production.mjs"
    staging_apps = {item["name"] for item in staging["containers"]}
    production_apps = {item["name"] for item in production["containers"]}
    assert staging_apps == {"veya-runtime"}
    assert production_apps == {"veya-production-runtime"}
    assert not staging_apps & production_apps


def test_rate_limit_namespaces_are_real_and_distinct():
    seen = {}
    for file_name in ("wrangler.staging.jsonc", "wrangler.production.jsonc"):
        for limit in load(file_name)["ratelimits"]:
            namespace = limit["namespace_id"]
            assert re.fullmatch(r"[0-9]+", namespace), (file_name, namespace)
            assert namespace not in seen, "namespace %s reused by %s and %s" % (namespace, seen.get(namespace), file_name)
            seen[namespace] = file_name


def test_no_placeholders_in_any_wrangler_config():
    for path in ROOT.glob("wrangler*.jsonc"):
        text = path.read_text(encoding="utf-8")
        assert "REPLACE_ME" not in text, path.name
        assert "<" not in strip_jsonc(text).replace("<=", ""), path.name


def test_release_pins_agree_everywhere():
    digest = manifest_sha256()
    pins = json.loads((ROOT / "deploy" / "cloudflare" / "release-pins.json").read_text(encoding="utf-8"))
    assert pins["VEYA_MANIFEST_SHA256"] == digest
    for file_name in ("wrangler.staging.jsonc", "wrangler.production.jsonc"):
        variables = load(file_name)["vars"]
        assert variables["VEYA_MANIFEST_SHA256"] == digest, file_name
        assert variables["VEYA_RELEASE_ID"] == pins["VEYA_RELEASE_ID"], file_name
        assert variables["VEYA_CHECKPOINT_SHA256"] == pins["VEYA_CHECKPOINT_SHA256"], file_name
    dockerfile = (ROOT / "deploy" / "cloudflare" / "Dockerfile").read_text(encoding="utf-8")
    assert "VEYA_MANIFEST_SHA256=%s" % digest in dockerfile
    build_meta = (ROOT / "worker" / "build-meta.mjs").read_text(encoding="utf-8")
    assert 'MANIFEST_SHA256 = "%s"' % digest in build_meta
    script = (ROOT / "scripts" / "write_deploy_meta.py").read_text(encoding="utf-8")
    assert 'AUDITED_MANIFEST_SHA256 = "%s"' % digest in script


def test_deploy_workflow_targets_only_veya_production():
    workflow = (ROOT / ".github" / "workflows" / "deploy-production.yml").read_text(encoding="utf-8")
    assert "WRANGLER_CONFIG: wrangler.production.jsonc" in workflow
    assert "workflow_dispatch" in workflow
    assert "prod-v*" in workflow
    # never deploys on an ordinary push to a branch, and never names the other configs
    assert "branches:" not in workflow
    assert "wrangler.staging.jsonc" not in workflow
    assert "wrangler.jsonc" not in workflow.replace("wrangler.production.jsonc", "")
    commands = [
        line for line in workflow.splitlines()
        if re.search(r"\bwrangler (deploy|versions|deployments|rollback)\b", line) and not line.strip().startswith(("#", "description:", "echo "))
    ]
    assert commands, "workflow runs no wrangler commands"
    for line in commands:
        assert '--config "$WRANGLER_CONFIG"' in line or line.rstrip().endswith("\\"), line
    assert "secrets.CLOUDFLARE_API_TOKEN" in workflow
    assert "wrangler versions upload" in workflow and "wrangler versions deploy" in workflow
    for pct in ("10", "50"):
        assert "for pct in 10 50" in workflow, pct
