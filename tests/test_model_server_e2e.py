"""End-to-end production startup test: release manifest, bundle, and model server.

Covers the two paths that had no test coverage:
  1. `load_release_manifest` (veya/release_bundle.py) — the audited startup path
     that verifies every file hash before attaching calibration and policy.
  2. `python3 -m web.model_server` (web/model_server.py) — the module-level
     startup script that binds the audited model service, exercised in a
     subprocess over real HTTP.

Run:  python3 tests/test_model_server_e2e.py [repo_root]
Needs the production checkpoint (models/production) and artifacts/release/
manifest.json, as shipped in this repository. No network required.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ != "__main__":
    import pytest
    pytest.skip(
        "real-checkpoint integration; run python3 tests/test_model_server_e2e.py",
        allow_module_level=True,
    )

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
MANIFEST = os.path.join(ROOT, "artifacts", "release", "manifest.json")
DEVICE = os.environ.get("VEYA_DEVICE", "cpu")

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def section(title):
    print("\n" + "=" * 78)
    print("  " + title)
    print("=" * 78)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
print("Loading release manifest (verifies every runtime and component hash)...")
started = time.time()
from veya.release_bundle import load_release_manifest  # noqa: E402

timings = {}
bundle, manifest = load_release_manifest(MANIFEST, device=DEVICE, root=ROOT, timings=timings)
ok("manifest/load_release_manifest", bundle is not None and manifest is not None)
ok("manifest/format v2", manifest["format"] == "veya-release-manifest-v2")
ok("manifest/checkpoint hash recorded", bool(bundle.checkpoint_hash))
ok("manifest/shortlist sha recorded", bool(bundle.shortlist_sha256))
ok("manifest/selective policy attached", bundle.selective_policy is not None)
ok("manifest/route temperature positive", bundle.route_temperature > 0)
ok("manifest/manifest sha recorded", bool(bundle.manifest_sha256))
print("  loaded in %.1fs (%s)" % (time.time() - started, ", ".join(
    "%s=%.1fs" % (k, v) for k, v in timings.items())))

# ---------------------------------------------------------------------------
section("1. ReleaseBundle.choice planning and scoring")
ok("plan/19 packed", bundle.plan(19)["path"] == "packed")
ok("plan/20 shortlist K=8",
   bundle.plan(20)["path"] == "shortlist" and bundle.plan(20)["k"] == 8)
ok("plan/0 refused", _raises(lambda: bundle.plan(0)))
ok("plan/bool refused", _raises(lambda: bundle.plan(True)))

small = {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}
scored = bundle.score_choice("Duplicate charge on invoice #4411", "Which label best fits?", small)
ok("score_choice/packed path", scored["path"] == "packed")
ok("score_choice/probabilities sum to 1", abs(scored["probabilities"].sum() - 1.0) < 1e-6)
ok("score_choice/prediction in labels", scored["prediction"] in scored["labels"])
ok("score_choice/no shortlist", scored["shortlist"] is None)

wide = {("opt%02d" % i): "option %d" % i for i in range(25)}
scored_wide = bundle.score_choice("Pick something", "Which label best fits?", wide)
ok("score_choice/25 options shortlist", scored_wide["path"] == "shortlist", scored_wide["path"])
ok("score_choice/shortlist has K=8", len(scored_wide["labels"]) == 8)
ok("score_choice/probabilities over shortlist", abs(scored_wide["probabilities"].sum() - 1.0) < 1e-6)

public = bundle.public_choice("Duplicate charge on invoice #4411", "Which label best fits?", small)
ok("public_choice/answers", public["decision"] == "answer" and not public["deferred"])
ok("public_choice/probabilities dict", isinstance(public["probabilities"], dict)
   and abs(sum(public["probabilities"].values()) - 1.0) < 0.001)
ok("public_choice/confidence in [0,1]", 0.0 <= public["confidence"] <= 1.0)
ok("public_choice/packed route", public["route"] == "packed")
ok("public_choice/usage tokens", public["usage"]["input_tokens"] > 0)

mixed = {
    "intent": {"type": "choice", "instructions": "Which intent?",
               "criteria": {"refund": "money back", "bug": "a defect"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "frustration": {"type": "score", "instructions": "How frustrated?",
                    "criteria": ["calm", "annoyed", "angry"]},
}
answers = bundle.answer_questions("Please refund the duplicate charge today.", mixed)
ok("answer_questions/answers all questions", set(answers["answers"]) == set(mixed))
ok("answer_questions/routes recorded", set(answers["routes"]) == set(mixed))
ok("answer_questions/packed choice route", answers["routes"]["intent"] == "packed")
ok("answer_questions/noul packed", answers["routes"]["urgent"] == "noul")
ok("answer_questions/score packed", answers["routes"]["frustration"] == "score")
ok("answer_questions/noul probability in [0,1]",
   0.0 <= answers["answers"]["urgent"]["noul"] <= 1.0)

# ---------------------------------------------------------------------------
section("2. web/model_server.py subprocess: audited model service over HTTP")
port = free_port()
env = dict(os.environ, VEYA_MODEL_PORT=str(port), VEYA_DEVICE=DEVICE)
proc = subprocess.Popen(
    [sys.executable, "-m", "web.model_server"], cwd=ROOT, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
try:
    base = "http://127.0.0.1:%d" % port
    deadline = time.time() + 300
    banner = None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "audited model service" in line:
            banner = line.strip()
            break
        if proc.poll() is not None:
            break
    ok("server/startup banner", banner is not None, "process exited early")
    if banner:
        print("  " + banner)

    health = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                health = json.loads(response.read())
            break
        except (urllib.error.URLError, socket.timeout, ConnectionResetError):
            time.sleep(2)
    ok("server/health ok", health is not None and health.get("status") == "ok", str(health))

    with urllib.request.urlopen(base + "/version", timeout=10) as response:
        version = json.loads(response.read())
    ok("server/version has model_version", version.get("model_version") == "0.3.5", str(version.get("model_version")))
    ok("server/version checkpoint sha", version.get("checkpoint_sha256") == bundle.checkpoint_hash)
    ok("server/version shortlist sha", version.get("shortlist_sha256") == bundle.shortlist_sha256)
    ok("server/version release metadata", version.get("release") == manifest["release"])

    request = json.dumps({
        "state": "Duplicate charge on invoice #4411",
        "instructions": "Which label best fits?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"},
    }).encode()
    req = urllib.request.Request(base + "/v1/choice", data=request,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read())
    ok("server/choice answers", result.get("decision") == "answer", str(result.get("decision")))
    ok("server/choice primitive", result.get("primitive") == "choice")
    ok("server/choice probabilities", isinstance(result.get("probabilities"), dict)
       and abs(sum(result["probabilities"].values()) - 1.0) < 0.001)
    ok("server/choice calibration_version", result.get("calibration_version") is not None)
    ok("server/choice packed route", result.get("route") == "packed")

    systemone = json.dumps({
        "state": "My payment failed twice, please fix it.",
        "questions": {
            "intent": {"type": "choice", "instructions": "Which intent?",
                       "criteria": {"billing": "payments", "technical": "bugs"}},
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        },
    }).encode()
    req = urllib.request.Request(base + "/v1/systemone", data=systemone,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read())
    ok("server/systemone answers both", set(result.get("answers") or {}) == {"intent", "urgent"})
    ok("server/systemone routes", set(result.get("routes") or {}) == {"intent", "urgent"})

    try:
        urllib.request.urlopen(base + "/v1/nope", timeout=10)
        ok("server/404 for unknown route", False)
    except urllib.error.HTTPError as exc:
        ok("server/404 for unknown route", exc.code == 404)

    req = urllib.request.Request(base + "/v1/choice", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        ok("server/invalid json -> 400", False)
    except urllib.error.HTTPError as exc:
        ok("server/invalid json -> 400", exc.code == 400)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()

# ---------------------------------------------------------------------------
section("SUMMARY")
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for item in FAIL:
    print("   FAIL", item)
raise SystemExit(1 if FAIL else 0)
