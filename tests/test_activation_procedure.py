"""Validation suite for the selective automation activation procedure.

artifacts/release/activation_procedure.json documents the conditional path for
enabling selective automation (currently BLOCKED_EXTERNAL on the G6/G8
prospective blind gates). These tests pin the procedure to the actual
repository state so any drift fails loudly:

- the procedure schema and required fields are present and well formed;
- the selective policy is still disabled (operating_epsilon "disabled",
  manifest automation.status "DISABLED");
- every hash recorded in current_state matches the actual artifact files
  (policy, manifest, runtime audit) and the manifest still pins the policy;
- all prerequisite gates (G6, G8, P1-P6) are listed with PASS required;
- the procedure requires BOTH CI tiers (release-candidate gate on a v* tag
  push and the dispatch-only production-evidence gate);
- rollback returns operating_epsilon to "disabled".

The suite intentionally does not enable anything; it only validates the
documentation artifact against the frozen release.
"""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "release"
PROCEDURE_PATH = ARTIFACTS_DIR / "activation_procedure.json"
POLICY_PATH = ARTIFACTS_DIR / "selective_policy.json"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
RUNTIME_AUDIT_PATH = ARTIFACTS_DIR / "runtime_audit.json"
RELEASE_GATE_PATH = ROOT / ".github" / "workflows" / "release-gate.yml"

EXPECTED_GATES = {
    "G6",
    "G8",
    "P1_data_validity",
    "P2_selective_safety",
    "P3_useful_coverage",
    "P4_shift_robustness",
    "P5_calibration_transfer",
    "P6_operational_viability",
}

EXPECTED_BINDINGS = (
    "route",
    "scorer_id",
    "score",
    "checkpoint_hash",
    "shortlist_sha256",
    "chosen_on",
)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _procedure() -> dict:
    return _load_json(PROCEDURE_PATH)


def _steps() -> dict:
    return {step["order"]: step for step in _procedure()["activation_steps"]}


# --------------------------------------------------------------------------
# Schema / required fields
# --------------------------------------------------------------------------


def test_procedure_exists_and_is_ascii():
    raw = PROCEDURE_PATH.read_text(encoding="utf-8")
    # Guards against Unicode confusable characters (e.g. Cyrillic lookalikes)
    # sneaking into paths and module references.
    assert raw.isascii(), "activation_procedure.json contains non-ASCII characters"
    json.loads(raw)  # round-trips as strict JSON


def test_procedure_format_and_blocked_status():
    procedure = _procedure()
    assert procedure["format"] == "veya-selective-activation-procedure-v1"
    assert procedure["status"] == "BLOCKED_EXTERNAL"
    blocked_on = procedure["blocked_on"]
    assert "G6" in blocked_on and "G8" in blocked_on
    assert "PASS" in blocked_on
    assert "target-domain" in blocked_on


def test_procedure_required_top_level_fields():
    procedure = _procedure()
    for key in (
        "format",
        "status",
        "blocked_on",
        "current_state",
        "prerequisites",
        "activation_steps",
        "rollback",
    ):
        assert key in procedure, "missing top-level field: %s" % key


def test_current_state_required_fields():
    current = _procedure()["current_state"]
    for key in (
        "operating_epsilon",
        "automation",
        "policy_file",
        "policy_sha256",
        "manifest_file",
        "manifest_sha256",
        "policy_bindings",
        "thresholds",
    ):
        assert key in current, "missing current_state field: %s" % key


def test_current_state_points_at_real_files():
    current = _procedure()["current_state"]
    assert current["policy_file"] == "artifacts/release/selective_policy.json"
    assert current["manifest_file"] == "artifacts/release/manifest.json"
    assert (ROOT / current["policy_file"]).is_file()
    assert (ROOT / current["manifest_file"]).is_file()


def test_activation_steps_ordered_and_well_formed():
    steps = _procedure()["activation_steps"]
    orders = [step["order"] for step in steps]
    assert orders == [1, 2, 3, 4, 5, 6]
    for step in steps:
        assert step["action"], "step %s has no action" % step["order"]
        assert step.get("acceptance"), "step %s has no acceptance criteria" % step["order"]
        assert (
            step.get("command")
            or step.get("workflow")
            or step.get("procedure")
            or step.get("commands")
        ), "step %s documents no executable path" % step["order"]


def test_rollback_required_fields():
    rollback = _procedure()["rollback"]
    for key in ("action", "trigger", "steps"):
        assert key in rollback, "missing rollback field: %s" % key
    assert rollback["steps"], "rollback has no steps"


# --------------------------------------------------------------------------
# Current policy is still disabled
# --------------------------------------------------------------------------


def test_policy_is_still_disabled():
    policy = _load_json(POLICY_PATH)
    assert policy["format"] == "veya-selective-policy-v1"
    assert policy["operating_epsilon"] == "disabled"
    assert policy["thresholds"]["disabled"] is None


def test_procedure_records_disabled_current_state():
    current = _procedure()["current_state"]
    assert current["operating_epsilon"] == "disabled"
    assert current["automation"] == "defer all"


def test_manifest_records_disabled_automation():
    manifest = _load_json(MANIFEST_PATH)
    automation = manifest.get("automation") or {}
    assert automation.get("status") == "DISABLED"
    current = _procedure()["current_state"]
    assert current.get("manifest_automation_status") == "DISABLED"


# --------------------------------------------------------------------------
# Recorded hashes match the actual files
# --------------------------------------------------------------------------


def test_procedure_policy_hash_matches_file():
    current = _procedure()["current_state"]
    assert current["policy_sha256"] == _sha256(POLICY_PATH)


def test_procedure_manifest_hash_matches_file():
    current = _procedure()["current_state"]
    assert current["manifest_sha256"] == _sha256(MANIFEST_PATH)


def test_manifest_still_pins_the_disabled_policy():
    # The policy file must not have been modified without a manifest re-mint:
    # this is what keeps an early "enabled" policy from ever loading.
    manifest = _load_json(MANIFEST_PATH)
    assert manifest["selective_policy_hash"] == _sha256(POLICY_PATH)
    assert manifest["selective_policy_hash"] == _procedure()["current_state"]["policy_sha256"]


def test_runtime_audit_pins_the_manifest():
    audit = _load_json(RUNTIME_AUDIT_PATH)
    assert audit["manifest_sha256"] == _sha256(MANIFEST_PATH)


def test_procedure_runtime_audit_pin_matches_file():
    current = _procedure()["current_state"]
    audit = _load_json(RUNTIME_AUDIT_PATH)
    assert current["runtime_audit_manifest_sha256"] == audit["manifest_sha256"]


def test_procedure_bindings_match_policy():
    current = _procedure()["current_state"]
    policy = _load_json(POLICY_PATH)
    bindings = current["policy_bindings"]
    for key in EXPECTED_BINDINGS:
        assert key in bindings, "missing recorded binding: %s" % key
        assert bindings[key] == policy[key], "binding drift: %s" % key


def test_procedure_thresholds_match_policy():
    current = _procedure()["current_state"]
    policy = _load_json(POLICY_PATH)
    assert current["thresholds"] == policy["thresholds"]
    assert policy["thresholds"]["disabled"] is None
    # The three preregistered thresholds stay frozen and selectable.
    for epsilon in ("0.02", "0.05", "0.10"):
        assert policy["thresholds"][epsilon] is not None


# --------------------------------------------------------------------------
# Prerequisite gates
# --------------------------------------------------------------------------


def test_all_prerequisite_gates_listed():
    gates = {entry["gate"] for entry in _procedure()["prerequisites"]}
    assert gates == EXPECTED_GATES


def test_every_prerequisite_requires_pass_with_evidence():
    for entry in _procedure()["prerequisites"]:
        gate = entry["gate"]
        assert entry["required_status"] == "PASS", gate
        assert (
            entry["evidence"] == "artifacts/release/prospective_blind_result.json"
        ), gate
        assert entry.get("evidence_fields"), "%s lists no evidence fields" % gate


# --------------------------------------------------------------------------
# Both CI tiers are required
# --------------------------------------------------------------------------


def test_release_candidate_gate_required():
    step = _steps()[5]
    assert step["action"] == "Run release-candidate gate"
    assert step["workflow"] == ".github/workflows/release-gate.yml"
    assert "v*" in step["trigger"] or "tag" in step["trigger"].lower()
    assert RELEASE_GATE_PATH.is_file()


def test_production_evidence_gate_required():
    step = _steps()[6]
    assert step["action"] == "Run production-evidence gate"
    assert step["workflow"] == ".github/workflows/release-gate.yml"
    assert step["workflow_dispatch_input"] == "run_production_gate=true"
    assert "run_production_gate=true" in step["command"]


def test_ci_tiers_come_after_local_fail_closed_verification():
    steps = _steps()
    assert "fail-closed" in steps[4]["action"].lower()
    assert "release-candidate" in steps[5]["action"].lower()
    assert "production-evidence" in steps[6]["action"].lower()
    acceptance = steps[6]["acceptance"].lower()
    assert "never runs on a tag push" in acceptance


def test_release_gate_workflow_defines_both_tiers():
    raw = RELEASE_GATE_PATH.read_text(encoding="utf-8")
    assert "release-candidate-gate:" in raw
    assert "production-evidence-gate:" in raw
    assert "workflow_dispatch" in raw
    assert "run_production_gate" in raw
    # The production-evidence tier is dispatch-only: it must never fire on a
    # tag push.
    assert "github.event_name == 'workflow_dispatch'" in raw


# --------------------------------------------------------------------------
# Step ordering: verdict verification precedes policy minting
# --------------------------------------------------------------------------


def test_verify_verdict_precedes_policy_mint():
    steps = _steps()
    assert steps[1]["action"] == "Verify prospective verdict"
    assert steps[2]["action"] == "Mint approved policy"


def test_step1_command_checks_pass_verdict_and_all_gates():
    step = _steps()[1]
    command = step["command"]
    assert "prospective_blind_result.json" in command
    assert "'PASS'" in command
    assert "all_primary_gates_pass" in command
    assert "primary_gates" in command


def test_step2_changes_only_operating_epsilon():
    step = _steps()[2]
    text = " ".join((step.get("details", ""), step.get("procedure", "")))
    assert "operating_epsilon" in text
    assert "Do not change threshold value" in text


# --------------------------------------------------------------------------
# Rollback returns to disabled
# --------------------------------------------------------------------------


def test_rollback_sets_operating_epsilon_back_to_disabled():
    rollback = _procedure()["rollback"]
    assert "disabled" in rollback["action"].lower()
    first_step = rollback["steps"][0].lower()
    assert "operating_epsilon" in first_step
    assert "disabled" in first_step


def test_rollback_re_mints_and_deploys_known_good_image():
    rollback = _procedure()["rollback"]
    joined = " ".join(rollback["steps"]).lower() + " " + rollback["action"].lower()
    assert "re-mint" in joined
    assert "known-good" in joined
    # The rollback trigger names the accepted-risk breach and drift alerts.
    trigger = rollback["trigger"].lower()
    assert "epsilon" in trigger
    assert "drift" in trigger
