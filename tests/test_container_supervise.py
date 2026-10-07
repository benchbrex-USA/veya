"""The container boot supervisor fails closed before the gateway starts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "cloudflare"))

import supervise  # noqa: E402

PINS = {
    "VEYA_RELEASE_ID": "0.3.5-audit-baseline.1",
    "VEYA_CHECKPOINT_SHA256": "a" * 64,
    "VEYA_CHECKPOINT_BYTES": "842609210",
    "VEYA_MANIFEST_SHA256": "b" * 64,
}


def test_pin_mismatch_does_not_fetch(tmp_path):
    marker = tmp_path / "fetched"
    environ = dict(PINS)
    environ["VEYA_CHECKPOINT_SHA256"] = "c" * 64
    try:
        supervise.supervise(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"],
            [],
            environ=environ,
            pins=PINS,
        )
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("pin mismatch must exit")
    assert not marker.exists()


def test_fetch_failure_does_not_start_children(tmp_path):
    marker = tmp_path / "started"
    code = supervise.supervise(
        [sys.executable, "-c", "raise SystemExit(3)"],
        [[sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"]],
        environ=PINS,
        pins=PINS,
    )
    assert code == 3
    assert not marker.exists()


def test_child_exit_stops_the_other(tmp_path):
    lived = tmp_path / "lived"
    code = supervise.supervise(
        [sys.executable, "-c", "raise SystemExit(0)"],
        [
            [sys.executable, "-c", "raise SystemExit(9)"],
            [sys.executable, "-c", f"import time; time.sleep(30); open({str(lived)!r}, 'w').write('x')"],
        ],
        environ=PINS,
        pins=PINS,
    )
    assert code == 9
    assert not lived.exists()
