"""Tests for research/veyabench/drift_monitor.py."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "veyabench"))

import pytest

from drift_monitor import drift_report, drift_triggers, partition_by_time, period_stats


def _ts(week, day=1, hour=12):
    # 2026-01-05 is a Monday; week n starts on that plus 7*(n-1) days.
    dt = datetime(2026, 1, 5, hour) + timedelta(days=7 * (week - 1) + (day - 1))
    return dt.isoformat()


def _record(week, i, *, confidence=0.9, automation="answer", correct=1,
            config_hash="cfg-a", route="billing", candidate_count=3, latency=50):
    return {
        "timestamp": _ts(week, day=(i % 7) + 1),
        "primitive": "classify",
        "candidate_count": candidate_count,
        "route": route,
        "deferred": automation == "defer",
        "confidence": confidence,
        "max_probability": confidence,
        "automation": automation,
        "config_hash": config_hash,
        "status": 200,
        "latency_ms": latency,
        "ground_truth": route,
        "correct": correct,
    }


def _make_records():
    """4 weeks: 1-2 stable low risk, 3 confidence shift, 4 risk increase."""
    records = []
    # Weeks 1-2: 100 records each, 90% confidence, ~2% wrong.
    for week in (1, 2):
        for i in range(100):
            records.append(
                _record(week, i, correct=0 if i % 50 == 0 else 1, confidence=0.95)
            )
    # Week 3: low-confidence records (confidence distribution shifts down).
    for i in range(100):
        records.append(_record(3, i, confidence=0.20, correct=1))
    # Week 4: 30% of accepted answers wrong (risk breach).
    for i in range(100):
        records.append(
            _record(4, i, confidence=0.95, correct=0 if i % 10 < 3 else 1)
        )
    return records


class TestPartitionByTime:
    def test_week_groupings(self):
        parts = partition_by_time(_make_records(), period="week")
        assert len(parts) == 4
        keys = list(parts.keys())
        assert all(k.startswith("2026-W") for k in keys)
        assert keys == sorted(keys)
        assert all(len(v) == 100 for v in parts.values())

    def test_day_groupings(self):
        records = [_record(1, i) for i in range(10)]
        parts = partition_by_time(records, period="day")
        assert len(parts) == 7
        assert all(len(v) >= 1 for v in parts.values())

    def test_records_preserved(self):
        records = [_record(1, i) for i in range(5)]
        parts = partition_by_time(records)
        assert sum(len(v) for v in parts.values()) == 5
        assert all(r["primitive"] == "classify" for v in parts.values() for r in v)


class TestPeriodStats:
    def test_counts_and_coverage(self):
        recs = [
            _record(1, 1, automation="answer"),
            _record(1, 2, automation="answer", correct=0),
            _record(1, 3, automation="defer"),
        ]
        stats = period_stats(recs)
        assert stats["n"] == 3
        assert stats["accepted_n"] == 2
        assert stats["deferred_n"] == 1
        assert stats["coverage"] == pytest.approx(2 / 3)
        assert stats["error_n"] == 1
        assert stats["risk"] == pytest.approx(0.5)
        assert stats["risk_ucb95"] > 0.5

    def test_confidence_histogram(self):
        recs = [_record(1, i, confidence=0.95) for i in range(10)]
        hist = period_stats(recs)["confidence_distribution"]
        assert len(hist) == 10
        assert sum(hist) == 10
        assert hist[9] == 10  # 0.95 falls in the last bin

    def test_input_and_latency(self):
        recs = [_record(1, i, candidate_count=i + 1, latency=10 * (i + 1)) for i in range(4)]
        stats = period_stats(recs)
        assert stats["input_length_distribution"]["p50"] == pytest.approx(2.5)
        assert stats["candidate_count_distribution"] == {1: 1, 2: 1, 3: 1, 4: 1}
        assert stats["mean_latency_ms"] == pytest.approx(25.0)

    def test_config_hashes(self):
        recs = [
            _record(1, 1, config_hash="cfg-a"),
            _record(1, 2, config_hash="cfg-b"),
            _record(1, 3, config_hash="cfg-a"),
        ]
        assert period_stats(recs)["config_hashes"] == ["cfg-a", "cfg-b"]

    def test_no_ground_truth(self):
        recs = []
        for i in range(4):
            r = _record(1, i, automation="answer" if i < 2 else "defer")
            r["ground_truth"] = None
            r["correct"] = None
            recs.append(r)
        stats = period_stats(recs)
        assert stats["error_n"] is None
        assert stats["risk"] is None
        assert stats["risk_ucb95"] is None


class TestDriftReport:
    def test_structure(self):
        report = drift_report(_make_records(), period="week")
        assert report["format"] == "veya-drift-report-v1"
        assert report["period"] == "week"
        assert len(report["partitions"]) == 4
        assert report["first_period"] == sorted(report["partitions"])[0]
        assert report["last_period"] == sorted(report["partitions"])[-1]


class TestDriftTriggers:
    def _no_alerts_for_stable(self):
        records = [_record(1, i) for i in range(100)] + [_record(2, i) for i in range(100)]
        report = drift_report(records)
        assert drift_triggers(report, epsilon=0.10) == []

    def test_risk_breach_week4(self):
        report = drift_report(_make_records())
        alerts = drift_triggers(report, epsilon=0.10)
        risk = [a for a in alerts if a["trigger"] == "risk_breach"]
        assert len(risk) == 1
        assert risk[0]["period"] == list(report["partitions"])[3]
        assert risk[0]["epsilon"] == 0.10
        assert risk[0]["risk_ucb95"] > 0.10

    def test_distribution_shift_week3(self):
        report = drift_report(_make_records())
        alerts = drift_triggers(report, epsilon=0.10)
        shifts = [a for a in alerts if a["trigger"] == "distribution_shift"]
        assert shifts, "week-3 confidence shift should fire"
        assert shifts[0]["metric"] == "confidence"
        assert shifts[0]["ks_pvalue"] < 0.01
        # Week 2 is identical in distribution to week 1; should not fire.
        assert all(a["period"] != list(report["partitions"])[1] for a in shifts)

    def test_coverage_drop(self):
        records = []
        # Week 1: mostly answered. Week 2: mostly deferred -> coverage drop >20%.
        for i in range(100):
            records.append(_record(1, i, automation="answer"))
        for i in range(100):
            records.append(_record(2, i, automation="defer"))
        report = drift_report(records)
        alerts = drift_triggers(report, epsilon=0.5)
        drops = [a for a in alerts if a["trigger"] == "coverage_drop"]
        assert len(drops) == 1
        assert drops[0]["coverage"] == pytest.approx(0.0)
        assert drops[0]["baseline_coverage"] == pytest.approx(1.0)

    def test_no_coverage_drop_when_stable(self):
        records = [_record(1, i) for i in range(100)] + [_record(2, i) for i in range(100)]
        report = drift_report(records)
        alerts = drift_triggers(report, epsilon=0.5)
        assert not [a for a in alerts if a["trigger"] == "coverage_drop"]

    def test_config_mismatch(self):
        records = (
            [_record(1, i, config_hash="cfg-a") for i in range(50)]
            + [_record(1, i, config_hash="cfg-b") for i in range(50)]
        )
        report = drift_report(records)
        alerts = drift_triggers(report, epsilon=0.5)
        mismatches = [a for a in alerts if a["trigger"] == "config_mismatch"]
        assert len(mismatches) == 1
        assert mismatches[0]["period"] == report["first_period"]
        assert sorted(mismatches[0]["hashes"]) == ["cfg-a", "cfg-b"]

    def test_stable_periods_no_alerts(self):
        records = [_record(1, i) for i in range(100)] + [_record(2, i) for i in range(100)]
        report = drift_report(records)
        assert drift_triggers(report, epsilon=0.10) == []
