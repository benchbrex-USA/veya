"""Shadow deployment drift monitor (protocol Section 13).

Partitions shadow telemetry into time periods, computes per-period
statistics, and raises alerts when risk, coverage, confidence
distributions, or config hashes drift across periods.
"""
import math
from collections import Counter, OrderedDict
from datetime import datetime

import numpy as np

from veya.selective import wilson_interval

REPORT_FORMAT = "veya-drift-report-v1"


def _parse_timestamp(ts):
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _period_key(dt, period):
    if period == "day":
        return dt.date().isoformat()
    if period == "week":
        iso = dt.isocalendar()
        return "%04d-W%02d" % (iso[0], iso[1])
    if period == "month":
        return "%04d-%02d" % (dt.year, dt.month)
    raise ValueError("unknown period %r" % period)


def partition_by_time(records, period="week"):
    """Group records into an ordered dict of period_key -> list of records."""
    buckets = OrderedDict()
    for rec in records:
        key = _period_key(_parse_timestamp(rec["timestamp"]), period)
        buckets.setdefault(key, []).append(rec)
    return OrderedDict(sorted(buckets.items()))


def _confidence_histogram(records, bins=10):
    hist = [0] * bins
    for rec in records:
        c = rec.get("confidence")
        if c is None:
            continue
        c = float(c)
        idx = min(bins - 1, max(0, int(c * bins)))
        hist[idx] += 1
    return hist


def _percentiles(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def period_stats(records):
    """Compute summary statistics for one period's records."""
    n = len(records)
    accepted_n = sum(1 for r in records if r.get("automation") == "answer")
    deferred_n = sum(1 for r in records if r.get("automation") == "defer")

    has_ground_truth = any(r.get("ground_truth") is not None for r in records)
    error_n = None
    risk = None
    risk_ucb95 = None
    if has_ground_truth:
        error_n = sum(
            1
            for r in records
            if r.get("automation") == "answer" and not r.get("correct")
        )
        if accepted_n > 0:
            risk = error_n / accepted_n
            risk_ucb95 = wilson_interval(error_n, accepted_n)[1]

    stats = {
        "n": n,
        "accepted_n": accepted_n,
        "deferred_n": deferred_n,
        "coverage": (accepted_n / n) if n else None,
        "error_n": error_n,
        "risk": risk,
        "risk_ucb95": risk_ucb95,
        "confidence_distribution": _confidence_histogram(records),
        "label_distribution": dict(
            Counter(
                r.get("ground_truth")
                if r.get("ground_truth") is not None
                else r.get("route")
                for r in records
            )
        ),
        "input_length_distribution": _percentiles(
            [r["candidate_count"] for r in records if r.get("candidate_count") is not None]
        ),
        "candidate_count_distribution": dict(
            Counter(r.get("candidate_count") for r in records)
        ),
        "mean_latency_ms": (
            float(np.mean([r["latency_ms"] for r in records])) if n else None
        ),
        "config_hashes": sorted(
            {r.get("config_hash") for r in records if r.get("config_hash") is not None}
        ),
    }
    return stats


def drift_report(records, period="week"):
    """Partition records and compute per-period statistics."""
    partitions = partition_by_time(records, period=period)
    stats = {key: period_stats(recs) for key, recs in partitions.items()}
    keys = list(stats.keys())
    return {
        "format": REPORT_FORMAT,
        "period": period,
        "partitions": stats,
        "first_period": keys[0] if keys else None,
        "last_period": keys[-1] if keys else None,
    }


def _ks_2samp_pvalue(a, b):
    """Two-sample KS test p-value without scipy (asymptotic Kolmogorov distribution)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return 1.0
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = (len(a) * len(b)) / (len(a) + len(b))
    return float(_kolmogorov_sf(d, n_eff))


def drift_triggers(report, epsilon, coverage_floor=0.05):
    """Analyze a drift report and return a list of alert dicts."""
    alerts = []
    partitions = report["partitions"]
    if not partitions:
        return alerts
    first_key = report["first_period"]
    first = partitions[first_key]

    for key, stats in partitions.items():
        if stats.get("risk_ucb95") is not None and stats["risk_ucb95"] > epsilon:
            alerts.append(
                {
                    "trigger": "risk_breach",
                    "period": key,
                    "risk_ucb95": stats["risk_ucb95"],
                    "epsilon": epsilon,
                }
            )
        if len(stats["config_hashes"]) > 1:
            alerts.append(
                {
                    "trigger": "config_mismatch",
                    "period": key,
                    "hashes": stats["config_hashes"],
                }
            )

    baseline_coverage = first.get("coverage")
    if baseline_coverage and baseline_coverage >= coverage_floor:
        for key, stats in partitions.items():
            if key == first_key or stats.get("coverage") is None:
                continue
            if baseline_coverage > 0 and stats["coverage"] < 0.8 * baseline_coverage:
                alerts.append(
                    {
                        "trigger": "coverage_drop",
                        "period": key,
                        "coverage": stats["coverage"],
                        "baseline_coverage": baseline_coverage,
                    }
                )

    # Raw confidence values are not retained in stats-only reports, so the
    # distribution check compares cumulative confidence histograms (an
    # empirical-CDF KS approximation over the 10 quantization bins).
    for key, stats in partitions.items():
        if key == first_key:
            continue
        p = _histogram_ks(first["confidence_distribution"], stats["confidence_distribution"])
        if p is not None and p < 0.01:
            alerts.append(
                {
                    "trigger": "distribution_shift",
                    "period": key,
                    "metric": "confidence",
                    "ks_pvalue": p,
                }
            )
    return alerts


def _histogram_ks(hist_a, hist_b):
    """Approximate KS p-value from confidence histograms via empirical CDFs."""
    ta, tb = sum(hist_a), sum(hist_b)
    if ta == 0 or tb == 0:
        return None
    bins = len(hist_a)
    ca = np.cumsum(hist_a) / ta
    cb = np.cumsum(hist_b) / tb
    d = float(np.max(np.abs(ca - cb)))
    n_eff = (ta * tb) / (ta + tb)
    return float(_kolmogorov_sf(d, n_eff))


def _kolmogorov_sf(d, n_eff):
    """Asymptotic Kolmogorov distribution survival function with the
    standard stephens correction. Identical distributions (d == 0) give p = 1."""
    if d <= 0.0:
        return 1.0
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    p = 0.0
    for j in range(1, 101):
        p += 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
    return min(1.0, max(0.0, p))
