"""Sample-size search against the protocol Section 7 reference values.

The Wilson upper bound is re-derived inline in these tests so they verify the
module against the formula independently instead of trusting its own code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from statistics import NormalDist

import pytest

from research.veyabench.sample_size import (  # noqa: E402
    required_n,
    wilson_upper_bound,
)

EPSILON = 0.02
REFERENCE = ((0, 189), (1, 280), (2, 361), (3, 437))


def independent_ucb(errors, accepted, z):
    """Wilson 95% upper bound, re-derived here so tests do not trust the module."""
    p = errors / accepted
    spread = p * (1 - p) / accepted + z * z / (4 * accepted * accepted)
    margin = z * math.sqrt(spread)
    return (p + z * z / (2 * accepted) + margin) / (1 + z * z / accepted)


@pytest.mark.parametrize(("max_errors", "expected"), REFERENCE)
def test_protocol_section7_reference_values(max_errors, expected):
    result = required_n(EPSILON, 1.0, max_errors)
    assert abs(result["accepted"] - expected) <= 5
    assert result["total"] == result["accepted"]  # full anticipated coverage
    z = NormalDist().inv_cdf(0.975)
    assert independent_ucb(max_errors, result["accepted"], z) <= EPSILON
    # The returned count is minimal: one fewer case already breaks the bound.
    assert independent_ucb(max_errors, result["accepted"] - 1, z) > EPSILON


def test_zero_errors_matches_the_closed_form():
    # 0 errors: ucb = z^2 / (n + z^2) <= eps  =>  n >= z^2 (1 - eps) / eps.
    z = NormalDist().inv_cdf(0.975)
    closed_form = math.ceil(z * z * (1 - EPSILON) / EPSILON)
    result = required_n(EPSILON, 1.0, 0)
    assert result["accepted"] == closed_form == 189


def test_more_allowed_errors_require_more_accepted_cases():
    counts = [required_n(EPSILON, 1.0, errors)["accepted"] for errors in range(4)]
    assert counts == [189, 280, 361, 437]


def test_confidence_monotonicity():
    loose = required_n(EPSILON, 1.0, 1, confidence=0.90)["accepted"]
    base = required_n(EPSILON, 1.0, 1)["accepted"]
    strict = required_n(EPSILON, 1.0, 1, confidence=0.99)["accepted"]
    assert loose < base < strict


def test_total_scales_with_anticipated_coverage():
    accepted = required_n(EPSILON, 1.0, 0)["accepted"]
    full = required_n(EPSILON, 1.0, 0)
    assert full["total"] == accepted
    high = required_n(EPSILON, 0.9, 0)  # high coverage
    assert high["accepted"] == accepted
    assert high["total"] == math.ceil(accepted / 0.9) == 210
    fractional = required_n(EPSILON, 0.8, 0)
    assert fractional["total"] == math.ceil(accepted / 0.8) == 237
    low = required_n(EPSILON, 0.25, 0)  # low coverage
    assert low["accepted"] == accepted
    assert low["total"] == 756


@pytest.mark.parametrize("coverage", [0.9, 0.7, 0.5, 0.25, 0.1])
def test_total_is_the_smallest_integer_that_clears_acceptance(coverage):
    result = required_n(EPSILON, coverage, 2)
    assert result["accepted"] == 361
    assert result["total"] >= result["accepted"]
    assert result["total"] * coverage >= result["accepted"] - 1e-9
    assert (result["total"] - 1) * coverage < result["accepted"]


def test_reported_bound_matches_the_public_helper():
    result = required_n(EPSILON, 0.7, 2)
    assert result["wilson_ucb_at_accepted"] == pytest.approx(
        wilson_upper_bound(2, result["accepted"])
    )
    assert result["wilson_ucb_at_accepted"] <= EPSILON
    assert result["z"] == pytest.approx(NormalDist().inv_cdf(0.975))


def test_wilson_upper_bound_helper():
    z = NormalDist().inv_cdf(0.975)
    assert wilson_upper_bound(0, 189) == pytest.approx(independent_ucb(0, 189, z))
    assert wilson_upper_bound(3, 437) <= EPSILON
    assert wilson_upper_bound(3, 436) > EPSILON
    assert wilson_upper_bound(1, 1) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="accepted"):
        wilson_upper_bound(0, 0)
    with pytest.raises(ValueError, match="errors"):
        wilson_upper_bound(2, 1)
    with pytest.raises(ValueError, match="errors"):
        wilson_upper_bound(-1, 10)
    with pytest.raises(ValueError, match="confidence"):
        wilson_upper_bound(0, 100, confidence=1.5)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="epsilon"):
        required_n(0.0, 1.0, 0)
    with pytest.raises(ValueError, match="epsilon"):
        required_n(1.0, 1.0, 0)
    with pytest.raises(ValueError, match="epsilon"):
        required_n("small", 1.0, 0)
    with pytest.raises(ValueError, match="anticipated_coverage"):
        required_n(EPSILON, 0.0, 0)
    with pytest.raises(ValueError, match="anticipated_coverage"):
        required_n(EPSILON, 1.5, 0)
    with pytest.raises(ValueError, match="max_errors"):
        required_n(EPSILON, 1.0, -1)
    with pytest.raises(ValueError, match="max_errors"):
        required_n(EPSILON, 1.0, 2.5)
    with pytest.raises(ValueError, match="confidence"):
        required_n(EPSILON, 1.0, 0, confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        required_n(EPSILON, 1.0, 0, confidence=1.0)
