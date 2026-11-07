import numpy as np
import pytest

from eval.metrics import (
    prf, fpr, ece, brier, auroc, bootstrap_ci, fit_temperature, apply_temperature,
)


def test_prf_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])
    p, r, f = prf(y, pred)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f == pytest.approx(2 / 3)


def test_prf_no_predicted_positives_is_zero_not_nan():
    assert prf(np.array([1, 0]), np.array([0, 0])) == (0.0, 0.0, 0.0)


def test_fpr():
    y = np.array([0, 0, 0, 0, 1])
    pred = np.array([1, 0, 0, 0, 1])
    assert fpr(y, pred) == pytest.approx(0.25)


def test_brier_hand_computed():
    y = np.array([1, 0])
    p = np.array([0.8, 0.4])
    assert brier(y, p) == pytest.approx((0.04 + 0.16) / 2)


def test_ece_perfectly_calibrated_is_zero_and_overconfident_is_large():
    y = np.array([1] * 5 + [0] * 5)
    assert ece(y, np.full(10, 0.5), bins=10) == pytest.approx(0.0)
    # always predicts 0.99 but right only half the time
    assert ece(y, np.full(10, 0.99), bins=10) == pytest.approx(0.49)


def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_bootstrap_ci_contains_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    pred = np.where(rng.random(200) < 0.85, y, 1 - y)
    groups = np.arange(200)
    f1 = lambda a, b: prf(a, b)[2]
    lo, hi = bootstrap_ci(y, pred, f1, groups, n=300, seed=1)
    point = f1(y, pred)
    assert lo <= point <= hi
    assert (lo, hi) == bootstrap_ci(y, pred, f1, groups, n=300, seed=1)


def test_bootstrap_ci_resamples_groups_so_clustered_data_gets_wider_ci():
    rng = np.random.default_rng(0)
    n_groups, per = 20, 10
    groups = np.repeat(np.arange(n_groups), per)
    y = np.tile([0, 1], n_groups * per // 2)
    # each group is either all-correct or all-wrong -> strong clustering
    good = rng.random(n_groups) < 0.8
    pred = np.where(good[groups], y, 1 - y)
    acc = lambda a, b: float(np.mean(a == b))
    lo_c, hi_c = bootstrap_ci(y, pred, acc, groups, n=400, seed=2)
    lo_i, hi_i = bootstrap_ci(y, pred, acc, np.arange(len(y)), n=400, seed=2)
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # logits that are right ~80% of the time but have huge magnitude
    sign = np.where(rng.random(n) < 0.8, 1, -1) * np.where(y == 1, 1, -1)
    logit = sign * rng.uniform(4, 8, n)
    T = fit_temperature(logit, y)
    assert T > 1.5
    before = ece(y, 1 / (1 + np.exp(-logit)), bins=10)
    after = ece(y, apply_temperature(logit, T), bins=10)
    assert after < before
