import numpy as np
import pytest

from eval.stats import mcnemar_exact, holm


def test_mcnemar_exact_hand_computed():
    # b = A right & B wrong = 8, c = A wrong & B right = 2  -> two-sided exact binomial
    y = np.array([1] * 10 + [0] * 10)
    a = y.copy()
    b = y.copy()
    # make A right/B wrong on 8 rows, A wrong/B right on 2 rows
    b[:8] = 1 - b[:8]
    a[8:10] = 1 - a[8:10]
    n01, n10, p = mcnemar_exact(y, a, b)
    assert (n01, n10) == (8, 2)
    # P(X<=2 | n=10, 0.5)*2 = 2*(1+10+45)/1024
    assert p == pytest.approx(2 * 56 / 1024)


def test_mcnemar_no_disagreement_gives_p_one():
    y = np.array([1, 0, 1, 0])
    assert mcnemar_exact(y, y, y)[2] == 1.0


def test_holm_correction_is_monotone_and_capped():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert all(a <= 1 for a in adj)
