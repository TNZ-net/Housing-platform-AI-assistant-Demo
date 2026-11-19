import numpy as np
import pandas as pd
from hp import drift
from hp.train import FEATURES

def sample(seed, n=2000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.lognormal(1, 0.5, n) for f in FEATURES})


def test_same_distribution_scores_near_zero():
    assert drift.psi(sample(1)["med_inc"], sample(2)["med_inc"]) < 0.05


def test_a_big_shift_scores_high():
    old, new = sample(1)["med_inc"], sample(2)["med_inc"] * 1.5
    assert drift.psi(old, new) > 0.25


def test_only_the_shifted_feature_is_flagged():
    new = sample(2)
    new["med_inc"] *= 1.5
    scores, flagged = drift.check_drift(sample(1), new, threshold=0.25)
    assert flagged == ["med_inc"]
    assert set(scores) == set(FEATURES)


def test_a_constant_feature_does_not_crash():
    assert drift.psi(pd.Series([5.0] * 100), pd.Series([5.0] * 50)) == 0.0