import numpy as np
import pandas as pd
import pytest

from hp import train


def fake_listings(n=900, batches=3, seed=0):
    """Fake database rows where price really does depend on income."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.uniform(1, 10, n) for f in train.FEATURES})
    df["price"] = 1 + 0.5 * df["med_inc"] + rng.normal(0, 0.1, n)
    df["batch"] = np.arange(n) % batches + 1
    return df


def test_the_model_never_trains_on_the_newest_batch():
    fit, test = train.time_split(fake_listings())
    assert fit["batch"].max() < test["batch"].min()
    assert set(test["batch"]) == {3}


def test_the_model_beats_guessing_the_average():
    _, metrics, _ = train.train_model(fake_listings())
    assert metrics["mae"] < metrics["baseline_mae"] / 2
    assert metrics["r2"] > 0.8
    assert metrics["train_batches"] == [1, 2] and metrics["test_batch"] == 3


def test_a_single_batch_is_not_enough():
    with pytest.raises(ValueError, match="at least 2"):
        train.train_model(fake_listings(batches=1))