import os
from contextlib import closing
import numpy as np
import pandas as pd
import psycopg
import pytest
from hp import ingest as ing
from hp.db import connect

TEST_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="set TEST_DATABASE_URL to run DB tests")


@pytest.fixture
def cfg():
    return {"n_batches": 3, "seed": 1}


@pytest.fixture(autouse=True)
def clean_db(monkeypatch):
    """Point every test at the TEST database and start it empty."""
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    with closing(connect()) as conn:
        conn.execute("TRUNCATE listings, ingest_log RESTART IDENTITY")


def make_rows(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "med_inc": rng.uniform(1, 10, n), "house_age": rng.integers(1, 50, n).astype(float),
        "ave_rooms": rng.uniform(3, 8, n), "ave_bedrms": rng.uniform(0.8, 1.5, n),
        "population": rng.integers(100, 3000, n).astype(float), "ave_occup": rng.uniform(1, 5, n),
        "latitude": rng.uniform(32, 42, n), "longitude": rng.uniform(-124, -114, n),
        "price": rng.uniform(0.5, 5, n)})


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    """Every test gets 50 synthetic rows instead of downloading the real dataset."""
    monkeypatch.setattr(ing, "fetch_batch", lambda batch, cfg: make_rows())


def count(table):
    with closing(connect()) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_loads_rows_and_logs_the_batch(cfg):
    assert ing.ingest(1, cfg) == 50
    assert count("listings") == 50 and count("ingest_log") == 1


def test_rerunning_a_batch_adds_nothing(cfg):
    ing.ingest(1, cfg)
    assert ing.ingest(1, cfg) == 0
    assert count("listings") == 50


def test_default_loads_the_next_batch_then_stops(cfg):
    for expected in (50, 100, 150, 150):  # the 4th call has nothing left to load
        ing.ingest(None, cfg)
        assert count("listings") == expected


def test_bad_rows_are_rejected_and_counted(cfg, monkeypatch):
    bad = make_rows()
    bad.loc[0, "price"] = -1.0
    bad.loc[1, "med_inc"] = np.nan
    monkeypatch.setattr(ing, "fetch_batch", lambda batch, cfg: bad)
    assert ing.ingest(1, cfg) == 48
    with closing(connect()) as conn:
        assert conn.execute("SELECT rows_rejected FROM ingest_log").fetchone()[0] == 2


def test_a_failed_load_leaves_nothing_behind(cfg, monkeypatch):
    bad = make_rows()
    bad.loc[0, "price"] = -1.0  # skip validate() so the database itself refuses it
    monkeypatch.setattr(ing, "validate", lambda df: (bad, 0))
    with pytest.raises(psycopg.errors.IntegrityError):
        ing.ingest(1, cfg)
    assert count("listings") == 0 and count("ingest_log") == 0


def test_out_of_range_batch_is_rejected(cfg):
    for batch in (0, 4):
        with pytest.raises(ValueError):
            ing.ingest(batch, cfg)


def test_pick_batch_deals_every_row_exactly_once(cfg):
    df = make_rows(31)
    parts = [ing.pick_batch(df, k, cfg) for k in (1, 2, 3)]
    assert sum(len(p) for p in parts) == 31
    assert sorted(pd.concat(parts)["price"]) == sorted(df["price"])