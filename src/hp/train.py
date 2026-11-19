"""Train a price model on the database and save it as a new version.

    python -m hp.train

Train on every batch except the newest, then test on the newest: "learn from the past,
get graded on the future". Grading on rows the model has already seen would flatter it.
"""
import logging
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from hp import load_config, registry, setup_logging
from hp.db import connect, read_df

log = logging.getLogger("train")

FEATURES = ["med_inc", "house_age", "ave_rooms", "ave_bedrms", "population",
            "ave_occup", "latitude", "longitude"]


def load_listings():
    with closing(connect()) as conn:
        return read_df(conn, "SELECT * FROM listings ORDER BY id")


def time_split(df):
    newest = df["batch"].max()
    return df[df["batch"] < newest], df[df["batch"] == newest]


def train_model(df, seed=42):
    train, test = time_split(df)
    if train.empty:
        raise ValueError("need at least 2 batches loaded: run  python -m hp.ingest  twice")
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(train[FEATURES], train["price"])
    guess = np.full(len(test), train["price"].mean())  # baseline: always guess the average
    metrics = {
        "mae": float(mean_absolute_error(test["price"], model.predict(test[FEATURES]))),
        "r2": float(r2_score(test["price"], model.predict(test[FEATURES]))),
        "baseline_mae": float(mean_absolute_error(test["price"], guess)),
        "n_train": len(train),
        "n_test": len(test),
        "train_batches": sorted(int(b) for b in train["batch"].unique()),
        "test_batch": int(test["batch"].iloc[0]),
    }
    return model, metrics, train[FEATURES]


def run(cfg=None):
    cfg = cfg or load_config()
    model, m, train_x = train_model(load_listings(), cfg["seed"])
    reference = train_x.sample(min(len(train_x), 5000), random_state=cfg["seed"])
    meta = {**m, "features": FEATURES,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    version = registry.save(model, meta, reference)
    log.info("saved v%d | MAE %.3f (guess-the-average: %.3f) | R2 %.3f | train batches %s | "
             "tested on batch %d", version, m["mae"], m["baseline_mae"], m["r2"],
             m["train_batches"], m["test_batch"])
    return version


def main():
    setup_logging("train")
    try:
        run()
    except Exception:
        log.exception("train failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()