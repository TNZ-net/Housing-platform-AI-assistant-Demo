import argparse
import logging
from contextlib import closing
from sklearn.datasets import fetch_california_housing
from hp import load_config, setup_logging
from hp.db import connect

log = logging.getLogger("ingest")
COLUMNS = {  # name in the source data -> name in our table
    "MedInc": "med_inc", "HouseAge": "house_age", "AveRooms": "ave_rooms",
    "AveBedrms": "ave_bedrms", "Population": "population", "AveOccup": "ave_occup",
    "Latitude": "latitude", "Longitude": "longitude", "MedHouseVal": "price",
}
DB_COLUMNS = list(COLUMNS.values())
INSERT = (f"INSERT INTO listings (batch, {', '.join(DB_COLUMNS)}) "
          f"VALUES ({', '.join(['%s'] * (len(DB_COLUMNS) + 1))})")

def pick_batch(df, batch, cfg):
    shuffled = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    return shuffled.iloc[batch - 1::cfg["n_batches"]]

def fetch_batch(batch, cfg):
    df = fetch_california_housing(as_frame=True).frame.rename(columns=COLUMNS)
    return pick_batch(df, batch, cfg)

def validate(df):
    ok = (df.notna().all(axis=1)
          & (df["price"] > 0)
          & (df["ave_rooms"] > 0)
          & df["latitude"].between(-90, 90)
          & df["longitude"].between(-180, 180))
    return df[ok], int((~ok).sum())

def ingest(batch=None, cfg=None):
    cfg = cfg or load_config()
    with closing(connect()) as conn:
        if batch is None:  # default: one past the highest batch loaded so far
            batch = conn.execute("SELECT COALESCE(MAX(batch), 0) + 1 FROM ingest_log").fetchone()[0]
            if batch > cfg["n_batches"]:
                log.info("all %d batches are already loaded", cfg["n_batches"])
                return 0
        if not 1 <= batch <= cfg["n_batches"]:
            raise ValueError(f"batch must be between 1 and {cfg['n_batches']}")
        if conn.execute("SELECT 1 FROM ingest_log WHERE batch = %s", (batch,)).fetchone():
            log.info("batch %d already loaded, skipping", batch)
            return 0

        clean, rejected = validate(fetch_batch(batch, cfg))
        rows = clean.assign(batch=batch)[["batch", *DB_COLUMNS]].itertuples(index=False, name=None)
        with conn.transaction():  # one transaction: both inserts are saved together, or neither is
            with conn.cursor() as cur:
                cur.executemany(INSERT, rows)
            conn.execute(
                "INSERT INTO ingest_log (batch, rows_loaded, rows_rejected) VALUES (%s, %s, %s)",
                (batch, len(clean), rejected))
        log.info("batch %d: loaded %d rows, rejected %d", batch, len(clean), rejected)
        return len(clean)

def main():
    ap = argparse.ArgumentParser(description="Load one batch of housing data into the database")
    ap.add_argument("--batch", type=int, help="batch number to load (default: the next one)")
    args = ap.parse_args()
    setup_logging("ingest")
    try:
        ingest(args.batch)
    except Exception:
        log.exception("ingest failed")
        raise SystemExit(1)   
if __name__ == "__main__":
    main()