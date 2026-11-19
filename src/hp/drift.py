import argparse
import logging
from contextlib import closing
import numpy as np
from hp import load_config, registry, setup_logging, train
from hp.db import connect, read_df
from hp.train import FEATURES
log = logging.getLogger("drift")

def psi(reference, new, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf  
    ref = np.histogram(reference, edges)[0] / len(reference)  
    cur = np.histogram(new, edges)[0] / len(new)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)  
    return float(np.sum((cur - ref) * np.log(cur / ref)))

def check_drift(reference, new, threshold):
    scores = {f: psi(reference[f], new[f]) for f in FEATURES}
    return scores, [f for f, s in scores.items() if s > threshold]

def main():
    ap = argparse.ArgumentParser(description="Compare a batch with the model's training data")
    ap.add_argument("--batch", type=int, help="batch to check (default: newest in the database)")
    ap.add_argument("--demo", action="store_true", help="shift incomes up 50% to fake drift")
    ap.add_argument("--retrain", action="store_true", help="retrain if drift is found")
    args = ap.parse_args()
    setup_logging("drift")
    try:
        cfg = load_config()
        _, meta, reference = registry.load()
        with closing(connect()) as conn:
            batch = args.batch or conn.execute("SELECT MAX(batch) FROM ingest_log").fetchone()[0]
            new = read_df(conn, "SELECT * FROM listings WHERE batch = %s", (batch,))
        if new.empty:
            raise ValueError(f"no rows found for batch {batch}: run  python -m hp.ingest")
        if args.demo:
            new["med_inc"] *= 1.5
        scores, flagged = check_drift(reference, new, cfg["drift_threshold"])
        print(f"batch {batch} vs model v{meta['version']} (alarm above {cfg['drift_threshold']})")
        for feature, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {feature:<11} {score:6.3f}  {'DRIFT' if feature in flagged else ''}")
        if not flagged:
            log.info("batch %s: no drift", batch)
        else:
            log.warning("batch %s: drift in %s", batch, ", ".join(flagged))
            if args.retrain:
                train.run(cfg)
    except Exception:
        log.exception("drift check failed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()