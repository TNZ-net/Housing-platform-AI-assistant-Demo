import json
import joblib
import pandas as pd
from hp import ROOT
MODELS = ROOT / "models"

def versions():
    return sorted(int(p.name[1:]) for p in MODELS.glob("v*") if p.name[1:].isdigit())

def save(model, meta, reference):
    version = max(versions(), default=0) + 1
    folder = MODELS / f"v{version}"
    folder.mkdir(parents=True)
    joblib.dump(model, folder / "model.joblib")
    reference.to_csv(folder / "reference.csv", index=False)  # what the model saw, for drift checks
    (folder / "meta.json").write_text(json.dumps({**meta, "version": version}, indent=2))
    return version

def load(version=None):
    if version is None:
        if not versions():
            raise FileNotFoundError("no model trained yet: run  python -m hp.train")
        version = max(versions())
    folder = MODELS / f"v{version}"
    return (joblib.load(folder / "model.joblib"),
            json.loads((folder / "meta.json").read_text()),
            pd.read_csv(folder / "reference.csv"))