import json
import logging
import time
from contextlib import asynccontextmanager, closing

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hp import registry, setup_logging
from hp.db import connect, read_df
from hp.train import FEATURES

setup_logging("api")
log = logging.getLogger("api")
state = {}  # the model being served lives here: {"model": ..., "meta": ...}
STATS_SQL = """
SELECT model_version, COUNT(*) AS n_predictions, AVG(predicted_price) AS avg_price,
       AVG(latency_ms) AS avg_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
FROM predictions
WHERE created_at > now() - interval '24 hours'
GROUP BY model_version
ORDER BY model_version
"""
class Listing(BaseModel):
    """One house block. Anything outside these rules is rejected automatically (HTTP 422)."""
    med_inc: float = Field(gt=0, description="median income, in $10,000s")
    house_age: float = Field(ge=0, description="median house age, in years")
    ave_rooms: float = Field(gt=0)
    ave_bedrms: float = Field(gt=0)
    population: float = Field(gt=0)
    ave_occup: float = Field(gt=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


def load_model():
    state["model"], state["meta"], _ = registry.load()  # 3rd value is drift data, unused here
    log.info("serving model v%s", state["meta"]["version"])


@asynccontextmanager
async def lifespan(app):
    """Code before `yield` runs once at startup; code after it runs once at shutdown."""
    try:
        load_model()
    except FileNotFoundError:
        log.warning("no trained model yet: run  python -m hp.train  then POST /reload")
    yield

app = FastAPI(title="Housing Platform", lifespan=lifespan)

def log_prediction(version, inputs, price, ms):
    """Save the request. Monitoring must never be able to break serving, so errors are swallowed."""
    try:
        with closing(connect()) as conn:
            conn.execute(
                "INSERT INTO predictions (model_version, inputs, predicted_price, latency_ms) "
                "VALUES (%s, %s::jsonb, %s, %s)",
                (version, json.dumps(inputs), price, ms))
    except Exception:
        log.exception("could not log prediction")

@app.post("/predict")
def predict(listing: Listing):
    if "model" not in state:
        raise HTTPException(503, "no model loaded: run  python -m hp.train  then POST /reload")
    inputs = listing.model_dump()
    start = time.perf_counter()
    price = float(state["model"].predict(pd.DataFrame([inputs])[FEATURES])[0])
    ms = (time.perf_counter() - start) * 1000
    version = state["meta"]["version"]
    log_prediction(version, inputs, price, ms)
    return {"price": round(price, 3), "price_usd": round(price * 100_000),
            "model_version": version}


@app.get("/health")
def health():
    try:
        with closing(connect()) as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok", "model_loaded": "model" in state, "database": db_ok}


@app.get("/model")
def model_info():
    if "meta" not in state:
        raise HTTPException(503, "no model loaded")
    return state["meta"]


@app.get("/stats")
def stats():
    with closing(connect()) as conn:
        return read_df(conn, STATS_SQL).round(3).to_dict("records")


@app.post("/reload")
def reload():
    try:
        load_model()
    except FileNotFoundError as err:
        raise HTTPException(404, str(err))
    return {"model_version": state["meta"]["version"]}