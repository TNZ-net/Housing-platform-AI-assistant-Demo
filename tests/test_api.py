import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from hp import api
from hp.train import FEATURES

client = TestClient(api.app)  # calls the API in-process, no server needed

GOOD = {"med_inc": 3.5, "house_age": 20, "ave_rooms": 5, "ave_bedrms": 1,
        "population": 1200, "ave_occup": 3, "latitude": 34.0, "longitude": -118.0}


@pytest.fixture
def served_model(monkeypatch):
    """Serve a fake model that always answers 2.5 (version 7)."""
    X = pd.DataFrame([[0] * len(FEATURES)], columns=FEATURES)
    model = DummyRegressor(strategy="constant", constant=2.5).fit(X, [2.5])
    monkeypatch.setitem(api.state, "model", model)
    monkeypatch.setitem(api.state, "meta", {"version": 7})


@pytest.fixture
def logged(served_model, monkeypatch):
    """Capture what would be written to the database instead of writing it."""
    calls = []
    monkeypatch.setattr(api, "log_prediction", lambda *args: calls.append(args))
    return calls


def db_down():
    raise RuntimeError("database is down")


def test_predict_returns_a_price_and_logs_it(logged):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json() == {"price": 2.5, "price_usd": 250000, "model_version": 7}
    assert len(logged) == 1


def test_bad_input_is_rejected_and_not_logged(logged):
    for bad in ({**GOOD, "med_inc": -1}, {**GOOD, "latitude": 95}, {"med_inc": 3}):
        assert client.post("/predict", json=bad).status_code == 422
    assert logged == []


def test_no_model_gives_503(monkeypatch):
    monkeypatch.delitem(api.state, "model", raising=False)
    assert client.post("/predict", json=GOOD).status_code == 503


def test_a_logging_failure_does_not_break_the_prediction(served_model, monkeypatch):
    monkeypatch.setattr(api, "connect", db_down)
    assert client.post("/predict", json=GOOD).status_code == 200


def test_health_reports_a_down_database(served_model, monkeypatch):
    monkeypatch.setattr(api, "connect", db_down)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "database": False}