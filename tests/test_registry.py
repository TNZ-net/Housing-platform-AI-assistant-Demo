import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from hp import registry

@pytest.fixture(autouse=True)
def temp_models_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS", tmp_path / "models")


def save_one(note):
    model = LinearRegression().fit([[0], [1]], [0, 1])
    return registry.save(model, {"note": note}, pd.DataFrame({"x": [0, 1]}))


def test_versions_count_up_from_one():
    assert [save_one("a"), save_one("b"), save_one("c")] == [1, 2, 3]
    assert registry.versions() == [1, 2, 3]


def test_load_returns_the_latest_unless_told_otherwise():
    save_one("first")
    save_one("second")
    model, meta, reference = registry.load()
    assert meta["note"] == "second" and meta["version"] == 2
    assert registry.load(1)[1]["note"] == "first"
    assert model.predict([[0.5]])[0] == pytest.approx(0.5)
    assert list(reference["x"]) == [0, 1]


def test_loading_with_no_model_gives_a_clear_error():
    with pytest.raises(FileNotFoundError, match="python -m hp.train"):
        registry.load()