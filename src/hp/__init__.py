import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  
load_dotenv(ROOT / ".env")  


def load_config(path="configs/platform.yaml"):
    return yaml.safe_load((ROOT / path).read_text())


def setup_logging(name):
    """Log every message to the screen and to logs/<name>.log."""
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / f"{name}.log"), logging.StreamHandler()],
    )