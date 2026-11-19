import json
import os
import urllib.error
import urllib.request
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from hp import ROOT
from hp.db import connect, read_df
DOCS = ROOT / "data" / "docs"

STATS = {
    "batch_summary": """
        SELECT batch, COUNT(*) AS n_rows, ROUND(AVG(price)::numeric, 3) AS avg_price,
               ROUND(AVG(med_inc)::numeric, 3) AS avg_income
        FROM listings GROUP BY batch ORDER BY batch""",
    "income_brackets": """
        SELECT CASE WHEN med_inc < 3 THEN 'low' WHEN med_inc < 6 THEN 'middle' ELSE 'high' END
                   AS income_bracket,
               COUNT(*) AS n_rows, ROUND(AVG(price)::numeric, 3) AS avg_price
        FROM listings GROUP BY income_bracket ORDER BY avg_price""",
    "ingest_history": """
        SELECT batch, rows_loaded, rows_rejected, loaded_at FROM ingest_log ORDER BY batch""",
    "model_usage": """
        SELECT model_version, COUNT(*) AS n_predictions,
               ROUND(AVG(latency_ms)::numeric, 2) AS avg_ms
        FROM predictions GROUP BY model_version ORDER BY model_version""",
}


def post_json(url, payload, headers=None, timeout=120):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:  # 4xx/5xx still carry a useful JSON body
        return err.code, json.load(err)

def predict_price(**features):
    url = os.environ.get("API_URL", "http://127.0.0.1:8000") + "/predict"
    status, body = post_json(url, features, timeout=10)
    if status != 200:
        return f"ERROR {status}: {body.get('detail', body)}"
    return f"Predicted price: ${body['price_usd']:,} (model v{body['model_version']})"


def get_stats(name=""):
    if name not in STATS:
        return f"ERROR: unknown query '{name}'. Choose one of: {', '.join(STATS)}"
    with closing(connect()) as conn:
        return read_df(conn, STATS[name]).to_string(index=False)

def read_file(path):
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [page.extract_text() or "" for page in PdfReader(path).pages]
    return [path.read_text(encoding="utf-8")]


def load_chunks(folder, size=100, step=80):
    chunks = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        for page, text in enumerate(read_file(path), start=1):
            words = text.split()
            for i in range(0, len(words), step):
                window = " ".join(words[i:i + size])
                chunks.append({"source": path.name, "page": page, "text": window})
    return chunks

@lru_cache(maxsize=4)
def build_index(folder):
    chunks = load_chunks(folder)
    if not chunks:
        return None
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    return chunks, vectorizer, vectorizer.fit_transform(c["text"] for c in chunks)

def search_docs(query="", folder=DOCS, k=3):
    index = build_index(Path(folder))
    if index is None:
        return f"No documents found in {folder}"
    chunks, vectorizer, matrix = index
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()  # cosine similarity
    best = [i for i in scores.argsort()[::-1][:k] if scores[i] > 0]
    if not best:
        return "No relevant passages found."
    return "\n\n".join(f"[{chunks[i]['source']} p.{chunks[i]['page']}] {chunks[i]['text']}"
                       for i in best)

TOOLS = {
    "predict_price": (predict_price,
                      "Predict the price of ONE house block. args: med_inc, house_age, ave_rooms, "
                      "ave_bedrms, population, ave_occup, latitude, longitude (all numbers)."),
    "get_stats": (get_stats,
                  "Numbers about the data we hold and the API's usage. args: name, one of: "
                  + ", ".join(STATS)),
    "search_docs": (search_docs,
                    "Search the project documents (dataset, columns, model, drift, API). "
                    "args: query (a short text)."),
}

def run_tool(name, args):
    if name not in TOOLS:
        return f"ERROR: unknown tool '{name}'. Choose one of: {', '.join(TOOLS)}"
    try:
        return str(TOOLS[name][0](**args))
    except Exception as err:
        return f"ERROR: {type(err).__name__}: {err}"