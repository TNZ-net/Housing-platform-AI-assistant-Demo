import os
import sys
from pathlib import Path
import pandas as pd
import psycopg
from hp import ROOT

def connect():
    """Connect to Postgres using DATABASE_URL (creating the tables on first use)."""
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute((ROOT / "sql" / "schema.sql").read_text())  # safe to repeat: IF NOT EXISTS
    return conn

def read_df(conn, sql, params=None):
    """Run a query and return the result as a DataFrame."""
    cur = conn.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])

if __name__ == "__main__":
    conn = connect()
    for query in Path(sys.argv[1]).read_text().split(";"):  # one statement per ';'
        if query.strip():
            cur = conn.execute(query)
            df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
            print(df.to_string(index=False), "\n")