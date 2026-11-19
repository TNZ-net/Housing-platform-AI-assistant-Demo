CREATE TABLE IF NOT EXISTS listings (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch       INTEGER NOT NULL,
    med_inc     DOUBLE PRECISION NOT NULL,
    house_age   DOUBLE PRECISION NOT NULL,
    ave_rooms   DOUBLE PRECISION NOT NULL CHECK (ave_rooms > 0),
    ave_bedrms  DOUBLE PRECISION NOT NULL,
    population  DOUBLE PRECISION NOT NULL,
    ave_occup   DOUBLE PRECISION NOT NULL,
    latitude    DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude   DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    price       DOUBLE PRECISION NOT NULL CHECK (price > 0)
);

CREATE INDEX IF NOT EXISTS idx_listings_batch ON listings (batch);

CREATE TABLE IF NOT EXISTS ingest_log (
    batch         INTEGER PRIMARY KEY,
    rows_loaded   INTEGER NOT NULL,
    rows_rejected INTEGER NOT NULL,
    loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   INTEGER NOT NULL,
    inputs          JSONB NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions (created_at);