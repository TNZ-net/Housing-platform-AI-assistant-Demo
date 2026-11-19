SELECT model_version, COUNT(*) AS n_predictions,
       ROUND(AVG(predicted_price)::numeric, 3) AS avg_price,
       ROUND(AVG(latency_ms)::numeric, 2) AS avg_ms,
       ROUND((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms))::numeric, 2) AS p95_ms
FROM predictions
WHERE created_at > now() - interval '24 hours'
GROUP BY model_version
ORDER BY model_version;

SELECT date_trunc('hour', created_at) AS hour, COUNT(*) AS n_predictions
FROM predictions
GROUP BY hour
ORDER BY hour DESC
LIMIT 24;

SELECT date_trunc('day', created_at)::date AS day,
       ROUND(AVG((inputs->>'med_inc')::float)::numeric, 3) AS avg_requested_income
FROM predictions
GROUP BY day
ORDER BY day DESC
LIMIT 7;