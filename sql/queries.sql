SELECT batch, COUNT(*) AS n_rows, ROUND(AVG(price)::numeric, 3) AS avg_price,
       ROUND(AVG(med_inc)::numeric, 3) AS avg_income
FROM listings
GROUP BY batch
ORDER BY batch;

SELECT CASE WHEN med_inc < 3 THEN 'low' WHEN med_inc < 6 THEN 'middle' ELSE 'high' END AS income_bracket,
       COUNT(*) AS n_rows, ROUND(AVG(price)::numeric, 3) AS avg_price
FROM listings
GROUP BY income_bracket
ORDER BY avg_price;

SELECT id, batch, price, med_inc, house_age
FROM listings
ORDER BY price DESC
LIMIT 10;