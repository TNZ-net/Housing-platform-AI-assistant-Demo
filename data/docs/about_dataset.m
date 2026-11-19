# About the housing data and this platform

## The dataset
The data is the California housing dataset, built from the 1990 U.S. census. It has 20,640 rows. Each row describes one census block group, the smallest geographic unit for which the census publishes sample data (typically 600 to 3,000 people).

## Columns
- med_inc: median income in the block group, in units of $10,000.
- house_age: median age of the houses, in years.
- ave_rooms: average number of rooms per household.
- ave_bedrms: average number of bedrooms per household.
- population: number of people living in the block group.
- ave_occup: average number of household members.
- latitude and longitude: where the block group is.
- price: median house value, in units of $100,000.

## Known limits
Prices are capped at $500,000, so the highest value in the data is 5.00001 (the cap), and a model cannot learn to predict above it. Rooms and bedrooms are averages per household, so they can look strange where there are few households, such as vacation areas. The data is from 1990, so predictions do not reflect today's prices.

## How the platform handles the data
The data is loaded in 12 batches, one batch per run of the ingest job, to simulate new data arriving each month. Rows are rejected during ingest if a value is missing, if price or rooms are not positive, or if latitude or longitude are out of range. Running the same batch twice adds nothing, because every loaded batch is recorded in the ingest log and skipped next time.

## The model
The price model is a gradient boosted tree model (HistGradientBoostingRegressor from scikit-learn). It is trained on all batches except the newest and tested on the newest batch, so the test score is not flattered by rows the model has already seen. Each trained model is saved as a numbered version (v1, v2, ...) with its scores. The score to watch is MAE, the average error in units of $100,000: an MAE of 0.4 means about $40,000 of error. The model is compared with a baseline that always guesses the average price.

## Drift
Drift means new data looks different from the data the model was trained on. The platform measures it with PSI (Population Stability Index) for each feature. Below 0.1 means stable, 0.1 to 0.25 means a small shift, and above 0.25 means a large shift that suggests retraining.

## The API
The prediction API answers POST /predict with a price for one block, GET /health to check it is running, GET /model for the serving model version and its scores, and GET /stats for the last 24 hours of predictions. Every prediction is logged to the database with its model version and response time.