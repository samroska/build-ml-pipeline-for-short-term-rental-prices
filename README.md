# ML Pipeline for Short-Term Rental Prices in NYC

An end-to-end MLflow pipeline that estimates nightly rental prices for NYC properties using a Random Forest model. The pipeline is designed to be rerun on fresh data samples as they arrive weekly.

## Overview

The pipeline covers every stage from raw data to a production-ready model artifact:

1. **download** — fetches a raw CSV sample and logs it to Weights & Biases
2. **basic_cleaning** — filters price outliers, drops properties outside NYC geo bounds, uploads `clean_sample.csv` to W&B
3. **data_check** — runs pytest-based data validation (column schema, neighborhood names, geo boundaries, KL-divergence against a reference dataset, row count, price range)
4. **data_split** — stratified train/val/test split, uploads `trainval_data.csv` and `test_data.csv` to W&B
5. **train_random_forest** — fits a full sklearn `Pipeline` (TF-IDF on listing name, ordinal/one-hot encoding, date feature engineering, zero-imputation) with a `RandomForestRegressor`; logs R² and MAE to W&B
6. **test_regression_model** — evaluates the `prod`-tagged model against the held-out test set (run manually after promoting a model)

## Requirements

- Python 3.13
- conda
- macOS or Ubuntu 22.04/24.04

## Setup

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
wandb login [your API key]
```

## Running the pipeline

Run all steps:

```bash
mlflow run .
```

Run specific steps:

```bash
mlflow run . -P steps=download,basic_cleaning
```

Override config values via Hydra:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.n_estimators=200 modeling.max_tfidf_features=15"
```

Run the test step after promoting a model to `prod` in W&B:

```bash
mlflow run . -P steps=test_regression_model
```

## Configuration

All parameters are in [config.yaml](config.yaml) and managed by Hydra. Key sections:

| Section | Key parameters |
|---|---|
| `etl` | `sample`, `min_price` (10), `max_price` (350) |
| `data_check` | `kl_threshold` (0.2) |
| `modeling` | `test_size`, `val_size`, `random_seed`, `stratify_by`, `max_tfidf_features` |
| `modeling.random_forest` | `n_estimators`, `max_depth`, `max_features`, `criterion`, etc. |

## Model

The inference pipeline is a single sklearn `Pipeline` with two named steps:

- **preprocessor** (`ColumnTransformer`):
  - `room_type` → `OrdinalEncoder`
  - `neighbourhood_group` → `SimpleImputer` + `OneHotEncoder`
  - numeric columns → `SimpleImputer(strategy="constant", fill_value=0)`
  - `last_review` → imputed + converted to days-since-last-review
  - `name` (listing title) → `SimpleImputer` + `TfidfVectorizer`
- **random_forest** → `RandomForestRegressor`

The model is exported in MLflow sklearn format and logged to W&B as a `model_export` artifact.

## Hyperparameter search

Use Hydra multi-run to sweep parameters, for example:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

Select the best run in W&B by sorting on `mae` ascending, then tag the `model_export` artifact as `prod`.

## Re-training on new data

After releasing a version on GitHub, retrain on a new data sample by pointing at the release:

```bash
mlflow run https://github.com/[your github username]/build-ml-pipeline-for-short-term-rental-prices.git \
           -v 1.0.0 \
           -P hydra_options="etl.sample='sample2.csv'"
```

## License

[License](LICENSE.txt)
