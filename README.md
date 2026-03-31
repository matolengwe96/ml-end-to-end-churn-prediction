# Telecom Churn Prediction Platform

Enterprise-grade, portfolio-ready machine learning project that predicts telecom customer churn and serves predictions through a production-style Streamlit interface and a hardened FastAPI service.

## Why this project matters

In telecom, churn directly impacts revenue, customer lifetime value, and acquisition cost. The goal is to identify high-risk customers early so retention teams can intervene with the right offer before cancellation.

## Solution overview

This repository implements a complete ML workflow:

1. Reliable data ingestion and cleaning for Telco churn data
2. Reusable preprocessing pipeline (imputation + scaling + one-hot encoding)
3. Multi-model training and evaluation
4. Best-model selection by F1 with ROC-AUC tie-breaker
5. Artifact persistence for reproducible inference
6. Streamlit app for business-friendly scoring
7. FastAPI service for system-to-system scoring
8. Drift monitoring, experiment tracking, and CI quality gates

## Dataset

- Source: [data/raw/churn.csv](data/raw/churn.csv)
- Target: Churn (`Yes` -> `1`, `No` -> `0`)
- Key cleaning decisions:

1. `TotalCharges` is converted using `errors="coerce"`
2. rows with invalid `TotalCharges` are removed
3. `customerID` is excluded from modeling

## Enterprise features included

1. Centralized runtime settings with environment overrides
2. Structured logging for training workflows
3. Inference input validation (required-column checks + feature alignment)
4. Rich model artifacts:

- trained pipeline
- model metrics
- transformed feature names
- training metadata (row counts, timestamp, selected model)

5. CI automation with lint, tests, and training smoke run
6. Tooling for reproducible development (`pyproject.toml`, `Makefile`, `ruff`, `pytest`)
7. Optional MLflow experiment tracking for training runs
8. Inference monitoring and drift reporting from saved prediction logs
9. API hardening with request IDs, optional API keys, and rate limiting
10. Deployment configuration for Docker, Render, and Azure Developer CLI

## Project structure

```text
.
├── .github/workflows/ci.yml
├── api/
│   └── main.py
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── churn.csv
│   └── processed/
├── models/
│   ├── best_model.joblib
│   ├── feature_columns.json
│   ├── model_metadata.json
│   └── model_metrics.json
├── notebooks/
│   └── eda.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── drift_monitoring.py
│   ├── evaluate.py
│   ├── experiment_tracking.py
│   ├── feature_engineering.py
│   ├── monitoring.py
│   ├── predict.py
│   ├── schemas.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_api.py
│   ├── test_integration.py
│   ├── test_monitoring.py
│   ├── test_predict.py
│   └── test_train.py
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Model training workflow

1. Load and clean data using [src/data_preprocessing.py](src/data_preprocessing.py)
2. Split train/test with reproducible seed from [src/config.py](src/config.py)
3. Train model candidates in [src/train.py](src/train.py):

- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier

4. Evaluate using [src/evaluate.py](src/evaluate.py)
5. Log the run to MLflow in [src/experiment_tracking.py](src/experiment_tracking.py)
6. Persist best model, metadata, and training baseline to [models](models)

## Quick start

### 1) Create and activate virtual environment

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train models

```bash
python -m src.train
```

Optional custom dataset path:

```bash
python -m src.train --data-path data/raw/churn.csv
```

### 4) Run app

```bash
streamlit run app/app.py
```

### 5) Run API

```bash
uvicorn api.main:app --reload --port 8000
```

### 6) Run tests

```bash
pytest
```

### 7) Run quality checks

```bash
ruff check src tests app api
```

### 8) Open MLflow UI

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Local training logs metrics and small artifacts to MLflow by default, but skips
serializing the full sklearn model into the MLflow run because file-backed
stores on Windows can become extremely slow. To opt into full MLflow model
logging when you need it:

```bash
# PowerShell
$env:CHURN_MLFLOW_LOG_MODEL = "1"
python -m src.train
```

### 9) Generate a drift report from logged predictions

```bash
make drift-report
```

## Example prediction usage

```python
from src.predict import load_model, predict_single

model = load_model()

customer = {
   "gender": "Female",
   "SeniorCitizen": 0,
   "Partner": "Yes",
   "Dependents": "No",
   "tenure": 12,
   "PhoneService": "Yes",
   "MultipleLines": "No",
   "InternetService": "DSL",
   "OnlineSecurity": "Yes",
   "OnlineBackup": "No",
   "DeviceProtection": "No",
   "TechSupport": "Yes",
   "StreamingTV": "No",
   "StreamingMovies": "No",
   "Contract": "Month-to-month",
   "PaperlessBilling": "Yes",
   "PaymentMethod": "Electronic check",
   "MonthlyCharges": 65.5,
   "TotalCharges": 780.0,
}

print(predict_single(customer, model=model))
```

## API production notes

Inference endpoints support enterprise-oriented controls in [api/main.py](api/main.py):

1. `x-request-id` propagation for request tracing
2. Optional `x-api-key` authentication via `CHURN_API_KEY`
3. In-memory per-IP rate limiting via `CHURN_API_RATE_LIMIT`
4. Monitoring endpoint at `/drift` for inference-vs-training comparison

## CI/CD quality gate

Pipeline defined in [ci.yml](.github/workflows/ci.yml):

1. Install dependencies
2. Lint with Ruff
3. Run Pytest suite
4. Execute training smoke test

## Deployment options

1. Docker: [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml)
2. Render: [render.yaml](render.yaml)
3. Azure Developer CLI / App Service: [azure.yaml](azure.yaml)

## Assumptions

1. Training and inference use the same data schema as Telco churn source.
2. Binary classification threshold defaults to model default (`0.5` for probabilistic classifiers).
3. Current deployment target is local Streamlit execution.

## Next enterprise upgrades

1. Add a real model registry and stage promotion policy on top of MLflow
2. Replace in-memory rate limiting with Redis-backed distributed throttling
3. Add automated retraining/orchestration with Prefect or GitHub Actions schedules
4. Emit prediction and drift metrics to Prometheus/Grafana or a cloud APM stack
