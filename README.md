# Telecom Churn Prediction Platform

Enterprise-grade, portfolio-ready machine learning project that predicts telecom customer churn and serves predictions through a production-style Streamlit interface.

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
7. Test suite and CI pipeline for quality gates

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

## Project structure

```text
.
├── .github/workflows/ci.yml
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
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
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
5. Persist best model and metadata to [models](models)

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

### 5) Run tests

```bash
pytest
```

### 6) Run quality checks

```bash
ruff check src tests app
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

## CI/CD quality gate

Pipeline defined in [ci.yml](.github/workflows/ci.yml):

1. Install dependencies
2. Lint with Ruff
3. Run Pytest suite
4. Execute training smoke test

## Assumptions

1. Training and inference use the same data schema as Telco churn source.
2. Binary classification threshold defaults to model default (`0.5` for probabilistic classifiers).
3. Current deployment target is local Streamlit execution.

## Next enterprise upgrades

1. Add experiment tracking (MLflow)
2. Add model registry + promotion workflow
3. Containerize app and training jobs
4. Add scheduled retraining and drift monitoring
5. Add API serving layer (FastAPI) for integration with CRM systems
