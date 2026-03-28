# Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts telecom customer churn and exposes predictions through a Streamlit web app.

## Project Overview

Customer churn is a critical business problem in telecom because acquiring a new customer is often more expensive than retaining an existing one. This project builds a practical churn scoring workflow that can help prioritize retention actions.

The application includes:

- reliable data loading and cleaning for the Telco Customer Churn dataset
- reusable preprocessing with sklearn pipelines
- multi-model training and comparison
- model artifact saving for repeatable inference
- an interactive Streamlit prediction app
- automated tests for core training and prediction logic

## Business Problem

Given a customer profile (contract type, internet service, billing behavior, tenure, and monthly charges), predict whether the customer is likely to churn.

Target:

- `Churn = Yes` (1): likely to leave
- `Churn = No` (0): likely to stay

Primary model selection metric:

- **F1 score** (balances precision and recall for churn detection)
- ROC-AUC used as tie-breaker when F1 is very close

## Dataset

- Source file: `data/raw/churn.csv`
- Dataset: Telco Customer Churn
- Key handling rule implemented:
  - `TotalCharges` converted to numeric with `errors="coerce"`
  - rows with invalid `TotalCharges` dropped
  - `customerID` dropped before modeling

## Project Structure

```text
.
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── churn.csv
│   └── processed/
├── models/
│   ├── best_model.joblib
│   ├── model_metrics.json
│   └── feature_columns.json
├── notebooks/
│   └── eda.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_predict.py
│   └── test_train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Workflow

1. Load and clean raw data
2. Split features/target and map churn labels to binary
3. Build preprocessing with `ColumnTransformer`
4. Train and compare:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
5. Select best model by F1 (ROC-AUC tie-breaker)
6. Save model and metrics
7. Serve predictions through Streamlit app

## Installation

From repository root:

```bash
python -m venv .venv
```

Activate environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python -m src.train
```

Artifacts created:

- `models/best_model.joblib`
- `models/model_metrics.json`
- `models/feature_columns.json`

## Run the Streamlit App

```bash
streamlit run app/app.py
```

If a trained model is missing, the app shows a clear instruction to run training first.

## Run Tests

```bash
pytest -q
```

## Sample Prediction Workflow (Python)

```python
from src.predict import load_model, predict_single

model = load_model()

sample_customer = {
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

print(predict_single(sample_customer, model=model))
```

## Model Training Summary

Training metrics are written to `models/model_metrics.json` on each run. The best model is selected based on churn-oriented F1 score.

## Future Improvements

- add cross-validation and hyperparameter tuning
- add model calibration and threshold optimization for retention campaigns
- add data validation checks for production inference
- package the app with Docker for deployment
- add CI pipeline for tests and linting
