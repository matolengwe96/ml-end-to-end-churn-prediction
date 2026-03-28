"""Data loading, cleaning, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAW_DATA_PATH


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw churn CSV with clear error handling."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Expected file: data/raw/churn.csv"
        )
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known issues in Telco churn data."""
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        cleaned = cleaned.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])

    return cleaned


def split_features_target(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned dataset into feature matrix and binary target."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    target_mapped = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if target_mapped.isna().any():
        invalid_values = sorted(df.loc[target_mapped.isna(), target_col].unique().tolist())
        raise ValueError(f"Unexpected target values in '{target_col}': {invalid_values}")

    features = df.drop(columns=[target_col])
    return features, target_mapped.astype(int)


def build_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create reusable sklearn preprocessor for numeric and categorical data."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features
