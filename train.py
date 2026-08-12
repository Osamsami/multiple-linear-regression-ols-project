"""Fit the hand-rolled OLS (normal equation) model and persist it to disk.

This mirrors the preprocessing and modeling steps used in
``notebooks/mlr_ols_project.ipynb`` as a standalone, non-interactive script:
load the raw dataset, impute missing values, one-hot encode the categorical
feature, split into train/test sets, standardize, fit the closed-form OLS
solution on the training set, and report train/test RMSE and R^2.

The fitted coefficients plus the scaler statistics and feature metadata
needed to reproduce predictions are saved to ``models/ols_model.pkl`` so
``app.py`` (the Streamlit demo) can load them without retraining.

Usage:
    python train.py
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "raw", "housing.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ols_model.pkl")

TARGET_COL = "median_house_value"
CATEGORICAL_COL = "ocean_proximity"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and impute the one column with missing values."""
    df = pd.read_csv(path)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode the categorical column and split into X / y."""
    df_encoded = pd.get_dummies(df, columns=[CATEGORICAL_COL], drop_first=True)
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]
    return X, y


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the intercept is fit as part of beta."""
    return np.c_[np.ones((X.shape[0], 1)), X]


def fit_ols(X_b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equation: beta = (X^T X)^-1 X^T y."""
    return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


def train_model(data_path: str = DATA_PATH) -> dict:
    """Run the full fit and return the artifact dict app.py needs — without
    touching disk. Lets app.py train on demand (the fit is sub-second on this
    dataset) when no cached models/ols_model.pkl is present, e.g. on a fresh
    Streamlit Cloud deployment."""
    df = load_data(data_path)
    X, y = build_features(df)
    feature_names = X.columns.tolist()
    ocean_categories = sorted(df[CATEGORICAL_COL].unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_b = add_bias(X_train_scaled)
    X_test_b = add_bias(X_test_scaled)

    beta = fit_ols(X_train_b, y_train.to_numpy())

    y_train_pred = X_train_b.dot(beta)
    y_test_pred = X_test_b.dot(beta)

    return {
        "beta": beta,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "categorical_col": CATEGORICAL_COL,
        "ocean_categories": ocean_categories,
        "metrics": {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "test_r2": float(r2_score(y_test, y_test_pred)),
        },
    }


def main() -> None:
    model = train_model()
    metrics = model["metrics"]

    print(f"Train RMSE: {metrics['train_rmse']:,.2f}")
    print(f"Train R2:   {metrics['train_r2']:.4f}")
    print(f"Test RMSE:  {metrics['test_rmse']:,.2f}")
    print(f"Test R2:    {metrics['test_r2']:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved fitted model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
