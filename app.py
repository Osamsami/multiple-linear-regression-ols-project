"""Streamlit demo for interactive median house value predictions.

Loads the fitted hand-rolled OLS model from ``models/ols_model.pkl`` if
present (from a prior ``python train.py`` run); otherwise trains it inline
on first load. The fit is sub-second on this dataset, so this keeps the app
deployable (e.g. on Streamlit Community Cloud) without needing a committed
binary artifact.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train import train_model

MODEL_PATH = os.path.join("models", "ols_model.pkl")

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# (label, min, max, default, step) — ranges based on the dataset's observed values.
NUMERIC_FIELD_CONFIG = {
    "longitude": ("Longitude", -124.35, -114.31, -119.57, 0.01),
    "latitude": ("Latitude", 32.54, 41.95, 35.63, 0.01),
    "housing_median_age": ("Housing median age (years)", 1.0, 52.0, 28.0, 1.0),
    "total_rooms": ("Total rooms (block group)", 2.0, 40000.0, 2635.0, 1.0),
    "total_bedrooms": ("Total bedrooms (block group)", 1.0, 6500.0, 537.0, 1.0),
    "population": ("Population (block group)", 3.0, 36000.0, 1425.0, 1.0),
    "households": ("Households (block group)", 1.0, 6100.0, 499.0, 1.0),
    "median_income": ("Median income (tens of thousands USD)", 0.5, 15.0, 3.87, 0.01),
}


@st.cache_resource
def load_model(path: str = MODEL_PATH) -> dict:
    if os.path.exists(path):
        return joblib.load(path)

    model = train_model()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
    except OSError:
        pass  # read-only deployment filesystem — fine, we keep the in-memory model
    return model


def build_feature_row(inputs: dict, model: dict) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching the exact column order used
    at training time, including the one-hot encoded ocean_proximity dummies
    (the category dropped at training time maps to all-zero dummies)."""
    row = {name: inputs[name] for name in NUMERIC_FEATURES}
    dummy_prefix = f"{model['categorical_col']}_"
    for col in model["feature_names"]:
        if col.startswith(dummy_prefix):
            category = col[len(dummy_prefix):]
            row[col] = 1.0 if inputs["ocean_proximity"] == category else 0.0
    return pd.DataFrame([row], columns=model["feature_names"])


def predict(inputs: dict, model: dict) -> float:
    X = build_feature_row(inputs, model).to_numpy(dtype=float)
    X_scaled = (X - model["scaler_mean"]) / model["scaler_scale"]
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    return float(X_b.dot(model["beta"])[0])


def main() -> None:
    st.set_page_config(page_title="California Housing Price Predictor", page_icon="\U0001F3E0")
    st.title("California Housing Price Predictor")
    st.caption(
        "Predicts median house value from block-group housing features using a "
        "hand-rolled OLS (normal equation) model — see train.py for how it was fit."
    )

    model = load_model()

    with st.form("prediction_form"):
        st.subheader("Block group features")
        col1, col2 = st.columns(2)
        inputs = {}
        for i, name in enumerate(NUMERIC_FEATURES):
            label, min_val, max_val, default, step = NUMERIC_FIELD_CONFIG[name]
            target_col = col1 if i % 2 == 0 else col2
            inputs[name] = target_col.number_input(
                label, min_value=min_val, max_value=max_val, value=default, step=step
            )

        inputs["ocean_proximity"] = st.selectbox(
            "Ocean proximity", options=model["ocean_categories"]
        )

        submitted = st.form_submit_button("Predict median house value")

    if submitted:
        prediction = predict(inputs, model)
        st.metric("Predicted median house value", f"${prediction:,.0f}")

        metrics = model["metrics"]
        st.caption(
            f"Model evaluation — train RMSE: {metrics['train_rmse']:,.0f}, "
            f"train R²: {metrics['train_r2']:.3f} | "
            f"test RMSE: {metrics['test_rmse']:,.0f}, "
            f"test R²: {metrics['test_r2']:.3f}"
        )

        st.subheader("Fitted model coefficients")
        coef_df = pd.DataFrame(
            {
                "feature": ["bias (intercept)"] + model["feature_names"],
                "coefficient": model["beta"],
            }
        )
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.caption(
            "Coefficients apply to standardized features (each feature is scaled to "
            "zero mean / unit variance before the model is applied)."
        )


if __name__ == "__main__":
    main()
