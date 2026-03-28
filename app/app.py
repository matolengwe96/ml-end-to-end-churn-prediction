"""Streamlit app for telecom churn prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_single


st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="centered")

st.title("Telecom Customer Churn Predictor")
st.caption(
    "Interactive ML application that estimates customer churn risk from telecom account features."
)


def build_input_form() -> pd.DataFrame:
    """Collect model features from UI and return one-row DataFrame."""
    with st.form("churn_form"):
        st.subheader("Customer Profile")

        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "MultipleLines", ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox(
            "OnlineSecurity", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox(
            "DeviceProtection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox(
            "StreamingMovies", ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment_method = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0, step=0.1
        )
        total_charges = st.number_input(
            "TotalCharges", min_value=0.0, max_value=10000.0, value=850.0, step=0.1
        )

        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return pd.DataFrame()

    input_row = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([input_row])


try:
    model = load_model()
except FileNotFoundError as exc:
    st.warning(
        "Trained model not found. Please run training first with: `python -m src.train`."
    )
    st.error(str(exc))
    st.stop()

input_df = build_input_form()
if not input_df.empty:
    try:
        prediction = predict_single(input_df, model=model)
        st.subheader("Prediction Result")
        st.metric("Predicted Segment", prediction["predicted_label"])

        if prediction["churn_probability"] is not None:
            st.metric("Churn Probability", f"{prediction['churn_probability']:.2%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
