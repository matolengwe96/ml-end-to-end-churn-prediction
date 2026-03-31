"""Streamlit dashboard for the Telecom Customer Churn Predictor.

Layout
• 🔮 Predict   — Input form, prediction result, and session history.
• 📊 Model Metrics — Hold-out and CV metrics for every trained model.
• 🔍 Explain   — SHAP feature-importance for the last prediction.
• ℹ️  About      — Business context and quick-start commands.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import METRICS_PATH, MODEL_METADATA_PATH
from src.explainability import explain_prediction
from src.predict import load_model, predict_single
from src.utils import load_json

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📉",
    layout="centered",
)

st.title("📉 Telecom Customer Churn Predictor")
st.caption(
    "Enterprise ML application that estimates customer churn risk "
    "from telecom account features."
)


# ── Cached resources ───────────────────────────────────────────────────────────────────


@st.cache_resource
def load_cached_model():
    """Load and cache the trained model pipeline to reduce UI latency."""
    return load_model()


# ── Guard: model must be trained before the UI is useful─────────────────────────
try:
    model = load_cached_model()
except FileNotFoundError as exc:
    st.warning(
        "Trained model not found. Run training first: `python -m src.train`"
    )
    st.error(str(exc))
    st.stop()

metadata: dict = {}
if MODEL_METADATA_PATH.exists():
    raw_meta = load_json(MODEL_METADATA_PATH)
    if isinstance(raw_meta, dict):
        metadata = raw_meta

# Prediction history persisted across Streamlit reruns.
if "history" not in st.session_state:
    st.session_state.history = []


# ── Tab layout ────────────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_explain, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Metrics", "🔍 Explain", "ℹ️ About"]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Predict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_predict:
    if metadata:
        with st.expander("ℹ️ Model info", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Best model", metadata.get("best_model", "—"))
            c2.metric("Dataset rows", metadata.get("dataset_rows", "—"))
            c1.metric("Feature count", metadata.get("feature_count", "—"))
            trained_at = metadata.get("trained_at_utc", "")
            c2.metric("Trained (UTC)", trained_at[:19] if trained_at else "—")

    def build_input_form() -> pd.DataFrame:
        """Render the input form and return a one-row DataFrame on submit."""
        with st.form("churn_form"):
            st.subheader("Customer Profile")

            col_left, col_right = st.columns(2)

            with col_left:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider(
                    "Tenure (months)", min_value=0, max_value=72, value=12
                )
                contract = st.selectbox(
                    "Contract", ["Month-to-month", "One year", "Two year"]
                )
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )

            with col_right:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox(
                    "Multiple Lines", ["Yes", "No", "No phone service"]
                )
                internet_service = st.selectbox(
                    "Internet Service", ["DSL", "Fiber optic", "No"]
                )
                online_security = st.selectbox(
                    "Online Security", ["Yes", "No", "No internet service"]
                )
                online_backup = st.selectbox(
                    "Online Backup", ["Yes", "No", "No internet service"]
                )
                device_protection = st.selectbox(
                    "Device Protection", ["Yes", "No", "No internet service"]
                )
                tech_support = st.selectbox(
                    "Tech Support", ["Yes", "No", "No internet service"]
                )
                streaming_tv = st.selectbox(
                    "Streaming TV", ["Yes", "No", "No internet service"]
                )
                streaming_movies = st.selectbox(
                    "Streaming Movies", ["Yes", "No", "No internet service"]
                )

            st.subheader("Billing")
            bc1, bc2 = st.columns(2)
            monthly_charges = bc1.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=0.1,
            )
            total_charges = bc2.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=850.0,
                step=0.1,
            )

            submitted = st.form_submit_button(
                "🔮 Predict Churn Risk", use_container_width=True
            )

        if not submitted:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
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
            ]
        )

    input_df = build_input_form()

    if not input_df.empty:
        try:
            prediction = predict_single(input_df, model=model)
            churn_prob = prediction["churn_probability"]
            label = prediction["predicted_label"]

            st.subheader("Prediction Result")

            if prediction["predicted_class"] == 1:
                st.error(f"⚠️ **{label}** — This customer is at risk of churning.")
            else:
                st.success(f"✅ **{label}** — This customer is likely to stay.")

            if churn_prob is not None:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
                st.progress(churn_prob, text=f"Risk level: {churn_prob:.1%}")

            # Append to in-session history.
            st.session_state.history.insert(
                0,
                {
                    "Tenure": input_df.at[0, "tenure"],
                    "Contract": input_df.at[0, "Contract"],
                    "Internet": input_df.at[0, "InternetService"],
                    "Monthly ($)": input_df.at[0, "MonthlyCharges"],
                    "Label": label,
                    "Probability": f"{churn_prob:.1%}" if churn_prob is not None else "N/A",
                },
            )
            # Cache for the Explain tab.
            st.session_state.last_input_df = input_df
            st.session_state.last_prediction = prediction
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if st.session_state.history:
        st.divider()
        st.subheader("Prediction History (this session)")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Model Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_metrics:
    st.subheader("Trained Model Comparison")

    if METRICS_PATH.exists():
        raw_metrics = load_json(METRICS_PATH)
        best = metadata.get("best_model")

        if isinstance(raw_metrics, dict):
            rows = []
            for model_name, m in raw_metrics.items():
                tag = "⭐ " if model_name == best else ""
                rows.append(
                    {
                        "Model": f"{tag}{model_name}",
                        "Accuracy": f"{m.get('accuracy', 0):.3f}",
                        "Precision": f"{m.get('precision', 0):.3f}",
                        "Recall": f"{m.get('recall', 0):.3f}",
                        "F1": f"{m.get('f1', 0):.3f}",
                        "ROC-AUC": (
                            f"{m['roc_auc']:.3f}" if m.get("roc_auc") else "—"
                        ),
                        "CV F1 Mean": (
                            f"{m['cv_f1_mean']:.3f}" if m.get("cv_f1_mean") else "—"
                        ),
                        "CV F1 ±Std": (
                            f"±{m['cv_f1_std']:.3f}" if m.get("cv_f1_std") else "—"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Best model selected by F1 score (ROC-AUC as tie-breaker). "
                "CV F1 is 5-fold cross-validation on the training partition."
            )
    else:
        st.info("No metrics found. Train the model first: `python -m src.train`")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Explain (SHAP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_explain:
    st.subheader("🔍 Feature Contributions (SHAP)")
    st.caption(
        "SHAP explains *why* the model made its decision by attributing "
        "the probability shift to each input feature."
    )
    last_input_df = st.session_state.get("last_input_df")
    last_pred = st.session_state.get("last_prediction")
    if last_input_df is None:
        st.info("Submit a prediction on the **🔮 Predict** tab first.")
    else:
        with st.spinner("Computing SHAP values …"):
            explanation = explain_prediction(model, last_input_df, top_n=15)

        if not explanation["shap_available"]:
            st.warning(
                "SHAP is not installed. Install it with `pip install shap` "
                "to enable explainability."
            )
        else:
            top_features = explanation["top_features"]
            base_value = explanation["base_value"]

            churn_prob_val = (last_pred or {}).get("churn_probability") or 0.0
            label_val = (last_pred or {}).get("predicted_label", "—")

            st.metric("Prediction", label_val, delta=f"{churn_prob_val:.1%} churn probability")
            if base_value is not None:
                st.caption(f"Model baseline (expected) value: **{base_value:.4f}**")

            if top_features:
                import matplotlib.pyplot as plt  # noqa: PLC0415

                features = [f["feature"] for f in top_features]
                values = [f["shap_value"] for f in top_features]
                colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

                fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.45)))
                bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_xlabel("SHAP value (impact on log-odds of churn)")
                ax.set_title("Feature contributions to this prediction")
                ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(
                    "**Red bars** push the prediction toward churn.  "
                    "**Green bars** push it away from churn."
                )

                with st.expander("Raw SHAP data", expanded=False):
                    st.dataframe(
                        pd.DataFrame(top_features),
                        use_container_width=True,
                        hide_index=True,
                    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: About
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_about:
    st.subheader("About This Project")
    st.markdown(
        """
### Business Problem
Customer churn directly impacts telecom revenue.  Identifying at-risk customers
early allows targeted retention campaigns, reducing acquisition cost and
protecting customer lifetime value.

### ML Pipeline
| Step | Detail |
|---|---|
| **Data** | Telco Customer Churn dataset (~7,043 records, 20 features) |
| **Preprocessing** | Median imputation • Standard scaling • One-hot encoding |
| **Models compared** | Logistic Regression • Random Forest • Gradient Boosting |
| **Selection** | Best F1, tie-broken by ROC-AUC |
| **Validation** | 80/20 stratified split + 5-fold CV |

### Architecture
```
src/          Core pipeline (config, preprocessing, evaluation, training, prediction)
api/          FastAPI REST service  →  POST /predict
app/          This Streamlit dashboard
tests/        Pytest — unit, API (FastAPI TestClient), and integration tests
```

### Quick Start
```bash
python -m src.train             # Train and save model
streamlit run app/app.py        # Launch this dashboard
uvicorn api.main:app --reload   # Start the REST API
pytest --cov=src --cov=api      # Run tests with coverage
```
        """
    )

