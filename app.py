"""
Car Price Prediction — Production-style Streamlit Application
CodeAlpha Data Science Internship

A regression service that estimates used-car resale value from a tuned
XGBoost model, trained on real Craigslist vehicle listings.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction | XGBoost",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2563eb"
ACCENT = "#0f172a"
MUTED = "#64748b"

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 2rem; }}
        h1, h2, h3 {{ letter-spacing: -0.02em; }}
        .metric-card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
        }}
        .metric-label {{
            color: {MUTED};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }}
        .metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        .section-tag {{
            display: inline-block;
            background: rgba(37, 99, 235, 0.12);
            color: {PRIMARY};
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.18rem 0.6rem;
            border-radius: 5px;
            margin-bottom: 0.4rem;
        }}
        .footer-note {{
            color: {MUTED};
            font-size: 0.8rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------
# Load model artifacts
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("xgb_model.pkl"),
        joblib.load("encoder.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("num_imputer.pkl"),
        joblib.load("cat_imputer.pkl"),
        joblib.load("unique_values.pkl"),
    )


model, encoder, scaler, num_imputer, cat_imputer, unique_values = load_artifacts()

NUMERICAL_FEATURES = ["year", "odometer"]
CATEGORICAL_FEATURES = ["manufacturer", "condition", "fuel", "transmission", "drive", "type", "state"]


def get_feature_importance():
    try:
        cat_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_names = [f"cat_{i}" for i in range(len(model.feature_importances_) - len(NUMERICAL_FEATURES))]
    names = NUMERICAL_FEATURES + cat_names
    importances = model.feature_importances_
    n = min(len(names), len(importances))
    return pd.DataFrame({"feature": names[:n], "importance": importances[:n]})


# -----------------------------------------------------------------------
# Sidebar — navigation & model card
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Algorithm</div>
            <div class="metric-value" style="font-size:1.1rem;">XGBoost Regressor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        **Pipeline**
        1. Outlier removal (IQR)
        2. Missing-value imputation
        3. One-Hot Encoding (categorical)
        4. Standard Scaling (numerical)
        5. Hyperparameter-tuned XGBoost
        """
    )
    st.write("")
    st.caption("CodeAlpha Data Science Internship — May/June 2026 batch")


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.markdown('<span class="section-tag">Regression Service</span>', unsafe_allow_html=True)
st.title("Car Price Prediction")
st.write(
    "Estimate the fair resale value of a used vehicle from its specifications. "
    "The model was trained on real-world Craigslist listings and tuned for "
    "minimum prediction error across vehicle segments."
)

st.divider()

tab_predict, tab_insights = st.tabs(["Prediction", "Model Insights"])

# -----------------------------------------------------------------------
# TAB 1 — Prediction
# -----------------------------------------------------------------------
with tab_predict:
    col_form, col_output = st.columns([1, 1.1], gap="large")

    with col_form:
        st.subheader("Vehicle Specifications")

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Year", min_value=1980, max_value=2026, value=2015, step=1)
        with c2:
            odometer = st.number_input("Odometer (mi)", min_value=0, max_value=400000, value=60000, step=1000)

        manufacturer = st.selectbox("Manufacturer", unique_values.get("manufacturer", []))
        condition = st.selectbox("Condition", unique_values.get("condition", []))

        c3, c4 = st.columns(2)
        with c3:
            fuel = st.selectbox("Fuel Type", unique_values.get("fuel", []))
        with c4:
            transmission = st.selectbox("Transmission", unique_values.get("transmission", []))

        c5, c6 = st.columns(2)
        with c5:
            drive = st.selectbox("Drive Type", unique_values.get("drive", []))
        with c6:
            car_type = st.selectbox("Body Type", unique_values.get("type", []))

        state = st.selectbox("State", unique_values.get("state", []))

        st.write("")
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Estimated Value")

        if predict_btn:
            row = pd.DataFrame([{
                "year": year, "odometer": odometer, "manufacturer": manufacturer,
                "condition": condition, "fuel": fuel, "transmission": transmission,
                "drive": drive, "type": car_type, "state": state,
            }])

            row[NUMERICAL_FEATURES] = num_imputer.transform(row[NUMERICAL_FEATURES])
            row[CATEGORICAL_FEATURES] = cat_imputer.transform(row[CATEGORICAL_FEATURES])

            num_part = scaler.transform(row[NUMERICAL_FEATURES])
            cat_part = encoder.transform(row[CATEGORICAL_FEATURES])
            final_features = np.hstack([num_part, cat_part])

            predicted_price = float(model.predict(final_features)[0])
            margin = predicted_price * 0.08

            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; padding:1.6rem;">
                    <div class="metric-label">Predicted Price</div>
                    <div class="metric-value" style="font-size:2.4rem;">${predicted_price:,.0f}</div>
                    <div class="footer-note">Estimated range: ${predicted_price - margin:,.0f} – ${predicted_price + margin:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown("**Vehicle Age vs. Mileage Context**")

            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.scatter([2026 - year], [odometer], s=120, color=PRIMARY, zorder=3, label="This vehicle")
            ax.set_xlabel("Vehicle Age (years)")
            ax.set_ylabel("Odometer (mi)")
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            st.pyplot(fig, use_container_width=True)

        else:
            st.info("Enter the vehicle specifications and click **Run Prediction** to see an estimate.")

# -----------------------------------------------------------------------
# TAB 2 — Model Insights
# -----------------------------------------------------------------------
with tab_insights:
    st.subheader("Feature Importance")
    st.write(
        "Relative contribution of each input feature to the model's pricing decisions, "
        "derived directly from the trained XGBoost ensemble."
    )

    importance_df = get_feature_importance().sort_values("importance", ascending=True).tail(12)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(importance_df["feature"], importance_df["importance"], color=PRIMARY)
    ax2.set_xlabel("Importance")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    st.pyplot(fig2, use_container_width=True)

    st.divider()

    st.subheader("Modeling Pipeline")
    st.markdown(
        """
        ```
        Raw Listings  →  Outlier Removal (IQR)  →  Imputation
                       →  Encoding / Scaling  →  Model Selection
                       →  Hyperparameter Tuning  →  XGBoost Regressor
        ```
        Eight regression algorithms were benchmarked during development —
        Linear Regression, Ridge, Lasso, KNN, Decision Tree, Random Forest,
        Gradient Boosting, and XGBoost — with XGBoost selected as the final
        model after hyperparameter tuning based on test-set R² and RMSE.
        """
    )

st.divider()
st.markdown(
    '<p class="footer-note">Built as part of the CodeAlpha Data Science Internship. '
    "Estimates are statistical approximations and should not be treated as formal appraisals.</p>",
    unsafe_allow_html=True,
)
