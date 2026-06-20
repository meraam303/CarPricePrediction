"""
Car Price Prediction — Interactive Streamlit App
CodeAlpha Data Science Internship
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    num_imputer = joblib.load("num_imputer.pkl")
    cat_imputer = joblib.load("cat_imputer.pkl")
    unique_values = joblib.load("unique_values.pkl")
    return model, encoder, scaler, num_imputer, cat_imputer, unique_values


model, encoder, scaler, num_imputer, cat_imputer, unique_values = load_artifacts()

NUMERICAL_FEATURES = ["year", "odometer"]
CATEGORICAL_FEATURES = ["manufacturer", "condition", "fuel", "transmission", "drive", "type", "state"]

st.title("🚗 Car Price Prediction")
st.markdown(
    "Estimate the resale price of a used car based on its specifications, "
    "using a tuned **XGBoost Regressor** trained on real Craigslist vehicle listings."
)

st.divider()

col_input, col_result = st.columns([1, 1.2])

with col_input:
    st.subheader("🔧 Car Details")

    year = st.number_input("Year", min_value=1980, max_value=2026, value=2015, step=1)
    odometer = st.number_input("Odometer (miles)", min_value=0, max_value=400000, value=60000, step=1000)

    manufacturer = st.selectbox("Manufacturer", unique_values.get("manufacturer", []))
    condition = st.selectbox("Condition", unique_values.get("condition", []))
    fuel = st.selectbox("Fuel Type", unique_values.get("fuel", []))
    transmission = st.selectbox("Transmission", unique_values.get("transmission", []))
    drive = st.selectbox("Drive Type", unique_values.get("drive", []))
    car_type = st.selectbox("Vehicle Type", unique_values.get("type", []))
    state = st.selectbox("State", unique_values.get("state", []))

    predict_btn = st.button("💰 Predict Price", type="primary", use_container_width=True)

with col_result:
    st.subheader("🎯 Predicted Price")

    if predict_btn:
        row = pd.DataFrame([{
            "year": year,
            "odometer": odometer,
            "manufacturer": manufacturer,
            "condition": condition,
            "fuel": fuel,
            "transmission": transmission,
            "drive": drive,
            "type": car_type,
            "state": state,
        }])

        row[NUMERICAL_FEATURES] = num_imputer.transform(row[NUMERICAL_FEATURES])
        row[CATEGORICAL_FEATURES] = cat_imputer.transform(row[CATEGORICAL_FEATURES])

        num_part = scaler.transform(row[NUMERICAL_FEATURES])
        cat_part = encoder.transform(row[CATEGORICAL_FEATURES])

        final_features = np.hstack([num_part, cat_part])
        predicted_price = model.predict(final_features)[0]

        st.success(f"### Estimated Price: ${predicted_price:,.0f}")
        st.caption("This is an estimate based on historical listing data and may not reflect exact market value.")
    else:
        st.write("👈 Fill in the car details and click **Predict Price**.")

st.divider()

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        - **Dataset**: Craigslist used-vehicle listings
        - **Preprocessing**: Outlier removal (IQR), missing-value imputation,
          One-Hot Encoding for categorical features, Standard Scaling for numerical features
        - **Models compared**: Linear Regression, Ridge, Lasso, KNN, Decision Tree,
          Random Forest, Gradient Boosting, XGBoost
        - **Final model**: Tuned XGBoost Regressor
        - Built as part of the **CodeAlpha Data Science Internship**
        """
    )
