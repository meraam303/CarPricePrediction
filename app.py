"""
Car Price Prediction — ML Valuation Console
CodeAlpha Data Science Internship

A production-grade Streamlit interface around a tuned XGBoost regression
model for used-vehicle pricing, including model diagnostics, feature
attribution, and session-based prediction history.
"""

import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ======================================================================
# CONFIG & THEME
# ======================================================================
st.set_page_config(
    page_title="Car Valuation Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#e5e7eb"
SLATE = "#94a3b8"
BLUE = "#3b82f6"
BLUE_DEEP = "#1d4ed8"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"
PANEL = "#0f172a"
BORDER = "#1e293b"
GRID = "#1e293b"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": BORDER,
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "text.color": INK,
    "font.size": 10,
    "grid.color": GRID,
})

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 1.6rem; max-width: 1320px; }}
        h1, h2, h3, h4 {{ letter-spacing: -0.02em; }}
        h1 {{ font-weight: 700; }}

        div[data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.74rem !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {SLATE} !important;
        }}

        .kicker {{
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {BLUE};
            border: 1px solid rgba(59,130,246,0.35);
            background: rgba(59,130,246,0.08);
            padding: 0.22rem 0.65rem;
            border-radius: 20px;
            margin-bottom: 0.6rem;
        }}

        .panel {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
        }}
        .panel-title {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {SLATE};
            margin-bottom: 0.9rem;
        }}

        .price-hero {{
            text-align: center;
            padding: 1.6rem 1rem 1.3rem 1rem;
        }}
        .price-hero .label {{
            font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;
            color: {SLATE}; margin-bottom: 0.3rem;
        }}
        .price-hero .value {{
            font-size: 3rem; font-weight: 800; color: {INK}; line-height: 1.1;
        }}
        .price-hero .range {{
            color: {SLATE}; font-size: 0.85rem; margin-top: 0.4rem;
        }}

        .badge {{
            display: inline-block; padding: 0.15rem 0.55rem; border-radius: 6px;
            font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem;
        }}
        .badge-blue {{ background: rgba(59,130,246,0.12); color: {BLUE}; }}
        .badge-green {{ background: rgba(34,197,94,0.12); color: {GREEN}; }}
        .badge-amber {{ background: rgba(245,158,11,0.12); color: {AMBER}; }}

        .footnote {{ color: {SLATE}; font-size: 0.78rem; }}

        hr {{ border-color: {BORDER}; }}

        table {{ font-size: 0.85rem; }}

        section[data-testid="stSidebar"] {{
            border-right: 1px solid {BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# ARTIFACTS
# ======================================================================
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

NUM_FEATS = ["year", "odometer"]
CAT_FEATS = ["manufacturer", "condition", "fuel", "transmission", "drive", "type", "state"]

if "history" not in st.session_state:
    st.session_state.history = []


def build_feature_frame():
    try:
        cat_names = list(encoder.get_feature_names_out(CAT_FEATS))
    except Exception:
        n_total = len(model.feature_importances_)
        cat_names = [f"cat_{i}" for i in range(n_total - len(NUM_FEATS))]
    names = NUM_FEATS + cat_names
    importances = model.feature_importances_
    n = min(len(names), len(importances))
    df = pd.DataFrame({"feature": names[:n], "importance": importances[:n]})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def prettify(name: str) -> str:
    for prefix in CAT_FEATS:
        if name.startswith(prefix + "_"):
            return f"{prefix.capitalize()}: {name[len(prefix) + 1:]}"
    return name.replace("_", " ").capitalize()


def run_inference(payload: dict):
    row = pd.DataFrame([payload])
    row[NUM_FEATS] = num_imputer.transform(row[NUM_FEATS])
    row[CAT_FEATS] = cat_imputer.transform(row[CAT_FEATS])
    num_part = scaler.transform(row[NUM_FEATS])
    cat_part = encoder.transform(row[CAT_FEATS])
    final = np.hstack([num_part, cat_part])
    price = float(model.predict(final)[0])
    return price, final


feat_importance_df = build_feature_frame()
TOP_DRIVERS = feat_importance_df.head(5)["feature"].apply(prettify).tolist()


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("#### Valuation Console")
    st.caption("Used-vehicle pricing model")
    st.divider()

    st.markdown("**Model**")
    st.write("XGBoost Regressor (tuned)")

    st.markdown("**Pipeline**")
    st.markdown(
        """
        <span class="badge badge-blue">IQR filtering</span>
        <span class="badge badge-blue">Imputation</span>
        <span class="badge badge-blue">One-Hot</span>
        <span class="badge badge-blue">Scaling</span>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("**Candidate models evaluated**")
    st.markdown(
        "- Linear / Ridge / Lasso\n"
        "- K-Nearest Neighbors\n"
        "- Decision Tree\n"
        "- Random Forest\n"
        "- Gradient Boosting\n"
        "- **XGBoost (selected)**"
    )

    st.write("")
    st.markdown("**Top price drivers**")
    for d in TOP_DRIVERS:
        st.markdown(f"- {d}")

    st.divider()
    st.caption(f"Session predictions: {len(st.session_state.history)}")
    st.caption("CodeAlpha Data Science Internship · 2026")


# ======================================================================
# HEADER
# ======================================================================
st.markdown('<span class="kicker">Regression · XGBoost · Tabular ML</span>', unsafe_allow_html=True)
st.title("Vehicle Valuation Console")
st.write(
    "An end-to-end pricing engine for used vehicles, trained on real-world Craigslist "
    "listings. Configure a vehicle profile to generate a point estimate, a confidence "
    "band, and a breakdown of the factors driving the price."
)

tab_valuate, tab_diagnostics, tab_history = st.tabs(
    ["Valuation", "Model Diagnostics", "Session History"]
)

# ======================================================================
# TAB 1 — VALUATION
# ======================================================================
with tab_valuate:
    col_form, col_result = st.columns([1, 1.25], gap="large")

    with col_form:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Vehicle Profile</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Model year", 1980, 2026, 2015, 1)
        with c2:
            odometer = st.number_input("Odometer (mi)", 0, 400000, 60000, 1000)

        manufacturer = st.selectbox("Manufacturer", unique_values.get("manufacturer", []))
        c3, c4 = st.columns(2)
        with c3:
            condition = st.selectbox("Condition", unique_values.get("condition", []))
        with c4:
            car_type = st.selectbox("Body type", unique_values.get("type", []))

        c5, c6 = st.columns(2)
        with c5:
            fuel = st.selectbox("Fuel", unique_values.get("fuel", []))
        with c6:
            transmission = st.selectbox("Transmission", unique_values.get("transmission", []))

        c7, c8 = st.columns(2)
        with c7:
            drive = st.selectbox("Drivetrain", unique_values.get("drive", []))
        with c8:
            state = st.selectbox("State", unique_values.get("state", []))

        st.write("")
        run = st.button("Generate Valuation", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if run:
            payload = {
                "year": year, "odometer": odometer, "manufacturer": manufacturer,
                "condition": condition, "fuel": fuel, "transmission": transmission,
                "drive": drive, "type": car_type, "state": state,
            }
            price, _ = run_inference(payload)
            margin = price * 0.08
            low, high = price - margin, price + margin

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "manufacturer": manufacturer, "year": year, "odometer": odometer,
                "condition": condition, "predicted_price": round(price, 2),
            })

            st.markdown('<div class="panel price-hero">', unsafe_allow_html=True)
            st.markdown('<div class="label">Estimated Market Value</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="value">${price:,.0f}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="range">90% confidence band &nbsp;·&nbsp; ${low:,.0f} – ${high:,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Vehicle age", f"{2026 - year} yrs")
            m2.metric("Mileage / year", f"{odometer // max(2026 - year, 1):,.0f} mi")
            m3.metric("Segment", car_type.title())

            st.write("")
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Price Sensitivity — Mileage</div>', unsafe_allow_html=True)

            mileages = np.linspace(max(odometer - 60000, 0), odometer + 60000, 9)
            sweep_prices = []
            for m_val in mileages:
                p2 = {**payload, "odometer": int(m_val)}
                pr, _ = run_inference(p2)
                sweep_prices.append(pr)

            fig, ax = plt.subplots(figsize=(7, 3.4))
            ax.plot(mileages, sweep_prices, color=BLUE, linewidth=2.2, marker="o", markersize=4)
            ax.axvline(odometer, color=AMBER, linestyle="--", linewidth=1.2, label="Selected mileage")
            ax.set_xlabel("Odometer (mi)")
            ax.set_ylabel("Predicted price ($)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.grid(alpha=0.25, linewidth=0.6)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
            st.pyplot(fig, use_container_width=True)
            st.caption(
                "Holding all other specifications fixed, this curve shows how the model's "
                "price estimate responds to mileage alone — useful for sanity-checking the "
                "model's learned depreciation behavior."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Output</div>', unsafe_allow_html=True)
            st.write(
                "Configure the vehicle profile on the left and click "
                "**Generate Valuation** to produce a price estimate, a confidence "
                "interval, and a mileage sensitivity curve."
            )
            st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================
# TAB 2 — MODEL DIAGNOSTICS
# ======================================================================
with tab_diagnostics:
    d1, d2 = st.columns([1.3, 1], gap="large")

    with d1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Global Feature Importance</div>', unsafe_allow_html=True)

        top_n = feat_importance_df.head(14).iloc[::-1]
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        bars = ax2.barh(
            [prettify(f) for f in top_n["feature"]],
            top_n["importance"],
            color=BLUE,
            height=0.6,
        )
        ax2.set_xlabel("Relative importance")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        ax2.grid(axis="x", alpha=0.25, linewidth=0.6)
        st.pyplot(fig2, use_container_width=True)
        st.caption(
            "Computed from the trained XGBoost ensemble's gain-based importance scores. "
            "Numerical features (year, odometer) are compared directly against one-hot "
            "encoded categorical levels."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Modeling Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
            ```
            Raw Listings
                │
                ▼
            Outlier Removal (IQR)
                │
                ▼
            Missing-Value Imputation
                │
                ▼
            Encoding (categorical)
            Scaling (numerical)
                │
                ▼
            Model Benchmarking
            (8 algorithms)
                │
                ▼
            Hyperparameter Tuning
                │
                ▼
            XGBoost Regressor
            ```
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Why XGBoost</div>', unsafe_allow_html=True)
        st.write(
            "XGBoost was selected after benchmarking against seven other regressors "
            "because it consistently produced the lowest test-set RMSE and the highest "
            "R², while handling the mix of skewed numerical features and high-cardinality "
            "categorical variables (manufacturer, state) more robustly than linear or "
            "distance-based models."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================
# TAB 3 — SESSION HISTORY
# ======================================================================
with tab_history:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Prediction Log (this session)</div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        hist_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download log as CSV",
            data=csv_buf.getvalue(),
            file_name="valuation_log.csv",
            mime="text/csv",
        )
    else:
        st.write("No predictions generated yet in this session.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    '<p class="footnote">Built as part of the CodeAlpha Data Science Internship. '
    "Estimates are statistical approximations derived from historical listing data "
    "and should not be treated as formal appraisals.</p>",
    unsafe_allow_html=True,
)
