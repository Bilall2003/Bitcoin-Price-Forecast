"""
Bitcoin Price Analysis & Prediction — Streamlit Edition
Converted from a Tkinter desktop app to a modern, professional Streamlit dashboard.
"""

import os
import time
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

# ========================================================================================
# CORE LOGIC (kept modular & separate from UI, same spirit as the original OOP design)
# ========================================================================================

class EDA:
    """Handles cleaning + descriptive analysis of the uploaded Bitcoin dataset."""

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.df = None

    def clean(self) -> pd.DataFrame:
        df = self.raw_df.copy()
        before_rows = len(df)

        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df.drop_duplicates(keep="first", inplace=True)
            df.reset_index(drop=True, inplace=True)

        na_count = df.isnull().sum().sum()
        if na_count > 0:
            df.dropna(inplace=True)

        self.df = df
        self.stats = {
            "rows_before": before_rows,
            "rows_after": len(df),
            "duplicates_removed": int(dup_count),
            "missing_removed": int(na_count),
        }
        return self.df

    def analysis(self):
        if self.df is None:
            raise ValueError("Data must be cleaned before analysis.")

        df = self.df.copy()
        if "Date" not in df.columns:
            raise ValueError("Column 'Date' not found in dataset.")

        df["Date"] = pd.to_datetime(df["Date"]).dt.year

        gr1 = df.groupby("Date").agg({"High": "max", "Low": "min"})
        gr2 = df.groupby("Date").agg({"Open": "max", "Close": "min"})
        gr3 = df.groupby("Date")["Volume"].mean().reset_index()
        gr4 = df.describe()

        self.df_yearly = df
        return gr1, gr2, gr3, gr4

    def visualize(self, gr1: pd.DataFrame, gr2: pd.DataFrame, plot_type: str):
        fig = plt.figure(figsize=(10, 5.5), dpi=110)
        palette_bg = "#0f172a"
        fig.patch.set_facecolor(palette_bg)
        ax = plt.gca()
        ax.set_facecolor(palette_bg)

        if plot_type == "High/Low Prices":
            melted = gr1.reset_index().melt(
                id_vars="Date", value_vars=["High", "Low"],
                var_name="Price Type", value_name="Value"
            )
            sns.barplot(x="Date", y="Value", hue="Price Type", data=melted, palette=["#4e8cff", "#22d3ee"])
            plt.title("High and Low of Bitcoin by Year", fontsize=14, color="white")
            plt.xticks(rotation=45, color="white")
            plt.yticks(color="white")
            plt.ylabel("Price (USD)", color="white")
            plt.xlabel("Year", color="white")

        elif plot_type == "Open/Close Prices":
            melted = gr2.reset_index().melt(
                id_vars="Date", value_vars=["Open", "Close"],
                var_name="Price Type", value_name="Value"
            )
            sns.barplot(x="Date", y="Value", hue="Price Type", data=melted, palette=["#818cf8", "#60a5fa"])
            plt.title("Open and Close of Bitcoin by Year", fontsize=14, color="white")
            plt.xticks(rotation=45, color="white")
            plt.yticks(color="white")
            plt.ylabel("Price (USD)", color="white")
            plt.xlabel("Year", color="white")

        elif plot_type == "Volume Trend":
            sns.lineplot(x=self.df_yearly["Date"], y=self.df_yearly["Volume"], color="#38bdf8", linewidth=2)
            plt.title("Average Trading Volume by Year", fontsize=14, color="white")
            plt.xlabel("Year", color="white")
            plt.ylabel("Volume", color="white")
            plt.xticks(color="white")
            plt.yticks(color="white")

        elif plot_type == "Correlation Heatmap":
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="mako", linewidths=0.5, cbar=True)
            plt.title("Correlation Heatmap", fontsize=14, color="white")
            plt.xticks(color="white")
            plt.yticks(color="white")

        plt.tight_layout()
        return fig


class ML:
    """Trains regression models to predict Bitcoin High/Low prices."""

    MODEL_MAP = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self, model_name: str):
        df = self.df.copy()

        X1 = df.drop(columns=["Date", "High"], axis=1, errors="ignore")
        Y1 = df["High"]
        x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, random_state=101, train_size=0.7)

        X2 = df.drop(columns=["Date", "Low"], axis=1, errors="ignore")
        Y2 = df["Low"]
        x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y2, random_state=42, train_size=0.7)

        model = self.MODEL_MAP[model_name]

        model.fit(x_train1, y_train1)
        preds1 = model.predict(x_test1)
        mae1 = mean_absolute_error(y_test1, preds1)
        mse1 = mean_squared_error(y_test1, preds1)
        r2_1 = r2_score(y_test1, preds1)

        model.fit(x_train2, y_train2)
        preds2 = model.predict(x_test2)
        mae2 = mean_absolute_error(y_test2, preds2)
        mse2 = mean_squared_error(y_test2, preds2)
        r2_2 = r2_score(y_test2, preds2)

        comparison = pd.DataFrame({
            "Open": x_test1["Open"].values if "Open" in x_test1 else np.nan,
            "Close": x_test1["Close"].values if "Close" in x_test1 else np.nan,
            "Volume": x_test1["Volume"].values if "Volume" in x_test1 else np.nan,
            "High (actual)": y_test1.values,
            "High (predicted)": preds1,
            "Low (actual)": y_test2.values,
            "Low (predicted)": preds2,
        })

        return {
            "model": model_name,
            "high_mae": mae1, "high_mse": mse1, "high_r2": r2_1,
            "low_mae": mae2, "low_mse": mse2, "low_r2": r2_2,
            "comparison": comparison,
            "high_actual": y_test1.values, "high_pred": preds1,
            "low_actual": y_test2.values, "low_pred": preds2,
        }

    @staticmethod
    def prediction_plot(actual, predicted, title):
        fig = plt.figure(figsize=(10, 4.2), dpi=110)
        bg = "#0f172a"
        fig.patch.set_facecolor(bg)
        ax = plt.gca()
        ax.set_facecolor(bg)
        plt.plot(actual, label="Actual", marker="o", markersize=4, color="#38bdf8")
        plt.plot(predicted, label="Predicted", marker="x", markersize=4, color="#f472b6")
        plt.title(title, fontsize=12, color="white")
        plt.xlabel("Sample", fontsize=10, color="white")
        plt.ylabel("Price (USD)", fontsize=10, color="white")
        plt.xticks(color="white")
        plt.yticks(color="white")
        plt.legend(facecolor="#1e293b", labelcolor="white")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        return fig


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.set_page_config(
    page_title="Bitcoin Analysis & Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------- Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
    }

    /* Hero header */
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
        border: 1px solid rgba(99, 179, 237, 0.25);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #93c5fd;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111c34;
        border: 1px solid rgba(99, 179, 237, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: #93c5fd !important; }
    div[data-testid="stMetricValue"] { color: #f8fafc !important; }

    /* Section cards */
    .card {
        background: #111c34;
        border: 1px solid rgba(99, 179, 237, 0.18);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .card h3 {
        color: #dbeafe;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #93c5fd;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc !important;
        border-bottom-color: #4e8cff !important;
    }

    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.1rem;
    }
    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(99, 179, 237, 0.18);
        border-radius: 10px;
        overflow: hidden;
    }

    .badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.18);
        color: #93c5fd;
        border: 1px solid rgba(99, 179, 237, 0.35);
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------- Session state
for key, default in {
    "raw_df": None, "clean_df": None, "eda_results": None,
    "ml_results": {}, "eda_obj": None, "file_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------- Header
st.markdown("""
<div class="hero">
    <h1>₿ Bitcoin Price Analysis &amp; Prediction</h1>
    <p>Exploratory data analysis and machine learning price prediction — a Streamlit dashboard</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded_file = st.file_uploader("Upload Bitcoin CSV file", type=["csv"])

    if uploaded_file is not None:
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.raw_df = pd.read_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.clean_df = None
            st.session_state.eda_results = None
            st.session_state.ml_results = {}
            st.session_state.eda_obj = None

    st.markdown("---")
    if st.session_state.raw_df is not None:
        st.success(f"Loaded: **{st.session_state.file_name}**")
        st.caption(f"{st.session_state.raw_df.shape[0]} rows · {st.session_state.raw_df.shape[1]} columns")
    else:
        st.info("No file loaded yet.")

    st.markdown("---")
    st.markdown("### ⚙️ About")
    st.caption(
        "Pipeline: **Clean → EDA → Visualize → ML Prediction**. "
        "Expected columns: `Date, Open, High, Low, Close, Adj Close, Volume`."
    )

# ---------------------------------------------------------------------------- Guard
if st.session_state.raw_df is None:
    st.markdown('<div class="card"><h3>👋 Get started</h3>'
                'Upload a Bitcoin historical price CSV from the sidebar to begin. '
                'The file should include <code>Date, Open, High, Low, Close, Adj Close, Volume</code> columns.'
                '</div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------- Tabs
tab_file, tab_eda, tab_ml, tab_results = st.tabs(
    ["📄 File Operations", "🔍 Exploratory Analysis", "🤖 Machine Learning", "📊 Results"]
)

# ============================================================ TAB 1: FILE OPERATIONS
with tab_file:
    st.markdown('<div class="card"><h3>Raw Data Preview</h3></div>', unsafe_allow_html=True)
    st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        clean_clicked = st.button("🧹 Clean File", use_container_width=True)
    with col2:
        save_clicked = st.button("💾 Save Cleaned CSV", use_container_width=True)

    if clean_clicked:
        with st.spinner("Cleaning data..."):
            start = time.time()
            eda_obj = EDA(st.session_state.raw_df)
            cleaned = eda_obj.clean()
            st.session_state.clean_df = cleaned
            st.session_state.eda_obj = eda_obj
            elapsed = time.time() - start
        st.success(f"File cleaned in {elapsed:.2f}s")

    if st.session_state.clean_df is not None:
        stats = st.session_state.eda_obj.stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows before", stats["rows_before"])
        m2.metric("Rows after", stats["rows_after"])
        m3.metric("Duplicates removed", stats["duplicates_removed"])
        m4.metric("Missing removed", stats["missing_removed"])

        st.markdown('<div class="card"><h3>Cleaned Data Preview</h3></div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)

        if save_clicked:
            out_path = "bitcoin_mod.csv"
            st.session_state.clean_df.to_csv(out_path, index=False)
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download bitcoin_mod.csv", data=f, file_name="bitcoin_mod.csv",
                    mime="text/csv", use_container_width=True
                )
    elif save_clicked:
        st.warning("Please clean the file first.")

# ============================================================ TAB 2: EDA
with tab_eda:
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please clean the file in the **File Operations** tab first.")
    else:
        run_eda = st.button("🔍 Perform EDA", use_container_width=False)

        if run_eda:
            with st.spinner("Running exploratory analysis..."):
                start = time.time()
                try:
                    gr1, gr2, gr3, gr4 = st.session_state.eda_obj.analysis()
                    st.session_state.eda_results = (gr1, gr2, gr3, gr4)
                    st.success(f"EDA completed in {time.time() - start:.2f}s")
                except Exception as e:
                    st.error(f"EDA error: {e}")

        if st.session_state.eda_results is not None:
            gr1, gr2, gr3, gr4 = st.session_state.eda_results

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="card"><h3>High / Low by Year</h3></div>', unsafe_allow_html=True)
                st.dataframe(gr1, use_container_width=True)
            with c2:
                st.markdown('<div class="card"><h3>Open / Close by Year</h3></div>', unsafe_allow_html=True)
                st.dataframe(gr2, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<div class="card"><h3>Average Volume by Year</h3></div>', unsafe_allow_html=True)
                st.dataframe(gr3, use_container_width=True)
            with c4:
                st.markdown('<div class="card"><h3>Descriptive Statistics</h3></div>', unsafe_allow_html=True)
                st.dataframe(gr4, use_container_width=True)

            st.markdown('<div class="card"><h3>📈 Visualization</h3></div>', unsafe_allow_html=True)
            vis_choice = st.selectbox(
                "Choose a chart",
                ["High/Low Prices", "Open/Close Prices", "Volume Trend", "Correlation Heatmap"],
            )
            if st.button("Show Visualization"):
                with st.spinner("Rendering chart..."):
                    fig = st.session_state.eda_obj.visualize(gr1, gr2, vis_choice)
                    st.pyplot(fig, use_container_width=True)

# ============================================================ TAB 3: ML
with tab_ml:
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please clean the file in the **File Operations** tab first.")
    elif not all(c in st.session_state.clean_df.columns for c in ["High", "Low", "Date"]):
        st.error("Dataset must contain 'Date', 'High', and 'Low' columns for prediction.")
    else:
        st.markdown('<div class="card"><h3>Model Selection</h3></div>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Choose a regression model",
            ["Random Forest", "Linear Regression", "Decision Tree"],
            horizontal=True,
        )

        if st.button("🚀 Run ML Model", use_container_width=False):
            with st.spinner(f"Training {model_choice}..."):
                start = time.time()
                try:
                    ml_engine = ML(st.session_state.clean_df)
                    result = ml_engine.run(model_choice)
                    st.session_state.ml_results[model_choice] = result
                    st.success(f"{model_choice} completed in {time.time() - start:.2f}s")
                except Exception as e:
                    st.error(f"ML error: {e}")

        if model_choice in st.session_state.ml_results:
            res = st.session_state.ml_results[model_choice]

            st.markdown(f'<div class="card"><h3>{model_choice} — Metrics</h3></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**High Price Prediction**")
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{res['high_mae']:.2f}")
                m2.metric("MSE", f"{res['high_mse']:.2e}")
                m3.metric("R²", f"{res['high_r2']:.4f}")
            with c2:
                st.markdown("**Low Price Prediction**")
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{res['low_mae']:.2f}")
                m2.metric("MSE", f"{res['low_mse']:.2e}")
                m3.metric("R²", f"{res['low_r2']:.4f}")

            st.markdown('<div class="card"><h3>Sample Predictions</h3></div>', unsafe_allow_html=True)
            st.dataframe(res["comparison"].head(10), use_container_width=True)

# ============================================================ TAB 4: RESULTS
with tab_results:
    if not st.session_state.ml_results:
        st.warning("⚠️ Run at least one ML model in the **Machine Learning** tab first.")
    else:
        st.markdown('<div class="card"><h3>📊 Model Comparison</h3></div>', unsafe_allow_html=True)

        compare_rows = []
        for name, res in st.session_state.ml_results.items():
            compare_rows.append({
                "Model": name,
                "High MAE": round(res["high_mae"], 3),
                "High R²": round(res["high_r2"], 4),
                "Low MAE": round(res["low_mae"], 3),
                "Low R²": round(res["low_r2"], 4),
            })
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

        chosen = st.selectbox("View detailed prediction plots for:", list(st.session_state.ml_results.keys()))
        res = st.session_state.ml_results[chosen]

        st.markdown(f'<div class="card"><h3>{chosen} — High Price: Actual vs Predicted</h3></div>', unsafe_allow_html=True)
        fig1 = ML.prediction_plot(res["high_actual"], res["high_pred"], f"{chosen} — High Price Prediction")
        st.pyplot(fig1, use_container_width=True)

        st.markdown(f'<div class="card"><h3>{chosen} — Low Price: Actual vs Predicted</h3></div>', unsafe_allow_html=True)
        fig2 = ML.prediction_plot(res["low_actual"], res["low_pred"], f"{chosen} — Low Price Prediction")
        st.pyplot(fig2, use_container_width=True)

# ---------------------------------------------------------------------------- Footer
st.markdown("""
<div style="text-align:center; color:#64748b; padding: 1.5rem 0 0.5rem 0; font-size:0.82rem;">
    Bitcoin Analysis &amp; Prediction Dashboard · Built with Streamlit
</div>
""", unsafe_allow_html=True)