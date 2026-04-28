# app.py
import streamlit as st
import pandas as pd

from data_utils import load_and_validate_data
from forecasting import evaluate_on_holdout, fit_and_forecast, make_future_index
from plotting import (
    plot_historical_series,
    plot_holdout_forecast,
    plot_future_forecast
)

# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Weekly High/Critical Vulnerability Count Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------
# Custom CSS + Background Image
# -----------------------------------
st.markdown("""
    <style>

    /* ── Background image ── */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("https://plus.unsplash.com/premium_photo-1681426694953-4d78658193dc?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.08;
        z-index: 0;
        pointer-events: none;
    }
    [data-testid="stAppViewContainer"] > * {
        position: relative;
        z-index: 1;
    }

    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f0f4f8;
        color: #1a202c;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #eef2ff 100%);
        border-right: 2px solid #c7d2fe;
    }
    [data-testid="stSidebar"] * {
        color: #1e1b4b !important;
    }

    /* ── Sidebar inner padding + hidden scrollbar ── */
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        padding: 0rem 1.2rem 1rem 1.2rem !important;
    }
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        display: none !important;
    }

    /* ── Collapse sidebar spacing ── */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stMultiSelect,
    [data-testid="stSidebar"] .stFileUploader {
        margin-bottom: 0rem !important;
    }
    [data-testid="stSidebar"] .stRadio > div,
    [data-testid="stSidebar"] .stMultiSelect > div {
        gap: 0.2rem !important;
    }

    /* ── Default sidebar label size ── */
    [data-testid="stSidebar"] label {
        font-size: 0.78rem !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* ── Default sidebar p size — applied to all p tags first ── */
    [data-testid="stSidebar"] p {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.78rem !important;
    }

    /* ── Dashboard Mode radio option text (first radio)
          MUST come after the p rule above to override it ── */
    [data-testid="stSidebar"] .stRadio:first-of-type p {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #3730a3 !important;
        line-height: 1.8 !important;
        margin: 0 !important;
    }

    /* ── Data Source radio option text (second radio) ── */
    [data-testid="stSidebar"] .stRadio:nth-of-type(2) p {
        font-size: 0.84rem !important;
        font-weight: 400 !important;
        color: #1e1b4b !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.4rem 0 !important;
        border-color: #c7d2fe !important;
    }

    /* ── Sidebar header ── */
    .sidebar-header {
        text-align: center;
        padding: 0rem 0 0rem 0;
    }
    .sidebar-header h2 {
        color: #4338ca !important;
        font-size: 1.4rem !important;
        font-weight: 800;
        margin: 0rem 0 0 0;
    }
    .sidebar-header p {
        color: #6366f1 !important;
        font-size: 0.8rem !important;
        margin: 0 !important;
    }

    /* ── Sidebar section labels ── */
    .sidebar-label {
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        color: #4338ca !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 0.4rem 0 0.15rem 0;
        display: block;
    }

    /* ── Sidebar Run button ── */
    [data-testid="stSidebar"] .stButton > button {
        background: #ffffff;
        color: #4f46e5 !important;
        border: 1.5px solid #6366f1 !important;
        border-radius: 8px;
        padding: 0.42rem 1rem;
        font-weight: 600;
        font-size: 0.88rem;
        width: 100%;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        box-shadow: none;
        margin-top: 0.4rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #eef2ff;
        color: #3730a3 !important;
        border-color: #4f46e5 !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.18);
        transform: translateY(-1px);
    }

    /* ── Multiselect tags ── */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #eef2ff !important;
        border: 1px solid #c7d2fe !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #4338ca !important;
        font-size: 0.78rem !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] button {
        color: #6366f1 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] button:hover {
        color: #3730a3 !important;
        background: #e0e7ff !important;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #ede9fe 100%);
        border: 1.5px solid #c7d2fe;
        border-radius: 16px;
        padding: 1.6rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(99,102,241,0.10);
    }
    .hero-banner h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #3730a3;
        margin-bottom: 0.35rem;
    }
    .hero-banner p {
        color: #6366f1;
        font-size: 0.93rem;
        margin: 0;
    }

    /* ── Mode strip ── */
    .mode-strip {
        background: #ffffff;
        border: 1.5px solid #c7d2fe;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.08);
    }
    .mode-strip h3 {
        color: #4338ca !important;
        margin-bottom: 0.5rem;
    }

    /* ── Headings ── */
    h2 { color: #4338ca !important; font-weight: 700; }
    h3 { color: #4f46e5 !important; font-weight: 700; }
    h4 { color: #6366f1 !important; }

    /* ── Section cards ── */
    .section-card {
        background: #ffffff;
        border: 1.5px solid #e0e7ff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.07);
    }
    .section-card p, .section-card li {
        color: #374151;
        line-height: 1.75;
    }
    .section-card h4 {
        color: #4f46e5 !important;
        margin-bottom: 0.5rem;
    }

    /* ── Result header accent ── */
    .result-header {
        background: linear-gradient(90deg, #eef2ff, #f5f3ff);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 0.65rem 1.2rem;
        margin: 1.4rem 0 0.8rem 0;
        color: #4338ca !important;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1.5px solid #c7d2fe;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.1);
    }
    [data-testid="metric-container"] label {
        color: #6b7280 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #4f46e5 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        font-size: 0.93rem;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border: 1.5px solid #e0e7ff;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 3px 10px rgba(79,70,229,0.3) !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #6366f1, #818cf8) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Code blocks ── */
    code {
        background: #eef2ff !important;
        color: #4f46e5 !important;
        border-radius: 4px;
        padding: 2px 6px;
    }
    pre {
        background: #eef2ff !important;
        border: 1.5px solid #c7d2fe;
        border-radius: 8px;
    }

    /* ── Markdown ── */
    .stMarkdown p, .stMarkdown li {
        color: #374151;
        line-height: 1.75;
    }

    /* ── Dividers ── */
    hr { border-color: #e0e7ff; margin: 1.2rem 0; }

    /* ── Scrollbar (main page) ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #f0f4f8; }
    ::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6366f1; }

    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# Global settings
# -----------------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "dataset", "ycrit_weekly_w_mon_copy.csv")

AVAILABLE_HORIZONS = [1, 4, 12, 26, 52]

# Friendly display labels shown to user in sidebar and all outputs
AVAILABLE_MODELS = [
    "Baseline: Seasonal Naive",
    "Model 1: ETS",
    "Model 2: SARIMA",
    "Model 3: Prophet"
]

# Maps friendly label → internal name passed to forecasting functions
MODEL_INTERNAL_NAME = {
    "Baseline: Seasonal Naive": "Seasonal Naive",
    "Model 1: ETS":             "ETS",
    "Model 2: SARIMA":          "SARIMA",
    "Model 3: Prophet":         "Prophet"
}

# ============================================================
# SIDEBAR — always rendered
# ============================================================
st.sidebar.markdown("""
    <div class="sidebar-header">
        <span style="font-size:1.6rem;">📈</span>
        <h2>Forecast Controls</h2>
        <p>Configure and run your forecast</p>
    </div>
    <hr/>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-label">🗂️ Dashboard Mode</p>', unsafe_allow_html=True)
mode = st.sidebar.radio(
    "mode",
    [
        "Mode A: Evaluation Mode",
        "Mode B: Forecast Mode",
        "About / Instructions"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown('<hr/>', unsafe_allow_html=True)

# Forecast controls only shown for Mode A and B
if mode in ("Mode A: Evaluation Mode", "Mode B: Forecast Mode"):

    st.sidebar.markdown('<p class="sidebar-label">📂 Data Source</p>', unsafe_allow_html=True)
    data_source = st.sidebar.radio(
        "data_source",
        ["Use default dissertation dataset", "Upload my own CSV"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload my own CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    st.sidebar.markdown('<hr/>', unsafe_allow_html=True)

    st.sidebar.markdown('<p class="sidebar-label">🤖 Models</p>', unsafe_allow_html=True)
    selected_models = st.sidebar.multiselect(
        "models",
        AVAILABLE_MODELS,
        default=["Model 1: ETS", "Model 2: SARIMA"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown('<p class="sidebar-label">📅 Horizons (weeks)</p>', unsafe_allow_html=True)
    selected_horizons = st.sidebar.multiselect(
        "horizons",
        AVAILABLE_HORIZONS,
        default=[4, 12],
        label_visibility="collapsed"
    )

    st.sidebar.markdown('<hr/>', unsafe_allow_html=True)
    run_action = st.sidebar.button("▶  Run Forecast")

else:
    # About mode — sidebar shows a gentle nudge only
    st.sidebar.markdown("""
        <p style="font-size:0.82rem; color:#a5b4fc; font-style:italic; text-align:center; padding:0.5rem 0;">
            Select Mode A or Mode B<br/>to activate forecast controls.
        </p>
    """, unsafe_allow_html=True)
    run_action = False
    selected_models = []
    selected_horizons = []
    uploaded_file = None
    data_source = "Use default dissertation dataset"

# ============================================================
# MAIN PANEL — Hero banner always shown
# ============================================================
st.markdown("""
    <div class="hero-banner">
        <h1>📈 Weekly High/Critical Vulnerability Count Forecast Dashboard</h1>
        <p>
            Evaluate forecasting models on historical holdout data &nbsp;|&nbsp;
            Generate future vulnerability count forecasts &nbsp;|&nbsp;
            Upload your own weekly dataset
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# ABOUT / INSTRUCTIONS PAGE
# ============================================================
if mode == "About / Instructions":

    st.markdown("---")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📖 About This Dashboard")
    st.write(
        "This application allows users to explore weekly forecasting models "
        "for critical vulnerability disclosure counts. The dashboard supports "
        "both model evaluation on historical holdout data and forward forecasting "
        "into future weeks."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔀 Dashboard Modes")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div class="section-card">
                <h4>📊 Mode A — Evaluation Mode</h4>
                <p>
                    Use this mode to compare model performance on a holdout portion
                    of the dataset. The final <em>h</em> weeks are treated as unseen
                    test data. Because actual holdout values are known, this mode reports:
                </p>
                <ul>
                    <li>MAE</li>
                    <li>RMSE</li>
                    <li>Actual vs forecast plots</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div class="section-card">
                <h4>🔭 Mode B — Forecast Mode</h4>
                <p>
                    Use this mode to generate future forecasts from the full available
                    dataset. Models are fit on all observed data and forecast ahead for
                    chosen horizons. Because future true values are unknown, MAE and
                    RMSE are <strong>not</strong> reported.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_models, col_horizons = st.columns(2)
    with col_models:
        st.subheader("🤖 Supported Models")
        st.markdown("""
            <div class="section-card">
                <ul>
                    <li>🔹 <strong>Baseline:</strong> Seasonal Naive</li>
                    <li>🔹 <strong>Model 1:</strong> ETS (Exponential Smoothing)</li>
                    <li>🔹 <strong>Model 2:</strong> SARIMA <em style="color:#6b7280;">(constrained search)</em></li>
                    <li>🔹 <strong>Model 3:</strong> Prophet</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col_horizons:
        st.subheader("📅 Supported Horizons")
        st.markdown("""
            <div class="section-card">
                <ul>
                    <li>🗓️ 1 week</li>
                    <li>🗓️ 4 weeks</li>
                    <li>🗓️ 12 weeks</li>
                    <li>🗓️ 26 weeks</li>
                    <li>🗓️ 52 weeks</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 Uploaded CSV Requirements")
    st.markdown("""
        <div class="section-card">
            <p>If uploading your own dataset, the CSV must contain:</p>
            <ul>
                <li><code>week</code> — weekly datetime values</li>
                <li><code>ycrit</code> — numeric weekly target values</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.code("week,ycrit\n2019-01-07,12\n2019-01-14,70\n2019-01-21,84", language="csv")

    st.markdown("""
        <div class="section-card">
            <ul>
                <li><code>week</code> must be parseable as dates</li>
                <li><code>ycrit</code> must be numeric</li>
                <li>Duplicate weeks are not allowed</li>
                <li>Dataset should contain enough weekly observations for seasonal forecasting</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 How to Use the Dashboard")
    st.markdown("""
        <div class="section-card">
            <ol>
                <li>Select <strong>Mode A</strong> or <strong>Mode B</strong> in the sidebar</li>
                <li>Choose whether to use the default dataset or upload your own CSV</li>
                <li>Select one or more forecasting models</li>
                <li>Select one or more forecast horizons</li>
                <li>Click <strong>▶ Run Forecast</strong></li>
                <li>Review the plots and tables produced by the dashboard</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODE A OR MODE B — main panel
# ============================================================
elif mode in ("Mode A: Evaluation Mode", "Mode B: Forecast Mode"):

    if mode == "Mode A: Evaluation Mode":
        st.info(
            "📊 **Mode A — Evaluation Mode:** The last *h* weeks are treated as unseen test data, "
            "where *h* is equivalent to the number of weeks (forecast horizon) selected. "
            "MAE and RMSE are computed because the true observed values are known."
        )
    else:
        st.info(
            "🔭 **Mode B — Forecast Mode:** Models are fit on the full available dataset "
            "and forecast ahead for selected horizons. MAE and RMSE are not shown because "
            "future observed values are not yet available."
        )

    if run_action:
        try:
            df_app = load_and_validate_data(DEFAULT_DATA_PATH, uploaded_file)

            st.success("✅ Dataset loaded and validated successfully.")
            st.markdown("---")

            st.subheader("📋 Input Data Preview")
            st.dataframe(df_app.tail(10), width="stretch")

            st.subheader("📊 Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Observations", len(df_app))
            col2.metric("Start Date", str(df_app.index.min().date()))
            col3.metric("End Date", str(df_app.index.max().date()))

            st.markdown("---")

            st.subheader("📈 Historical Weekly Series")
            fig_hist = plot_historical_series(df_app)
            st.plotly_chart(fig_hist, width="stretch")

            if len(selected_models) == 0:
                st.warning("⚠️ Please select at least one model.")
                st.stop()

            if len(selected_horizons) == 0:
                st.warning("⚠️ Please select at least one horizon.")
                st.stop()

            # ── MODE A ──────────────────────────────────────────
            if mode == "Mode A: Evaluation Mode":
                metric_rows = []
                progress_text = st.empty()

                for horizon in selected_horizons:
                    st.markdown(
                        f'<div class="result-header">📊 Holdout Evaluation — {horizon}-Week Horizon</div>',
                        unsafe_allow_html=True
                    )

                    for model_label in selected_models:
                        internal_name = MODEL_INTERNAL_NAME[model_label]

                        progress_text.info(
                            f"⏳ Evaluating **{model_label}** at **{horizon}-week** horizon…"
                        )

                        actual_index, actual_values, pred_values, intervals, mae, r = evaluate_on_holdout(
                            df_app["ycrit"],
                            internal_name,
                            horizon
                        )

                        metric_rows.append({
                            "Model": model_label,
                            "Horizon_Weeks": horizon,
                            "MAE": mae,
                            "RMSE": r
                        })

                        c1, c2 = st.columns(2)
                        c1.metric(f"{model_label} MAE ({horizon}w)", f"{mae:.3f}")
                        c2.metric(f"{model_label} RMSE ({horizon}w)", f"{r:.3f}")

                        fig_eval = plot_holdout_forecast(
                            actual_index,
                            actual_values,
                            pred_values,
                            intervals,
                            model_label,
                            horizon
                        )
                        st.plotly_chart(fig_eval, width="stretch")

                progress_text.success("✅ All selected evaluations completed.")

                metrics_df = pd.DataFrame(metric_rows).sort_values(["Horizon_Weeks", "MAE"])

                st.markdown("---")
                st.subheader("📊 Model Comparison Table")
                st.dataframe(metrics_df, width="stretch")

                csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Evaluation Metrics CSV",
                    data=csv_bytes,
                    file_name="evaluation_metrics.csv",
                    mime="text/csv"
                )

            # ── MODE B ──────────────────────────────────────────
            else:
                progress_text = st.empty()

                for horizon in selected_horizons:
                    st.markdown(
                        f'<div class="result-header">🔭 Future Forecast — {horizon}-Week Horizon</div>',
                        unsafe_allow_html=True
                    )

                    future_index = make_future_index(df_app.index.max(), horizon)
                    forecast_df = pd.DataFrame(index=future_index)

                    for model_label in selected_models:
                        internal_name = MODEL_INTERNAL_NAME[model_label]

                        progress_text.info(
                            f"⏳ Forecasting with **{model_label}** at **{horizon}-week** horizon…"
                        )

                        pred_values, intervals = fit_and_forecast(
                            df_app["ycrit"],
                            internal_name,
                            horizon
                        )

                        forecast_df[model_label] = pred_values

                        fig_forecast = plot_future_forecast(
                            df_app["ycrit"].tail(26).index,
                            df_app["ycrit"].tail(26).values,
                            future_index,
                            pred_values,
                            intervals,
                            model_label,
                            horizon
                        )
                        st.plotly_chart(fig_forecast, width="stretch")

                    progress_text.success("✅ All selected future forecasts completed.")

                    st.markdown("---")
                    st.subheader(f"📋 Forecast Table — {horizon}-Week Horizon")
                    st.dataframe(forecast_df, width="stretch")

                    csv_bytes = forecast_df.to_csv().encode("utf-8")
                    st.download_button(
                        label=f"⬇️  Download Forecast CSV ({horizon} weeks)",
                        data=csv_bytes,
                        file_name=f"forecast_{horizon}_weeks.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error: {e}")