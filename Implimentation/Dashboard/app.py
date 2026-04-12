# app.py
# Main Streamlit application with three homepage options:
# 1. Mode A: Evaluation Mode
# 2. Mode B: Forecast Mode
# 3. About / Instructions
# ============================================================

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
    page_title="Weekly Vulnerability Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------
# Global settings
# -----------------------------------
DEFAULT_DATA_PATH = "../Datasets/Extracted_Data/ycrit_weekly_w_mon_copy.csv"
AVAILABLE_HORIZONS = [1, 4, 12, 26, 52]
AVAILABLE_MODELS = ["Seasonal Naive", "ETS", "SARIMA", "Prophet"]

# -----------------------------------
# Main title
# -----------------------------------
st.title("Interactive Weekly Vulnerability Forecast Dashboard")

st.write(
    "This dashboard supports model evaluation on holdout data, "
    "future forecasting, and user guidance for custom dataset upload."
)

# -----------------------------------
# Homepage mode selection
# -----------------------------------
st.markdown("## Select an Option")

mode = st.radio(
    "Choose dashboard option",
    [
        "Mode A: Evaluation Mode",
        "Mode B: Forecast Mode",
        "About / Instructions"
    ]
)

# ============================================================
# ABOUT / INSTRUCTIONS PAGE
# ============================================================
if mode == "About / Instructions":
    st.subheader("About This Dashboard")

    st.write(
        "This application allows users to explore weekly forecasting models "
        "for critical vulnerability disclosure counts. The dashboard supports "
        "both model evaluation on historical holdout data and forward forecasting "
        "into future weeks."
    )

    st.subheader("Dashboard Modes")

    st.markdown(
        """
        **Mode A: Evaluation Mode**  
        Use this mode to compare model performance on a holdout portion of the dataset.  
        The final \(h\) weeks of the series are treated as unseen test data, and the selected 
        models are trained on the earlier history before forecasting the holdout window.  
        Because the actual holdout values are known, this mode reports:
        - MAE
        - RMSE
        - Actual vs forecast plots

        **Mode B: Forecast Mode**  
        Use this mode to generate future forecasts from the full available dataset.  
        The selected models are fit on all observed data and then used to forecast ahead 
        for the chosen horizon(s).  
        Because future true values are unknown, this mode does **not** report MAE or RMSE.

        **About / Instructions**  
        This page explains how the dashboard works, how the two modes differ, 
        and how uploaded CSV files should be formatted.
        """
    )

    st.subheader("Supported Models")
    st.markdown(
        """
        The dashboard currently supports the following models:
        - **Seasonal Naive**
        - **ETS (Exponential Smoothing)**
        - **SARIMA**  
          SARIMA search is intentionally constrained to reduce memory usage.
        - **Prophet**
        """
    )

    st.subheader("Supported Forecast Horizons")
    st.markdown(
        """
        Users can select one or more of the following weekly horizons:
        - **1 week**
        - **4 weeks**
        - **12 weeks**
        - **26 weeks**
        - **52 weeks**
        """
    )

    st.subheader("Default Dataset")
    st.write(
        "By default, the dashboard uses the dissertation's cleaned weekly Ycrit dataset."
    )

    st.subheader("Uploaded CSV Requirements")
    st.markdown(
        """
        If uploading your own dataset, the CSV must contain the following columns:

        - `week` : weekly datetime values
        - `ycrit` : numeric weekly target values

        Example format:

        ```csv
        week,ycrit
        2019-01-07,12
        2019-01-14,70
        2019-01-21,84
        ```

        Additional requirements:
        - `week` must be parseable as dates
        - `ycrit` must be numeric
        - duplicate weeks are not allowed
        - the dataset should contain enough weekly observations for seasonal forecasting
        """
    )

    st.subheader("How to Use the Dashboard")
    st.markdown(
        """
        1. Select **Mode A** or **Mode B**
        2. Choose whether to use the default dataset or upload your own CSV
        3. Select one or more forecasting models
        4. Select one or more forecast horizons
        5. Click **Run**
        6. Review the plots and tables produced by the dashboard
        """
    )

# ============================================================
# MODE A OR MODE B UI
# ============================================================
else:
    if mode == "Mode A: Evaluation Mode":
        st.info(
            "Mode A evaluates selected models on a holdout portion of the dataset. "
            "The last h weeks are treated as unseen test data, allowing the dashboard "
            "to compute MAE and RMSE because the true observed values are known."
        )
    else:
        st.info(
            "Mode B fits selected models on the full available dataset and generates "
            "future forecasts for the selected horizons. MAE and RMSE are not shown "
            "because future observed values are not yet available."
        )

    st.sidebar.title("Forecast Controls")

    data_source = st.sidebar.radio(
        "Choose data source",
        ["Use default dissertation dataset", "Upload my own CSV"]
    )

    uploaded_file = None
    if data_source == "Upload my own CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    selected_models = st.sidebar.multiselect(
        "Select one or more models",
        AVAILABLE_MODELS,
        default=["ETS", "SARIMA"]
    )

    selected_horizons = st.sidebar.multiselect(
        "Select one or more forecast horizons (weeks)",
        AVAILABLE_HORIZONS,
        default=[4, 12]
    )

    run_action = st.sidebar.button("Run")

    if run_action:
        try:
            df_app = load_and_validate_data(DEFAULT_DATA_PATH, uploaded_file)

            st.success("Dataset loaded and validated successfully.")

            st.subheader("Input Data Preview")
            st.dataframe(df_app.tail(10))

            st.subheader("Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Observations", len(df_app))
            col2.metric("Start Date", str(df_app.index.min().date()))
            col3.metric("End Date", str(df_app.index.max().date()))

            # Historical plot
            st.subheader("Historical Weekly Series")
            fig_hist = plot_historical_series(df_app)
            st.plotly_chart(fig_hist, width="stretch")

            if len(selected_models) == 0:
                st.warning("Please select at least one model.")
                st.stop()

            if len(selected_horizons) == 0:
                st.warning("Please select at least one horizon.")
                st.stop()

            if mode == "Mode A: Evaluation Mode":
                metric_rows = []
                progress_text = st.empty()

                for horizon in selected_horizons:
                    st.markdown(f"## Holdout Evaluation: {horizon}-Week Horizon")

                    for model_name in selected_models:
                        progress_text.write(f"Evaluating {model_name} at {horizon}-week horizon...")

                        actual_index, actual_values, pred_values, intervals, mae, r = evaluate_on_holdout(
                            df_app["ycrit"],
                            model_name,
                            horizon
                        )

                        metric_rows.append({
                            "Model": model_name,
                            "Horizon_Weeks": horizon,
                            "MAE": mae,
                            "RMSE": r
                        })

                        c1, c2 = st.columns(2)
                        c1.metric(f"{model_name} MAE ({horizon}w)", f"{mae:.3f}")
                        c2.metric(f"{model_name} RMSE ({horizon}w)", f"{r:.3f}")

                        fig_eval = plot_holdout_forecast(
                            actual_index,
                            actual_values,
                            pred_values,
                            intervals,
                            model_name,
                            horizon
                        )
                        st.plotly_chart(fig_eval, width="stretch")

                progress_text.write("All selected evaluations completed.")

                metrics_df = pd.DataFrame(metric_rows).sort_values(["Horizon_Weeks", "MAE"])

                st.subheader("Model Comparison Table")
                st.dataframe(metrics_df)

                csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Evaluation Metrics CSV",
                    data=csv_bytes,
                    file_name="evaluation_metrics.csv",
                    mime="text/csv"
                )

            else:
                progress_text = st.empty()

                for horizon in selected_horizons:
                    st.markdown(f"## Future Forecast: {horizon}-Week Horizon")

                    future_index = make_future_index(df_app.index.max(), horizon)
                    forecast_df = pd.DataFrame(index=future_index)

                    for model_name in selected_models:
                        progress_text.write(f"Forecasting with {model_name} at {horizon}-week horizon...")

                        pred_values, intervals = fit_and_forecast(
                            df_app["ycrit"],
                            model_name,
                            horizon
                        )

                        forecast_df[model_name] = pred_values

                        fig_forecast = plot_future_forecast(
                            df_app["ycrit"].tail(26).index,
                            df_app["ycrit"].tail(26).values,
                            future_index,
                            pred_values,
                            intervals,
                            model_name,
                            horizon
                        )
                        st.plotly_chart(fig_forecast, width="stretch")

                    progress_text.write("All selected future forecasts completed.")

                    st.subheader(f"Forecast Table ({horizon}-Week Horizon)")
                    st.dataframe(forecast_df)

                    csv_bytes = forecast_df.to_csv().encode("utf-8")
                    st.download_button(
                        label=f"Download Forecast CSV ({horizon} weeks)",
                        data=csv_bytes,
                        file_name=f"forecast_{horizon}_weeks.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error: {e}")