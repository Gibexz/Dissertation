# ============================================================
# forecasting.py
# Forecasting functions for the Streamlit dashboard
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet

# Seasonal cycle for weekly annual data
SEASONALITY = 52


def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def prophet_fit(y_hist: pd.Series):
    """
    Fit Prophet on historical weekly data.
    """
    dfp = y_hist.reset_index()
    dfp.columns = ["ds", "y"]
    dfp["ds"] = pd.to_datetime(dfp["ds"]).dt.tz_localize(None)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(dfp)
    return model


def prophet_forecast(model, h):
    """
    Forecast h weeks ahead with Prophet.
    Returns forecast and confidence interval.
    """
    future = model.make_future_dataframe(periods=h, freq="W-MON")
    future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)
    fcst = model.predict(future).tail(h)
    return fcst["yhat"].values, fcst[["yhat_lower", "yhat_upper"]].values


def fit_and_forecast(y: pd.Series, model_name: str, horizon: int):
    """
    Fit selected model on full history and forecast horizon weeks ahead.

    Used in Forecast Mode.
    """

    if model_name == "Seasonal Naive":
        last_season = y.iloc[-SEASONALITY:].values
        reps = int(np.ceil(horizon / len(last_season)))
        forecast = np.tile(last_season, reps)[:horizon]
        return forecast, None

    elif model_name == "ETS":
        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONALITY
        ).fit(optimized=True)

        forecast = model.forecast(horizon).values
        return forecast, None

    elif model_name == "SARIMA":
        # Constrained search to reduce memory use
        model = auto_arima(
            y,
            seasonal=True,
            m=SEASONALITY,
            stepwise=True,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            max_d=1,
            max_D=1
        )

        forecast, conf_int = model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=0.05
        )
        return forecast, conf_int

    elif model_name == "Prophet":
        model = prophet_fit(y)
        forecast, conf_int = prophet_forecast(model, horizon)
        return forecast, conf_int

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def evaluate_on_holdout(series: pd.Series, model_name: str, horizon: int):
    """
    Evaluate a model using the last h observations as holdout.

    Used in Evaluation Mode.

    Returns:
    - forecast index
    - actual values
    - predicted values
    - intervals
    - MAE
    - RMSE
    """

    if len(series) <= horizon + SEASONALITY:
        raise ValueError(
            f"Not enough data to evaluate horizon {horizon}. "
            f"Need more than {horizon + SEASONALITY} observations."
        )

    # Holdout split
    y_train = series.iloc[:-horizon]
    y_test = series.iloc[-horizon:]

    # Forecast the holdout window
    y_pred, intervals = fit_and_forecast(y_train, model_name, horizon)

    # Compute accuracy metrics
    mae = mean_absolute_error(y_test.values, y_pred)
    r = rmse(y_test.values, y_pred)

    return y_test.index, y_test.values, y_pred, intervals, mae, r


def make_future_index(last_date, horizon):
    """
    Create weekly future dates beginning one week after last_date.
    """
    return pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )