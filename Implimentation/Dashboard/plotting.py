# # ============================================================
# # plotting.py
# # Plotting utilities for the Streamlit dashboard
# # ============================================================

# import matplotlib.pyplot as plt


# def plot_historical_series(df_app):
#     """
#     Plot the full historical weekly ycrit series.
#     """
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(df_app.index, df_app["ycrit"], color="black", linewidth=1.8)
#     ax.set_title("Historical Weekly Ycrit Series")
#     ax.set_xlabel("Week")
#     ax.set_ylabel("Ycrit")
#     ax.grid(True, alpha=0.3)
#     return fig

# def plot_holdout_forecast(actual_index, actual_values, pred_values, intervals, model_name, horizon):
#     """
#     Plot actual holdout values against predicted holdout values.
#     """
#     fig, ax = plt.subplots(figsize=(12, 5))
#     ax.plot(actual_index, actual_values, label="Actual", color="black", linewidth=2)
#     ax.plot(actual_index, pred_values, label=f"{model_name} Forecast", linewidth=2)

#     # Plot confidence intervals if available
#     if intervals is not None:
#         ax.fill_between(
#             actual_index,
#             intervals[:, 0],
#             intervals[:, 1],
#             alpha=0.2,
#             label="95% Interval"
#         )

#     ax.set_title(f"{model_name} Holdout Evaluation ({horizon}-Week Horizon)")
#     ax.set_xlabel("Week")
#     ax.set_ylabel("Ycrit")
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     return fig

# def plot_future_forecast(history_index, history_values, future_index, pred_values, intervals, model_name, horizon):
#     """
#     Plot recent history together with future forecast.
#     """
#     fig, ax = plt.subplots(figsize=(12, 5))

#     # Show recent history for context
#     ax.plot(history_index, history_values, label="Recent History", color="black", linewidth=2)
#     ax.plot(future_index, pred_values, label=f"{model_name} Forecast", linewidth=2)

#     # Plot forecast intervals if available
#     if intervals is not None:
#         ax.fill_between(
#             future_index,
#             intervals[:, 0],
#             intervals[:, 1],
#             alpha=0.2,
#             label="95% Interval"
#         )

#     ax.set_title(f"{model_name} Future Forecast ({horizon}-Week Horizon)")
#     ax.set_xlabel("Week")
#     ax.set_ylabel("Ycrit")
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     return fig




# ==============================================================


# ============================================================
# plotting.py
# Interactive plotting utilities for the Streamlit dashboard
# using Plotly
# ============================================================

# Import Plotly graph objects for interactive plotting
import plotly.graph_objects as go


def plot_historical_series(df_app):
    """
    Create an interactive Plotly chart of the historical weekly ycrit series.
    """

    # Create a new figure
    fig = go.Figure()

    # Add the historical time series line
    fig.add_trace(
        go.Scatter(
            x=df_app.index,
            y=df_app["ycrit"],
            mode="lines",
            name="Historical Ycrit",
            line=dict(color="black", width=2)
        )
    )

    # Update layout settings
    fig.update_layout(
        title="Historical Weekly Ycrit Series",
        xaxis_title="Week",
        yaxis_title="Ycrit",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def plot_holdout_forecast(actual_index, actual_values, pred_values, intervals, model_name, horizon):
    """
    Create an interactive Plotly chart for holdout evaluation.
    Shows:
    - actual values
    - forecast values
    - optional confidence interval
    """

    fig = go.Figure()

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=actual_index,
            y=actual_values,
            mode="lines",
            name="Actual",
            line=dict(color="black", width=2)
        )
    )

    # Add forecast values
    fig.add_trace(
        go.Scatter(
            x=actual_index,
            y=pred_values,
            mode="lines",
            name=f"{model_name} Forecast",
            line=dict(width=2)
        )
    )

    # Add confidence interval if available
    if intervals is not None:
        lower = intervals[:, 0]
        upper = intervals[:, 1]

        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=actual_index,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # Lower bound with fill to upper
        fig.add_trace(
            go.Scatter(
                x=actual_index,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 100, 255, 0.2)",
                name="95% Interval",
                hoverinfo="skip"
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{model_name} Holdout Evaluation ({horizon}-Week Horizon)",
        xaxis_title="Week",
        yaxis_title="Ycrit",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def plot_future_forecast(history_index, history_values, future_index, pred_values, intervals, model_name, horizon):
    """
    Create an interactive Plotly chart for future forecast mode.
    Shows:
    - recent history
    - future forecast
    - optional confidence interval
    """

    fig = go.Figure()

    # Add recent historical values
    fig.add_trace(
        go.Scatter(
            x=history_index,
            y=history_values,
            mode="lines",
            name="Recent History",
            line=dict(color="black", width=2)
        )
    )

    # Add forecast values
    fig.add_trace(
        go.Scatter(
            x=future_index,
            y=pred_values,
            mode="lines",
            name=f"{model_name} Forecast",
            line=dict(width=2)
        )
    )

    # Add confidence interval if available
    if intervals is not None:
        lower = intervals[:, 0]
        upper = intervals[:, 1]

        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=future_index,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # Lower bound with shaded fill
        fig.add_trace(
            go.Scatter(
                x=future_index,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 100, 255, 0.2)",
                name="95% Interval",
                hoverinfo="skip"
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{model_name} Future Forecast ({horizon}-Week Horizon)",
        xaxis_title="Week",
        yaxis_title="Ycrit",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig