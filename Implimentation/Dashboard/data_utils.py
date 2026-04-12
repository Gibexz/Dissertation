import pandas as pd


def load_and_validate_data(default_data_path, file_obj=None):
    """
    Load either the default dataset or a user-uploaded CSV file.

    Expected columns:
    - week
    - ycrit

    Returns:
    - cleaned DataFrame indexed by week
    """

    if file_obj is None:
        df = pd.read_csv(default_data_path)
    else:
        df = pd.read_csv(file_obj)

    required_cols = {"week", "ycrit"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'week' and 'ycrit'.")

    df["week"] = pd.to_datetime(df["week"], errors="coerce", utc=True).dt.tz_localize(None)
    if df["week"].isna().any():
        raise ValueError("Some values in 'week' could not be parsed as valid datetimes.")

    df["ycrit"] = pd.to_numeric(df["ycrit"], errors="coerce")
    if df["ycrit"].isna().any():
        raise ValueError("Column 'ycrit' must contain only numeric values.")

    df = df.sort_values("week").drop_duplicates(subset=["week"])
    df = df.set_index("week")

    # Explicitly set weekly Monday frequency
    df = df.asfreq("W-MON")

    # Check for missing values introduced by frequency alignment
    if df["ycrit"].isna().any():
        raise ValueError(
            "Dataset has missing weekly periods after applying W-MON frequency. "
            "Please ensure the data are complete and weekly."
        )

    if len(df) < 60:
        raise ValueError("Dataset is too short. Please provide at least 60 weekly observations.")

    return df