from typing import Optional, Iterable, Sequence
import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame, datetime_col: Optional[str] = None) -> pd.DataFrame:
    """Ensure the DataFrame has a DatetimeIndex.  If `datetime_col` is given it will be converted to datetime and set as the
    index. Otherwise the function will try to convert the existing index to datetimes. Returns a copy of the input DataFrame. """
    df = df.copy()
    if datetime_col is not None:
        if datetime_col not in df.columns:
            raise ValueError(f"datetime_col '{datetime_col}' not found in DataFrame")
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("DataFrame index is not datetime-like and no valid datetime_col was provided")
    return df


def add_returns(df: pd.DataFrame, column: str, periods: int = 1, new_col: Optional[str] = None) -> pd.DataFrame:
    """Add percent returns (simple returns) for `column` using DataFrame.pct_change.Returns a new DataFrame with the added column named `new_col` if provided or
    `{column}_ret` by default. """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    new_col = new_col or f"{column}_ret"
    out = df.copy()
    out[new_col] = out[column].pct_change(periods=periods)
    return out


def add_log_returns(df: pd.DataFrame, column: str, periods: int = 1, new_col: Optional[str] = None) -> pd.DataFrame:
    """Compute log-returns for `column`.The result is log(x_t) - log(x_{t-periods}). Non-positive values will produce NaN (log undefined)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    new_col = new_col or f"{column}_logret"
    out = df.copy()
    # Use np.log on positive values only
    with np.errstate(divide='ignore', invalid='ignore'):
        out[new_col] = np.log(out[column]).diff(periods=periods)
    return out


def add_rolling_stats(df: pd.DataFrame, column: str, windows: Sequence[int] = (5, 10, 20), stats: Sequence[str] = ("mean", "std")) -> pd.DataFrame:
    """Add rolling statistics for `column`.For each window in `windows` this will add columns named `{column}_roll_{stat}_{window}` for each stat in `stats`.
    Supported stats: 'mean', 'std', 'min', 'max', 'median'."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    out = df.copy()
    for w in windows:
        roll = out[column].rolling(window=w, min_periods=1)
        for stat in stats:
            colname = f"{column}_roll_{stat}_{w}"
            if stat == "mean":
                out[colname] = roll.mean()
            elif stat == "std":
                out[colname] = roll.std()
            elif stat == "min":
                out[colname] = roll.min()
            elif stat == "max":
                out[colname] = roll.max()
            elif stat == "median":
                out[colname] = roll.median()
            else:
                raise ValueError(f"Unsupported stat '{stat}'")
    return out


def add_volatility(df: pd.DataFrame, column: str, window: int = 20, returns_col: Optional[str] = None, new_col: Optional[str] = None) -> pd.DataFrame:
    """Add rolling volatility (standard deviation of returns). If `returns_col` is given, compute volatility on that series. Otherwise the
    function computes simple returns internally and uses those."""
    out = df.copy()
    if returns_col is None:
        rtn = out[column].pct_change()
    else:
        if returns_col not in out.columns:
            raise ValueError(f"returns_col '{returns_col}' not found")
        rtn = out[returns_col]
    new_col = new_col or f"{column}_vol_{window}"
    out[new_col] = rtn.rolling(window=window, min_periods=1).std()
    return out


def add_lags(df: pd.DataFrame, column: str, lags: Iterable[int]) -> pd.DataFrame:
    """Add lagged copies of `column`. For each k in `lags` a new column `{column}_lag_{k}` is created (shifted by k)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    out = df.copy()
    for k in lags:
        out[f"{column}_lag_{k}"] = out[column].shift(k)
    return out


def add_spread(df: pd.DataFrame, col1: str, col2: str, new_col: Optional[str] = None, ratio: bool = False) -> pd.DataFrame:
    """Add either the difference or ratio between two columns. By default this adds `{col1}_minus_{col2}` (col1 - col2). If `ratio=True`
    it computes `{col1}_over_{col2}` (col1 / col2)."""
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("One of the specified columns not found in DataFrame")
    out = df.copy()
    if ratio:
        new_col = new_col or f"{col1}_over_{col2}"
        with np.errstate(divide='ignore', invalid='ignore'):
            out[new_col] = out[col1] / out[col2]
    else:
        new_col = new_col or f"{col1}_minus_{col2}"
        out[new_col] = out[col1] - out[col2]
    return out


def seasonal_difference(df: pd.DataFrame, column: str, period: int = 7, new_col: Optional[str] = None) -> pd.DataFrame:
    """Compute seasonal difference: x_t - x_{t-period}.Handy to remove a known seasonal pattern (for example period=7 for weekly seasonality)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    out = df.copy()
    new_col = new_col or f"{column}_sdiff_{period}"
    out[new_col] = out[column] - out[column].shift(period)
    return out


def add_monthly_diff(df: pd.DataFrame, column: str, months: int = 1, new_col: Optional[str] = None) -> pd.DataFrame:
    """Compute the difference between a value and the value `months` earlier.This is calendar-aligned: for each timestamp t we look up the value at
    t - DateOffset(months=months) and subtract. Missing exact matches (weekends,holidays) result in NaN. The function expects a DatetimeIndex; use `_ensure_datetime_index` if
    needed. """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    out = _ensure_datetime_index(df)
    new_col = new_col or f"{column}_mdiff_{months}m"
    shifted_index = out.index - pd.DateOffset(months=months)

    # Build a mapping timestamp -> last observed value (works with duplicate timestamps)
    ts_to_val = out[column].groupby(out.index).last()
    prev_vals = ts_to_val.reindex(shifted_index).values
    out[new_col] = out[column].values - prev_vals
    return out


def add_weekly_diff(df: pd.DataFrame, column: str, weeks: int = 1, new_col: Optional[str] = None) -> pd.DataFrame:
    """Compute calendar-aligned weekly difference. Looks up the value on the same weekday `weeks` earlier and subtracts. Missing
    matches produce NaN. Expects a DatetimeIndex."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    out = _ensure_datetime_index(df)
    new_col = new_col or f"{column}_wdiff_{weeks}w"
    shifted_index = out.index - pd.DateOffset(weeks=weeks)

    ts_to_val = out[column].groupby(out.index).last()
    prev_vals = ts_to_val.reindex(shifted_index).values
    out[new_col] = out[column].values - prev_vals
    return out


def generate_features(
    df: pd.DataFrame,column: str,datetime_col: Optional[str] = None,windows: Sequence[int] = (5, 10, 20),lags: Sequence[int] = (1, 2, 3),
    seasonal_period: Optional[int] = None,monthly_diff: bool = False,monthly_diff_months: int = 1,) -> pd.DataFrame:
    """High-level wrapper that creates a compact set of time-series features. Ensures the DataFrame uses a DatetimeIndex (when `datetime_col` is given),
    then adds returns, log-returns, rolling mean/std, volatility, lagged returns and optional seasonal/monthly diffs."""
    out = _ensure_datetime_index(df, datetime_col=datetime_col)
    out = add_returns(out, column, periods=1, new_col=f"{column}_ret")
    out = add_log_returns(out, column, periods=1, new_col=f"{column}_logret")
    out = add_rolling_stats(out, column, windows=windows, stats=("mean", "std"))
    out = add_volatility(out, column, window=max(windows), returns_col=f"{column}_ret", new_col=f"{column}_vol_{max(windows)}")
    out = add_lags(out, column=f"{column}_ret", lags=lags)
    # add first-difference
    out[f"{column}_diff_1"] = out[column].diff(1)
    # when a seasonal_period is provided, add calendar-aligned quarterly/yearly diffs
    if seasonal_period is not None:
        # quarterly (3 months)
        out = add_monthly_diff(out, column, months=3, new_col=f"{column}_mdiff_3m")
        # yearly (12 months)
        out = add_monthly_diff(out, column, months=12, new_col=f"{column}_mdiff_12m")
    if monthly_diff:
        out = add_monthly_diff(out, column, months=monthly_diff_months, new_col=f"{column}_mdiff_{monthly_diff_months}m")
    return out


__all__ = [
    "add_returns",
    "add_log_returns",
    "add_rolling_stats",
    "add_volatility",
    "add_lags",
    "add_spread",
    "seasonal_difference",
    "generate_features",
    "add_monthly_diff",
    "add_weekly_diff",
]
