from pathlib import Path
import pandas as pd
from features import generate_features, add_weekly_diff, add_monthly_diff
import numpy as np


def main():
    repo_root = Path(__file__).parent
    csv_path = repo_root / "dataset_cleaned.csv"
    if not csv_path.exists():
        print(f"Dataset not found at {csv_path}. Please run from repo root or place the CSV there.")
        return
    df = pd.read_csv(csv_path)
    # Heuristic: try to find a datetime-like column
    datetime_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            datetime_col = c
            break
    # Heuristic: try to find a price/series column if there's more than one column
    price_col = None
    if 'price' in df.columns:
        price_col = 'price'
    else:
        # take first numeric column that's not datetime
        for c in df.columns:
            if c == datetime_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                price_col = c
                break
    if price_col is None:
        print("No numeric series column found to generate features from.")
        return


    # generate both a seasonal diff (example period) and a monthly diff column
    out = generate_features(
        df,
        column=price_col,
        datetime_col=datetime_col,
        windows=(5, 10, 20),
        lags=(1, 2, 3),
        seasonal_period=12,
        monthly_diff=True,
        monthly_diff_months=1,
    )

    # Force presence of the standard derived columns in `out` (compute if missing)
    base = price_col
    # list of expected derived columns with short descriptions
    expected = [
        (f"{base}_ret", "Simple percent change (x_t / x_{t-1} - 1)"),
        (f"{base}_logret", "Log return: diff of log(x) (useful for additive returns)"),
        (f"{base}_roll_mean_5", "5-period rolling mean of the series"),
        (f"{base}_roll_std_5", "5-period rolling standard deviation"),
        (f"{base}_roll_mean_10", "10-period rolling mean"),
        (f"{base}_roll_std_10", "10-period rolling std"),
        (f"{base}_roll_mean_20", "20-period rolling mean"),
        (f"{base}_roll_std_20", "20-period rolling std"),
        (f"{base}_vol_20", "Volatility: std of returns over a 20-period window"),
        (f"{base}_ret_lag_1", "Return lagged by 1 period"),
        (f"{base}_ret_lag_2", "Return lagged by 2 periods"),
        (f"{base}_ret_lag_3", "Return lagged by 3 periods"),
        (f"{base}_diff_1", "First difference: x_t - x_{t-1}"),
        (f"{base}_wdiff_1w", "Calendar-aligned weekly difference (x_t - x_{t-1 week})"),
        (f"{base}_mdiff_3m", "Calendar-aligned 3-month difference (quarterly)"),
        (f"{base}_mdiff_12m", "Calendar-aligned 12-month difference (yearly)"),
        (f"{base}_mdiff_1m", "Calendar-aligned 1-month difference (month-over-month)"),
    ]

    # compute simple ones inline if they are missing
    if f"{base}_ret" not in out.columns:
        out[f"{base}_ret"] = out[base].pct_change()
    if f"{base}_logret" not in out.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f"{base}_logret"] = np.log(out[base]).diff()
    # rolling stats
    for w in (5, 10, 20):
        mean_col = f"{base}_roll_mean_{w}"
        std_col = f"{base}_roll_std_{w}"
        if mean_col not in out.columns:
            out[mean_col] = out[base].rolling(window=w, min_periods=1).mean()
        if std_col not in out.columns:
            out[std_col] = out[base].rolling(window=w, min_periods=1).std()
    # volatility (std of returns)
    if f"{base}_vol_20" not in out.columns:
        rtn = out.get(f"{base}_ret")
        if rtn is None:
            rtn = out[base].pct_change()
        out[f"{base}_vol_20"] = rtn.rolling(window=20, min_periods=1).std()
    # lags of returns
    for k in (1, 2, 3):
        lag_col = f"{base}_ret_lag_{k}"
        if lag_col not in out.columns:
            out[lag_col] = out[f"{base}_ret"].shift(k)
    # first diff
    if f"{base}_diff_1" not in out.columns:
        out[f"{base}_diff_1"] = out[base].diff(1)
    # weekly and monthly calendar-aligned diffs using helpers (they expect a datetime index)
    if f"{base}_wdiff_1w" not in out.columns:
        out = add_weekly_diff(out, base, weeks=1, new_col=f"{base}_wdiff_1w")
    if f"{base}_mdiff_1m" not in out.columns:
        out = add_monthly_diff(out, base, months=1, new_col=f"{base}_mdiff_1m")
    if f"{base}_mdiff_3m" not in out.columns:
        out = add_monthly_diff(out, base, months=3, new_col=f"{base}_mdiff_3m")
    if f"{base}_mdiff_12m" not in out.columns:
        out = add_monthly_diff(out, base, months=12, new_col=f"{base}_mdiff_12m")

    # Print a pretty table of the first 10 rows using pandas (better readability)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
        print(out.head(10).to_string(index=False))

    # Save the full generated DataFrame to CSV
    out_all_csv = repo_root / 'features_all.csv'
    out.to_csv(out_all_csv, index=False)
    print(f"Saved full output to: {out_all_csv}")

   
    print('\nPreview (first 10 rows) of the generated data:')
    first10 = out.head(10)
    out_preview_csv = repo_root / 'features_preview.csv'
    first10.to_csv(out_preview_csv, index=False)
    
    df_preview = pd.read_csv(out_all_csv)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
        print(df_preview.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
