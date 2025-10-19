# FEATURES — concise reference

This repository provides a small set of time-series feature helpers in
`features.py` and a minimal demo `feature_demo.py` that reads
`dataset_cleaned.csv`, generates features and writes `features_all.csv`.

Essentials
- Main convenience: `generate_features(df, column, datetime_col=None)`
- Key generated columns for `{col}`:
  - `{col}_ret`, `{col}_logret`
  - rolling means/std: windows 5,10,20 -> `{col}_roll_mean_W`, `{col}_roll_std_W`
  - `{col}_vol_20` (rolling std of returns)
  - `{col}_ret_lag_1/2/3`, `{col}_diff_1`
  - calendar diffs (only when DataFrame has a DatetimeIndex):
    `{col}_wdiff_1w`, `{col}_mdiff_1m`, `{col}_mdiff_3m`, `{col}_mdiff_12m`

# Notes on features (short)

This file describes what `features.py` does and what the columns mean. Keep it
handy as a quick reference.

Main function
- `generate_features(df, column, datetime_col=None)` — convenience wrapper
  that produces commonly used time-series features for `column`.

What it adds
- `{col}_ret`, `{col}_logret`
- Rolling stats: `{col}_roll_mean_{W}`, `{col}_roll_std_{W}` for W in (5,10,20)
- `{col}_vol_20` (rolling std of returns)
- `{col}_ret_lag_1/2/3`, `{col}_diff_1`
- Calendar-aligned diffs when a `DatetimeIndex` is present: `{col}_wdiff_1w`,
  `{col}_mdiff_1m`, `{col}_mdiff_3m`, `{col}_mdiff_12m`

Column meanings (quick)
- `{col}_ret`: (x_t / x_{t-1}) - 1
- `{col}_logret`: log(x_t) - log(x_{t-1})
- Rolling mean/std: mean / std over the last W observations
- `{col}_vol_20`: std of `{col}_ret` over a 20-sample window
- `lag_k`: value shifted by k rows (past value)
- `diff_1`: x_t - x_{t-1}
- Calendar diffs: difference between value and the value one calendar
  period earlier (week/month/quarter/year); requires a `DatetimeIndex`

Quick use
```powershell
python feature_demo.py
```

Notes
- Calendar diffs need the DataFrame to have a `DatetimeIndex`. If the CSV
  contains a date column, set `datetime_col` when calling `generate_features`
  or set the DataFrame index before calling.
- To use a specific base column instead of the automatic selection in the demo,
  edit `feature_demo.py` or add a small CLI wrapper.
- DatetimeIndex: a pandas Index of dtype datetime64[ns]; required for calendar
  diffs to align by calendar periods rather than fixed row offsets.
