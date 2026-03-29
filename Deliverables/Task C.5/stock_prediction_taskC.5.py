# File: stock_prediction_task2.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
#

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import mplfinance as mpf
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN

import yfinance as yf


# ------------------------------------------------------------------------------
# Helper functions (Task C.2)
# ------------------------------------------------------------------------------

def _safe_filename(s):
    """
    Convert a string into a filesystem-safe filename.
    Example: "CBA.AX_2020-01-01_2024-07-02.csv" is already safe,
    but this also protects against unexpected characters.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _flatten_yfinance_columns(df):
    """
    yfinance sometimes returns MultiIndex columns like:
        ('Close','CBA.AX'), ('Open','CBA.AX'), ...
    But our code (and many plotting libs) expect flat columns:
        'Close', 'Open', 'High', 'Low', 'Volume'

    If df.columns is MultiIndex, we keep only the first level ('Close', 'Open', ...).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df

def aggregate_ohlcv_n_days(df_ohlcv: pd.DataFrame, n_days_per_candle: int = 1) -> pd.DataFrame:
    """
    Aggregate OHLCV to make 1 candle represent n trading days (n >= 1).

    Rules for aggregation:
    - Open  = first Open in the group
    - High  = max High in the group
    - Low   = min Low in the group
    - Close = last Close in the group
    - Volume = sum Volume in the group
    """
    if n_days_per_candle <= 1:
        return df_ohlcv.copy()

    df = df_ohlcv.copy()
    grp = np.arange(len(df)) // n_days_per_candle

    agg = df.groupby(grp).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    # Use the first date of each block as the candle timestamp
    first_dates = df.groupby(grp).apply(lambda x: x.index[0])
    agg.index = pd.to_datetime(first_dates.values)

    return agg

def plot_candlestick_chart(df_raw: pd.DataFrame,
                           company: str,
                           last_n_candles: int = 60,
                           n_days_per_candle: int = 1,
                           save_path: str = None):
    """
    Display stock OHLCV data using a candlestick chart.

    Parameters:
    - df_raw: DataFrame containing Open/High/Low/Close/Volume with DatetimeIndex.
    - company: ticker symbol used in the plot title.
    - last_n_candles: how many candles to show at the end of the series.
    - n_days_per_candle: each candle represents n trading days (n >= 1). (C.3 requirement)
    - save_path: if provided, saves the plot to a PNG file for report screenshots.
    """
    df = _flatten_yfinance_columns(df_raw)

    # Keep only required columns for candlestick
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Aggregate to n-day candles
    df_candle = aggregate_ohlcv_n_days(df, n_days_per_candle=n_days_per_candle)

    # Take last N candles for readability
    df_candle = df_candle.tail(last_n_candles)

    title = f"{company} Candlestick (n={n_days_per_candle} trading day(s)/candle)"

    mpf.plot(
        df_candle,
        type="candle",
        volume=True,
        title=title,
        style="yahoo",
        savefig=save_path
    )


def download_or_load_stock_data(
    company,
    start_date,
    end_date,
    cache_dir="data_cache",
    use_cache=True,
    force_download=False,
):
    """
    Download OHLCV stock data from yfinance OR load from local CSV cache.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_name = _safe_filename(f"{company}_{start_date}_{end_date}.csv")
    cache_path = os.path.join(cache_dir, cache_name)

    if use_cache and (not force_download) and os.path.exists(cache_path):
        print("Loaded from cache:", cache_path)  
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(company, start=start_date, end=end_date, progress=False)
        df = _flatten_yfinance_columns(df)
        if use_cache:
            df.to_csv(cache_path)
            print("Saved to cache:", cache_path)  

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df

def plot_moving_window_boxplot(df_raw: pd.DataFrame,
                               price_column: str = "Close",
                               window_size: int = 20,
                               step: int = 5,
                               company: str = "",
                               save_path: str = None):
    """
    Display stock data using a boxplot chart over moving windows.

    Parameters:
    - price_column: which column to use (usually 'Close').
    - window_size: number of consecutive trading days per window.
    - step: how many days to move forward between windows (controls overlap).
    """
    df = _flatten_yfinance_columns(df_raw)
    series = df[price_column].dropna()

    windows = []
    labels = []

    for start in range(0, len(series) - window_size + 1, step):
        windows.append(series.iloc[start:start + window_size].values)
        labels.append(str(series.index[start].date()))

    plt.figure(figsize=(12, 5))
    plt.boxplot(windows, showfliers=False)
    plt.title(f"{company} Moving-Window Boxplot ({price_column}) | window={window_size}, step={step}")
    plt.xlabel("Window index")
    plt.ylabel("Price")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()
    
def load_and_process_multi_feature_dataset(
    company: str,
    dataset_start: str,
    dataset_end: str,
    feature_columns,
    target_column: str = "Close",
    prediction_days=60,
    # Task C.5 additions:
    forecast_horizon: int = 1,   # predict t + forecast_horizon
    k_steps: int = 1,            # if >1 => multistep output of length k
    # (c) Split methods:
    split_method="date",   # "ratio" | "date" | "random"
    train_ratio=0.8,
    split_date=None,
    random_seed=42,
    # (b) NaN handling:
    nan_strategy="ffill_bfill",  # "ffill_bfill" | "drop" | "interpolate" | "ffill" | "bfill"
    # (e) Scaling options:
    scale_features=True,
    feature_range=(0, 1),
    # (d) Local caching:
    cache_dir="data_cache",
    use_cache=True,
):
    """
    Task C.2: A single function to load + process dataset with multiple features.

    Requirement mapping:
    (a) dataset_start, dataset_end define the whole dataset range.
    (b) nan_strategy controls how NaN values are handled.
    (c) split_method supports ratio/date/random splits.
    (d) use_cache/cache_dir enables local save/load.
    (e) scale_features stores scalers in a dict for future access.

    Returns:
      X_train, y_train, X_test, y_test: ready for LSTM (X is 3D: samples x lookback x n_features)
      scalers: dict containing 'feature_scaler' and 'target_scaler' for inverse_transform later
      df_raw: original OHLCV dataframe
      df_clean: cleaned dataframe used for training/testing
      test_target_dates: dates corresponding to y_test (useful for plotting)
    """
    # -----------------------------
    # 1) Load raw data (download or cache)
    # -----------------------------
    df_raw = download_or_load_stock_data(
        company=company,
        start_date=dataset_start,
        end_date=dataset_end,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    # Keep only the needed feature columns
    df = df_raw.copy()
    df = _flatten_yfinance_columns(df)

    # --- Task C.5: ensure "Adj Close" exists if requested ---
    if "Adj Close" in feature_columns and "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            print("Warning: 'Adj Close' not found in data; using 'Close' as proxy for 'Adj Close'.")
        else:
            raise ValueError("Adj Close requested but neither 'Adj Close' nor 'Close' exists in downloaded data.")

    # Some tickers/data sources may not include all columns.
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in downloaded data: {missing}. Available: {df.columns.tolist()}")

    df = df[feature_columns].copy()

    # Replace +/-inf with NaN so it can handle all missing/invalid values consistently
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -----------------------------
    # 2) Handle NaNs (Requirement b)
    # -----------------------------
    if nan_strategy == "drop":
        df = df.dropna()
    elif nan_strategy == "ffill":
        df = df.ffill()
    elif nan_strategy == "bfill":
        df = df.bfill()
    elif nan_strategy == "ffill_bfill":
        # Common time-series baseline:
        # - forward fill uses last known value
        # - back fill fixes NaNs at the start of the series
        df = df.ffill().bfill()
    elif nan_strategy == "interpolate":
        # Time interpolation can be helpful for small gaps; we still ffill/bfill as a safety net
        df = df.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError("nan_strategy must be one of: drop, ffill, bfill, ffill_bfill, interpolate")

    # Final safety: remove any remaining NaN rows
    df = df.dropna()
    df_clean = df
    print("NaN before:", df_raw.isna().sum().sum())
    print("NaN after :", df_clean.isna().sum().sum())
    if len(df_clean) <= prediction_days + 5:
        raise ValueError("Not enough data after cleaning to build sequences. Try a larger date range.")

    # -----------------------------
    # 3) Decide split (Requirement c)
    # -----------------------------
    n = len(df_clean)

    if split_method not in {"ratio", "date", "random"}:
        raise ValueError("split_method must be one of: 'ratio', 'date', 'random'")

    if split_method == "date":
        if split_date is None:
            raise ValueError("split_date must be provided when split_method='date'")
        split_ts = pd.to_datetime(split_date)
        # Find the boundary index (row position) where test starts
        split_idx = int(np.searchsorted(df_clean.index.values, np.datetime64(split_ts), side="right"))
        # Ensure boundary is within range and allows lookback
        split_idx = max(split_idx, prediction_days + 1)
        split_idx = min(split_idx, n - 1)
    elif split_method == "ratio":
        split_idx = int(n * train_ratio)
        split_idx = max(split_idx, prediction_days + 1)
        split_idx = min(split_idx, n - 1)
    else:
        # For random split, we don't use split_idx; we split sequences later.
        split_idx = None

    # -----------------------------
    # 4) Fit scalers (Requirement e)
    #    IMPORTANT: fit on TRAIN data only (avoid leakage)
    # -----------------------------
    x_scaler = MinMaxScaler(feature_range=feature_range) if scale_features else None
    y_scaler = MinMaxScaler(feature_range=feature_range) if scale_features else None

    # For ratio/date split, "train rows" are all rows before split_idx.
    # For random split, we will compute train rows after selecting train sequences.
    if split_method in {"ratio", "date"}:
        train_df = df_clean.iloc[:split_idx]

        if scale_features:
            x_scaler.fit(train_df[feature_columns].values)
            y_scaler.fit(train_df[[target_column]].values)

            X_scaled = x_scaler.transform(df_clean[feature_columns].values)
            y_scaled = y_scaler.transform(df_clean[[target_column]].values).reshape(-1)
        else:
            X_scaled = df_clean[feature_columns].values
            y_scaled = df_clean[target_column].values.astype(float)

        # 5) Build sequences (supports single-step OR multistep)
        # -----------------------------
        X_all, y_all, end_positions = [], [], []

        # window ends at i-1; first predicted day is target_start = i + forecast_horizon - 1
        # last predicted day index = target_end = target_start + (k_steps - 1)
        max_i = n - (forecast_horizon + k_steps - 1)

        for i in range(prediction_days, max_i):
            X_all.append(X_scaled[i - prediction_days:i, :])

            target_start = i + forecast_horizon - 1
            target_end = target_start + (k_steps - 1)

            if k_steps == 1:
                y_all.append(y_scaled[target_start])  # scalar
            else:
                y_all.append(y_scaled[target_start:target_end + 1])  # length-k vector

            # use the LAST target day for split decisions (avoids leakage)
            end_positions.append(target_end)

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        end_positions = np.array(end_positions)

        # Split sequences:
        train_mask = end_positions < split_idx
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[~train_mask], y_all[~train_mask]

        test_target_pos = end_positions[~train_mask]
        test_target_dates = df_clean.index[test_target_pos]

    else:
        # -----------------------------
        # Random split (Requirement c)
        # NOTE: Random split is not ideal for forecasting because it can mix future/past.
        # We still implement it because Task C.2 asks for the option.
        # -----------------------------
        rng = np.random.default_rng(random_seed)

        end_positions = np.arange(prediction_days, n)
        perm = rng.permutation(end_positions)

        train_size = int(len(perm) * train_ratio)
        train_ends = np.sort(perm[:train_size])
        test_ends = np.sort(perm[train_size:])

        # Fit scalers ONLY on rows used by TRAIN sequences (avoid using test rows)
        train_rows = set()
        for i in train_ends:
            # sequence uses [i-prediction_days, ..., i] (include target row)
            if split_method == "random" and (forecast_horizon != 1 or k_steps != 1):
                raise ValueError("Random split is only supported for single-step (forecast_horizon=1, k_steps=1) in this version.")
            target_start = i + forecast_horizon - 1
            target_end = target_start + (k_steps - 1)
            train_rows.update(range(i - prediction_days, target_end + 1))
        train_rows = sorted([r for r in train_rows if 0 <= r < n])

        train_df = df_clean.iloc[train_rows]

        if scale_features:
            x_scaler.fit(train_df[feature_columns].values)
            y_scaler.fit(train_df[[target_column]].values)

            X_scaled = x_scaler.transform(df_clean[feature_columns].values)
            y_scaled = y_scaler.transform(df_clean[[target_column]].values).reshape(-1)
        else:
            X_scaled = df_clean[feature_columns].values
            y_scaled = df_clean[target_column].values.astype(float)

        # Build all sequences once
        X_all = []
        y_all = []
        # last predicted day index for each sample (avoid going out of bounds)
        all_ends = np.arange(
            prediction_days + forecast_horizon - 1 + (k_steps - 1),
            n
        )

        # Convert end index -> window end i (where window is [i-prediction_days : i])
        # end = (i + forecast_horizon - 1 + (k_steps - 1))
        # so i = end - (forecast_horizon - 1) - (k_steps - 1)
        def end_to_i(end_idx: int) -> int:
            return end_idx - (forecast_horizon - 1) - (k_steps - 1)

        X_all, y_all = [], []
        for end_idx in all_ends:
            i = end_to_i(end_idx)

            X_all.append(X_scaled[i - prediction_days:i, :])

            target_start = i + forecast_horizon - 1
            target_end = target_start + (k_steps - 1)

            if k_steps == 1:
                y_all.append(y_scaled[target_start])
            else:
                y_all.append(y_scaled[target_start:target_end + 1])

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        # Select train/test sequences by end position
        train_mask = np.isin(all_ends, train_ends)
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[~train_mask], y_all[~train_mask]

        test_target_dates = df_clean.index[all_ends[~train_mask]]

    scalers = {
        "feature_scaler": x_scaler,
        "target_scaler": y_scaler,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }

    return X_train, y_train, X_test, y_test, scalers, df_raw, df_clean, test_target_dates


from typing import List, Union, Optional

# -----------------------------
# Task C.5 wrappers (v0.4)
# -----------------------------
def prepare_multistep_univariate(**kwargs):
    # Task 5.1: multistep, single feature (Close only)
    return load_and_process_multi_feature_dataset(
        feature_columns=["Close"],
        target_column="Close",
        **kwargs
    )

def prepare_multivariate_singlestep(**kwargs):
    # Task 5.2: multivariate, single-step
    return load_and_process_multi_feature_dataset(
        feature_columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        target_column="Close",
        k_steps=1,
        **kwargs
    )

def prepare_multivariate_multistep(k_steps: int, **kwargs):
    # Task 5.3: multivariate + multistep
    return load_and_process_multi_feature_dataset(
        feature_columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        target_column="Close",
        k_steps=k_steps,
        **kwargs
    )

def build_recurrent_model(
    layer_name: str,
    input_shape: tuple,
    num_layers: int = 3,
    units: Union[int, List[int]] = 50,
    dropout: float = 0.2,
    dense_units: int = 1,
    optimizer: str = "adam",
    loss: str = "mean_squared_error",
):
    name = layer_name.upper()
    layer_map = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN, "SIMPLERNN": SimpleRNN}
    if name not in layer_map:
        raise ValueError("layer_name must be one of: LSTM, GRU, RNN (SimpleRNN)")

    RNNLayer = layer_map[name]

    if isinstance(units, int):
        units_list = [units] * num_layers
    else:
        units_list = list(units)
        if len(units_list) != num_layers:
            raise ValueError("If units is a list, its length must equal num_layers.")

    model = Sequential(name=f"{name}_{num_layers}layers")

    for i in range(num_layers):
        return_seq = i < (num_layers - 1)
        if i == 0:
            model.add(RNNLayer(units=units_list[i], return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(RNNLayer(units=units_list[i], return_sequences=return_seq))

        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model

# ------------------------------------------------------------------------------
# Main program 
# ------------------------------------------------------------------------------

COMPANY = "CBA.AX"

# (a) Whole dataset range (no more manual train/test downloads)
DATASET_START = "2020-01-01"
DATASET_END = "2024-07-02"
print(f"Dataset range: {DATASET_START} → {DATASET_END}")
# Multi-feature inputs (Open/High/Low/Close/Volume). You can add 'Adj Close' if available.
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COLUMN = "Close"

# Number of days to look back
PREDICTION_DAYS = 60

# -----------------------------
# Task C.5 config (v0.4)
# -----------------------------
TASK5_MODE = "multivariate_multistep"
FORECAST_HORIZON = 1
K_STEPS = 5

# (c) Split method options:
# - "date": split at a date boundary (recommended for time series)
# - "ratio": split by a train_ratio boundary
# - "random": split sequences randomly (provided for Task C.2 requirement, but not ideal for forecasting)
SPLIT_METHOD = "date"
SPLIT_DATE = "2023-08-01"  # boundary date (similar to TRAIN_END in v0.1)
TRAIN_RATIO = 0.8
print("Split method used:", SPLIT_METHOD)

# (b) NaN handling
NAN_STRATEGY = "ffill_bfill"

# (d) caching
CACHE_DIR = "data_cache"
USE_CACHE = True

# (e) scaling
SCALE_FEATURES = True

# --- Quick split-method evidence (NO TRAINING) ---
for method in ["date", "ratio", "random"]:
    Xtr, ytr, Xte, yte, *_ = load_and_process_multi_feature_dataset(
        company=COMPANY,
        dataset_start=DATASET_START,
        dataset_end=DATASET_END,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        prediction_days=PREDICTION_DAYS,

        split_method=method,
        split_date=SPLIT_DATE,       
        train_ratio=TRAIN_RATIO,     
        random_seed=42,             

        nan_strategy=NAN_STRATEGY,
        scale_features=SCALE_FEATURES,  

        use_cache=USE_CACHE,
        cache_dir=CACHE_DIR
    )
    print(f"[Split={method}] X_train={Xtr.shape}, X_test={Xte.shape}")

# Load + process dataset (Task C.2 function)
x_train, y_train, x_test, y_test, scalers, df_raw, df_clean, test_dates = load_and_process_multi_feature_dataset(
    company=COMPANY,
    dataset_start=DATASET_START,
    dataset_end=DATASET_END,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN,
    prediction_days=PREDICTION_DAYS,
    split_method=SPLIT_METHOD,
    train_ratio=TRAIN_RATIO,
    split_date=SPLIT_DATE,
    nan_strategy=NAN_STRATEGY,
    scale_features=SCALE_FEATURES,
    cache_dir=CACHE_DIR,
    use_cache=USE_CACHE,
)

if TASK5_MODE == "multistep":
    x_train, y_train, x_test, y_test, scalers, df_raw, df_clean, test_dates = prepare_multistep_univariate(
        company=COMPANY,
        dataset_start=DATASET_START,
        dataset_end=DATASET_END,
        prediction_days=PREDICTION_DAYS,
        split_method=SPLIT_METHOD,
        train_ratio=TRAIN_RATIO,
        split_date=SPLIT_DATE,
        random_seed=42,
        nan_strategy=NAN_STRATEGY,
        scale_features=SCALE_FEATURES,
        cache_dir=CACHE_DIR,
        use_cache=USE_CACHE,
        forecast_horizon=FORECAST_HORIZON,
        k_steps=K_STEPS,
    )

elif TASK5_MODE == "multivariate":
    x_train, y_train, x_test, y_test, scalers, df_raw, df_clean, test_dates = prepare_multivariate_singlestep(
        company=COMPANY,
        dataset_start=DATASET_START,
        dataset_end=DATASET_END,
        prediction_days=PREDICTION_DAYS,
        split_method=SPLIT_METHOD,
        train_ratio=TRAIN_RATIO,
        split_date=SPLIT_DATE,
        random_seed=42,
        nan_strategy=NAN_STRATEGY,
        scale_features=SCALE_FEATURES,
        cache_dir=CACHE_DIR,
        use_cache=USE_CACHE,
        forecast_horizon=FORECAST_HORIZON,  # allows “specified day in the future”
    )

else:  # multivariate_multistep
    x_train, y_train, x_test, y_test, scalers, df_raw, df_clean, test_dates = prepare_multivariate_multistep(
        k_steps=K_STEPS,
        company=COMPANY,
        dataset_start=DATASET_START,
        dataset_end=DATASET_END,
        prediction_days=PREDICTION_DAYS,
        split_method=SPLIT_METHOD,
        train_ratio=TRAIN_RATIO,
        split_date=SPLIT_DATE,
        random_seed=42,
        nan_strategy=NAN_STRATEGY,
        scale_features=SCALE_FEATURES,
        cache_dir=CACHE_DIR,
        use_cache=USE_CACHE,
        forecast_horizon=FORECAST_HORIZON,
    )

print("Task5 mode:", TASK5_MODE)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print('Raw df shape:', df_raw.shape)
print('Clean df shape:', df_clean.shape)
print('Train/Test samples:', x_train.shape[0], x_test.shape[0])
print('X_train shape (samples, lookback, n_features):', x_train.shape)
print('Scaler keys:', list(scalers.keys()))
print('Feature columns:', scalers['feature_columns'])
print('Target column:', scalers['target_column'])


# ------------------------------------------------------------------------------
# Build the Model (Task C.4 v0.3)
# ------------------------------------------------------------------------------

n_features = x_train.shape[2]
input_shape = (PREDICTION_DAYS, n_features)

MODEL_TYPE = "GRU"          # try: "LSTM", "GRU", "RNN"
NUM_LAYERS = 3
UNITS = [50, 50, 50]         # or just 50
DROPOUT = 0.2





# n_features = x_train.shape[2]
# input_shape = (PREDICTION_DAYS, n_features)



DENSE_UNITS = K_STEPS if (TASK5_MODE in ["multistep", "multivariate_multistep"] and K_STEPS > 1) else 1

model = build_recurrent_model(
    layer_name=MODEL_TYPE,
    input_shape=input_shape,
    num_layers=NUM_LAYERS,
    units=UNITS,
    dropout=DROPOUT,
    dense_units=DENSE_UNITS,
    optimizer="adam",
    loss="mean_squared_error",
)

print("Model type:", MODEL_TYPE)
model.summary()

# -----------------------------
# Task C.3 Visualizations (v0.2)
# -----------------------------
plot_candlestick_chart(df_raw, company=COMPANY, last_n_candles=60, n_days_per_candle=1,
                       save_path="candlestick_n1.png")

plot_candlestick_chart(df_raw, company=COMPANY, last_n_candles=60, n_days_per_candle=5,
                       save_path="candlestick_n5.png")  #  C.3 requirement

plot_moving_window_boxplot(df_raw, price_column="Close", window_size=20, step=5, company=COMPANY,
                           save_path="moving_window_boxplot.png")

# -----------------------------
# Training (timed) - Task C.4 experiments
# -----------------------------
EPOCHS = 25
BATCH_SIZE = 64

t0 = time.perf_counter()
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
train_time = time.perf_counter() - t0
print("Train time (sec):", train_time)

# -----------------------------
# Test evaluation (MAE/MSE/RMSE)
# -----------------------------
predicted_scaled = model.predict(x_test)
target_scaler = scalers["target_scaler"]

if DENSE_UNITS == 1:
    predicted_prices = target_scaler.inverse_transform(predicted_scaled).reshape(-1)
    actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
else:
    pred_flat = predicted_scaled.reshape(-1, 1)
    true_flat = y_test.reshape(-1, 1)
    predicted_prices = target_scaler.inverse_transform(pred_flat).reshape(-1, DENSE_UNITS)
    actual_prices = target_scaler.inverse_transform(true_flat).reshape(-1, DENSE_UNITS)

mae = mean_absolute_error(actual_prices.reshape(-1), predicted_prices.reshape(-1))
mse = mean_squared_error(actual_prices.reshape(-1), predicted_prices.reshape(-1))
rmse = np.sqrt(mse)
print("Test MAE:", mae)
print("Test MSE:", mse)
print("Test RMSE:", rmse)


# Get the actual prices for the same test targets (aligned with test_dates)
# NOTE: y_test contains scaled targets; we rebuild actual values from df_clean.
# The first test target date corresponds to row position >= PREDICTION_DAYS.
if DENSE_UNITS == 1:
    actual_prices = df_clean.loc[test_dates, TARGET_COLUMN].values.reshape(-1)
# Plot predictions vs actual
if DENSE_UNITS == 1:
    actual_plot = actual_prices
    pred_plot = predicted_prices
    pred_label = f"Predicted {COMPANY} Price"
else:
    # Show only the first step (t+1) to keep the plot readable
    actual_plot = actual_prices[:, 0]
    pred_plot = predicted_prices[:, 0]
    pred_label = f"Predicted {COMPANY} Price (t+1 of {DENSE_UNITS})"

plt.plot(actual_plot, color="black", label=f"Actual {COMPANY} Price")
plt.plot(pred_plot, color="green", label=pred_label)
plt.title(f"{COMPANY} Share Price (Multi-feature {MODEL_TYPE})")
plt.xlabel("Time (test index)")
plt.ylabel("Price")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# Predict next step(s) using last lookback window
# ------------------------------------------------------------------------------

# IMPORTANT: use the same feature columns that the scaler/model was trained on
FEATURES_FOR_MODEL = scalers["feature_columns"]  # this includes 'Adj Close' in multivariate modes
last_window = df_clean[FEATURES_FOR_MODEL].values[-PREDICTION_DAYS:]  # shape: (lookback, n_features)

if SCALE_FEATURES:
    x_scaler = scalers["feature_scaler"]
    last_window_scaled = x_scaler.transform(last_window)
else:
    last_window_scaled = last_window

real_data = np.array([last_window_scaled])  # shape: (1, lookback, n_features)

next_scaled = model.predict(real_data)

# Inverse-transform next prediction(s)
if DENSE_UNITS == 1:
    next_price = target_scaler.inverse_transform(next_scaled)[0, 0]
    print(f"Next-day prediction for {COMPANY}: {next_price}")
else:
    # target_scaler was fit on 1 column, so reshape to (-1,1) first
    next_prices = target_scaler.inverse_transform(next_scaled.reshape(-1, 1)).reshape(-1)
    print(f"Next {DENSE_UNITS}-day predictions for {COMPANY}: {next_prices}")