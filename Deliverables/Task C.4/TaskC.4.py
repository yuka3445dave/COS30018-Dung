# File: stock_prediction_taskC4.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
#
# Student extensions:
# - Task C.2: reusable data loading + preprocessing
# - Task C.3: candlestick + moving-window boxplot + custom n-day input
# - Task C.4: dynamic DL builder + multiple DL experiments + combined result graph

import os
import re
import time
import random
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import mplfinance as mpf
import yfinance as yf

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN


# ------------------------------------------------------------------------------
# Reproducibility helper
# ------------------------------------------------------------------------------

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ------------------------------------------------------------------------------
# Helper functions (Task C.2)
# ------------------------------------------------------------------------------

def _safe_filename(s):
    """
    Convert a string into a filesystem-safe filename.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _flatten_yfinance_columns(df):
    """
    Flatten yfinance MultiIndex columns if present.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


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


def load_and_process_multi_feature_dataset(
    company: str,
    dataset_start: str,
    dataset_end: str,
    feature_columns,
    target_column: str = "Close",
    prediction_days=60,
    split_method="date",   # "ratio" | "date" | "random"
    train_ratio=0.8,
    split_date=None,
    random_seed=42,
    nan_strategy="ffill_bfill",  # "ffill_bfill" | "drop" | "interpolate" | "ffill" | "bfill"
    scale_features=True,
    feature_range=(0, 1),
    cache_dir="data_cache",
    use_cache=True,
):
    """
    Task C.2: load + process dataset with multiple features.
    """
    # 1) Load raw data
    df_raw = download_or_load_stock_data(
        company=company,
        start_date=dataset_start,
        end_date=dataset_end,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    df = df_raw.copy()
    df = _flatten_yfinance_columns(df)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in downloaded data: {missing}. Available: {df.columns.tolist()}")

    df = df[feature_columns].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2) Handle NaNs
    if nan_strategy == "drop":
        df = df.dropna()
    elif nan_strategy == "ffill":
        df = df.ffill()
    elif nan_strategy == "bfill":
        df = df.bfill()
    elif nan_strategy == "ffill_bfill":
        df = df.ffill().bfill()
    elif nan_strategy == "interpolate":
        df = df.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError("nan_strategy must be one of: drop, ffill, bfill, ffill_bfill, interpolate")

    df = df.dropna()
    df_clean = df

    print("NaN before:", df_raw.isna().sum().sum())
    print("NaN after :", df_clean.isna().sum().sum())

    if len(df_clean) <= prediction_days + 5:
        raise ValueError("Not enough data after cleaning to build sequences. Try a larger date range.")

    # 3) Decide split
    n = len(df_clean)

    if split_method not in {"ratio", "date", "random"}:
        raise ValueError("split_method must be one of: 'ratio', 'date', 'random'")

    if split_method == "date":
        if split_date is None:
            raise ValueError("split_date must be provided when split_method='date'")
        split_ts = pd.to_datetime(split_date)
        split_idx = int(np.searchsorted(df_clean.index.values, np.datetime64(split_ts), side="right"))
        split_idx = max(split_idx, prediction_days + 1)
        split_idx = min(split_idx, n - 1)
    elif split_method == "ratio":
        split_idx = int(n * train_ratio)
        split_idx = max(split_idx, prediction_days + 1)
        split_idx = min(split_idx, n - 1)
    else:
        split_idx = None

    # 4) Fit scalers
    x_scaler = MinMaxScaler(feature_range=feature_range) if scale_features else None
    y_scaler = MinMaxScaler(feature_range=feature_range) if scale_features else None

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

        # 5) Build sequences
        X_all = []
        y_all = []
        end_positions = []

        for i in range(prediction_days, n):
            X_all.append(X_scaled[i - prediction_days:i, :])
            y_all.append(y_scaled[i])
            end_positions.append(i)

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        end_positions = np.array(end_positions)

        train_mask = end_positions < split_idx
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[~train_mask], y_all[~train_mask]

        test_target_pos = end_positions[~train_mask]
        test_target_dates = df_clean.index[test_target_pos]

    else:
        rng = np.random.default_rng(random_seed)

        end_positions = np.arange(prediction_days, n)
        perm = rng.permutation(end_positions)

        train_size = int(len(perm) * train_ratio)
        train_ends = np.sort(perm[:train_size])

        train_rows = set()
        for i in train_ends:
            train_rows.update(range(i - prediction_days, i + 1))
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

        X_all = []
        y_all = []
        all_ends = np.arange(prediction_days, n)

        for i in all_ends:
            X_all.append(X_scaled[i - prediction_days:i, :])
            y_all.append(y_scaled[i])

        X_all = np.array(X_all)
        y_all = np.array(y_all)

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


# ------------------------------------------------------------------------------
# Helper functions (Task C.3)
# ------------------------------------------------------------------------------

def aggregate_ohlcv_n_days(df_ohlcv: pd.DataFrame, n_days_per_candle: int = 1) -> pd.DataFrame:
    """
    Aggregate OHLCV to make 1 candle represent n trading days.
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

    first_dates = df.groupby(grp).apply(lambda x: x.index[0])
    agg.index = pd.to_datetime(first_dates.values)

    return agg


def ask_user_n_days_per_candle(default_value: int = 5) -> int:
    """
    Ask user to enter how many trading days each candlestick should represent.
    """
    user_input = input(
        f"Enter number of trading days per candlestick (default={default_value}): "
    ).strip()

    if user_input == "":
        print(f"No input given. Using default n={default_value}")
        return default_value

    try:
        n = int(user_input)
        if n < 1:
            raise ValueError
        print(f"Using user-selected n={n}")
        return n
    except ValueError:
        print(f"Invalid input. Using default n={default_value}")
        return default_value


def plot_candlestick_chart(
    df_raw: pd.DataFrame,
    company: str,
    last_n_candles: int = 60,
    n_days_per_candle: int = 1,
    save_path: str = None
):
    """
    Display stock OHLCV data using a candlestick chart.
    """
    df = _flatten_yfinance_columns(df_raw)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df_candle = aggregate_ohlcv_n_days(df, n_days_per_candle=n_days_per_candle)
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


def plot_moving_window_boxplot(
    df_raw: pd.DataFrame,
    price_column: str = "Close",
    window_size: int = 20,
    step: int = 5,
    company: str = "",
    save_path: str = None,
    max_xticks: int = 12
):
    """
    Improved moving-window boxplot with readable x-axis labels.
    """
    df = _flatten_yfinance_columns(df_raw)
    series = df[price_column].dropna()

    windows = []
    labels = []

    for start in range(0, len(series) - window_size + 1, step):
        windows.append(series.iloc[start:start + window_size].values)
        labels.append(str(series.index[start].date()))

    if len(windows) == 0:
        raise ValueError("No windows could be created. Try smaller window_size or step.")

    plt.figure(figsize=(14, 6))
    plt.boxplot(windows, showfliers=False)

    plt.title(f"{company} Moving-Window Boxplot ({price_column}) | window={window_size}, step={step}")
    plt.xlabel("Window start date")
    plt.ylabel("Price")

    n_windows = len(windows)
    tick_step = max(1, n_windows // max_xticks)
    tick_positions = list(range(1, n_windows + 1, tick_step))
    tick_labels = [labels[i - 1] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


# ------------------------------------------------------------------------------
# Helper functions (Task C.4)
# ------------------------------------------------------------------------------

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
    """
    Task C.4 (v0.3): Build a DL model dynamically.
    """
    name = layer_name.upper()
    layer_map = {
        "LSTM": LSTM,
        "GRU": GRU,
        "RNN": SimpleRNN,
        "SIMPLERNN": SimpleRNN,
    }
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
        return_seq = (i < num_layers - 1)

        if i == 0:
            model.add(RNNLayer(units=units_list[i], return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(RNNLayer(units=units_list[i], return_sequences=return_seq))

        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def inverse_transform_target(values, target_scaler):
    arr = np.asarray(values).reshape(-1, 1)
    if target_scaler is None:
        return arr.reshape(-1)
    return target_scaler.inverse_transform(arr).reshape(-1)


def plot_prediction_vs_actual(
    actual_prices,
    predicted_prices,
    company,
    label,
    save_path=None,
    show=False
):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, label=f"Predicted {label}")
    plt.title(f"{company} Prediction vs Actual | {label}")
    plt.xlabel("Time (test index)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_dl_predictions(
    actual_prices,
    prediction_map,
    company,
    save_path=None,
    show=True
):
    """
    Combined line graph: actual + all DL experiment prediction lines.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color="black", linewidth=2, label=f"Actual {company} Price")

    for label, preds in prediction_map.items():
        plt.plot(preds, label=label)

    plt.title(f"{company} DL Model Comparison (All Prediction Lines)")
    plt.xlabel("Time (test index)")
    plt.ylabel("Price")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def run_dl_experiment(
    experiment_name,
    layer_name,
    input_shape,
    x_train,
    y_train,
    x_test,
    actual_prices,
    target_scaler,
    output_dir,
    num_layers=3,
    units=50,
    dropout=0.2,
    epochs=25,
    batch_size=64,
):
    """
    Train one DL configuration, save its prediction plot, and return metrics.
    """
    tf.keras.backend.clear_session()

    model = build_recurrent_model(
        layer_name=layer_name,
        input_shape=input_shape,
        num_layers=num_layers,
        units=units,
        dropout=dropout,
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
    )

    print("\n" + "=" * 80)
    print("Running experiment:", experiment_name)
    print("=" * 80)
    model.summary()

    t0 = time.perf_counter()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    train_time = time.perf_counter() - t0

    predicted_scaled = model.predict(x_test, verbose=0)
    predicted_prices = inverse_transform_target(predicted_scaled, target_scaler)

    mae, mse, rmse = regression_metrics(actual_prices, predicted_prices)

    indiv_plot_path = os.path.join(output_dir, f"{experiment_name}_pred_vs_actual.png")
    plot_prediction_vs_actual(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        company=COMPANY,
        label=experiment_name,
        save_path=indiv_plot_path,
        show=False,
    )

    return {
        "experiment_name": experiment_name,
        "model_type": layer_name,
        "num_layers": num_layers,
        "units": str(units),
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_time_sec": train_time,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "predicted_prices": predicted_prices,
        "model": model,
    }


# ------------------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------------------

set_global_seed(42)

COMPANY = "CBA.AX"

DATASET_START = "2020-01-01"
DATASET_END = "2024-07-02"
print(f"Dataset range: {DATASET_START} → {DATASET_END}")

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COLUMN = "Close"

PREDICTION_DAYS = 60

SPLIT_METHOD = "date"
SPLIT_DATE = "2023-08-01"
TRAIN_RATIO = 0.8
print("Split method used:", SPLIT_METHOD)

NAN_STRATEGY = "ffill_bfill"
CACHE_DIR = "data_cache"
USE_CACHE = True
SCALE_FEATURES = True

# Output folders
BASE_OUTPUT_DIR = "outputs_c4"
VIS_DIR = os.path.join(BASE_OUTPUT_DIR, "visualizations")
EXP_DIR = os.path.join(BASE_OUTPUT_DIR, "dl_experiments")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

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

# Load + process dataset
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

print("Raw df shape:", df_raw.shape)
print("Clean df shape:", df_clean.shape)
print("Train/Test samples:", x_train.shape[0], x_test.shape[0])
print("X_train shape (samples, lookback, n_features):", x_train.shape)
print("Scaler keys:", list(scalers.keys()))
print("Feature columns:", scalers["feature_columns"])
print("Target column:", scalers["target_column"])

# ------------------------------------------------------------------------------
# Task C.3 Visualizations
# ------------------------------------------------------------------------------

plot_candlestick_chart(
    df_raw,
    company=COMPANY,
    last_n_candles=60,
    n_days_per_candle=1,
    save_path=os.path.join(VIS_DIR, "candlestick_n1.png")
)

plot_candlestick_chart(
    df_raw,
    company=COMPANY,
    last_n_candles=60,
    n_days_per_candle=5,
    save_path=os.path.join(VIS_DIR, "candlestick_n5.png")
)

user_n = ask_user_n_days_per_candle(default_value=5)
custom_candle_path = os.path.join(VIS_DIR, f"candlestick_n{user_n}_custom.png")

plot_candlestick_chart(
    df_raw,
    company=COMPANY,
    last_n_candles=60,
    n_days_per_candle=user_n,
    save_path=custom_candle_path
)

print(f"Saved custom candlestick chart: {custom_candle_path}")

plot_moving_window_boxplot(
    df_raw,
    price_column="Close",
    window_size=20,
    step=5,
    company=COMPANY,
    save_path=os.path.join(VIS_DIR, "moving_window_boxplot.png"),
    max_xticks=12
)

# ------------------------------------------------------------------------------
# Task C.4 Experiments
# ------------------------------------------------------------------------------

target_scaler = scalers["target_scaler"]
actual_prices = inverse_transform_target(y_test, target_scaler)

n_features = x_train.shape[2]
input_shape = (PREDICTION_DAYS, n_features)

EXPERIMENT_CONFIGS = [
    {"model_type": "LSTM", "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2, "epochs": 25, "batch_size": 64},
    {"model_type": "GRU",  "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2, "epochs": 25, "batch_size": 64},
    {"model_type": "RNN",  "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2, "epochs": 25, "batch_size": 64},
    {"model_type": "LSTM", "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2, "epochs": 32, "batch_size": 50},
    {"model_type": "GRU",  "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2, "epochs": 32, "batch_size": 50},
    {"model_type": "RNN",  "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2, "epochs": 32, "batch_size": 50},
]

results = []
prediction_map = {}
best_result = None

for cfg in EXPERIMENT_CONFIGS:
    experiment_name = f"{cfg['model_type']}_L{cfg['num_layers']}_E{cfg['epochs']}_B{cfg['batch_size']}"

    result = run_dl_experiment(
        experiment_name=experiment_name,
        layer_name=cfg["model_type"],
        input_shape=input_shape,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        actual_prices=actual_prices,
        target_scaler=target_scaler,
        output_dir=EXP_DIR,
        num_layers=cfg["num_layers"],
        units=cfg["units"],
        dropout=cfg["dropout"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
    )

    results.append(result)
    prediction_map[experiment_name] = result["predicted_prices"]

    if (best_result is None) or (result["mae"] < best_result["mae"]):
        best_result = result

# Save results table
results_df = pd.DataFrame([
    {
        "experiment_name": r["experiment_name"],
        "model_type": r["model_type"],
        "num_layers": r["num_layers"],
        "units": r["units"],
        "dropout": r["dropout"],
        "epochs": r["epochs"],
        "batch_size": r["batch_size"],
        "train_time_sec": r["train_time_sec"],
        "mae": r["mae"],
        "mse": r["mse"],
        "rmse": r["rmse"],
    }
    for r in results
])

results_df = results_df.sort_values("mae").reset_index(drop=True)
results_csv_path = os.path.join(EXP_DIR, "dl_experiment_results.csv")
results_df.to_csv(results_csv_path, index=False)
print("\nSaved results table:", results_csv_path)
print(results_df)

# Combined line graph: all DL outputs
combined_plot_path = os.path.join(EXP_DIR, "all_dl_predictions_comparison.png")
plot_all_dl_predictions(
    actual_prices=actual_prices,
    prediction_map=prediction_map,
    company=COMPANY,
    save_path=combined_plot_path,
    show=True,
)
print("Saved combined DL comparison graph:", combined_plot_path)

# ------------------------------------------------------------------------------
# Best model summary + next-day prediction
# ------------------------------------------------------------------------------

print("\nBest experiment:")
print("Experiment name:", best_result["experiment_name"])
print("Model type:", best_result["model_type"])
print("MAE:", best_result["mae"])
print("MSE:", best_result["mse"])
print("RMSE:", best_result["rmse"])
print("Train time (sec):", best_result["train_time_sec"])

best_model = best_result["model"]

last_window = df_clean[FEATURE_COLUMNS].values[-PREDICTION_DAYS:]

if SCALE_FEATURES:
    x_scaler = scalers["feature_scaler"]
    last_window_scaled = x_scaler.transform(last_window)
else:
    last_window_scaled = last_window

real_data = np.array([last_window_scaled])

next_scaled = best_model.predict(real_data, verbose=0)
next_price = inverse_transform_target(next_scaled, target_scaler)
print(f"Next-day prediction for {COMPANY} using best DL model: {next_price}")