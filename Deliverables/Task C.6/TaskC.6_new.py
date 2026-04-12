# File: stock_prediction_taskC6.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
#
# Student extensions:
# - Task C.2: reusable data loading + preprocessing
# - Task C.3: candlestick + moving-window boxplot + custom n-day input
# - Task C.4: dynamic DL builder
# - Task C.5: multivariate / multistep preparation logic
# - Task C.6: ensemble forecasting with SARIMA + DL + visual experiment comparison

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
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    forecast_horizon: int = 1,
    k_steps: int = 1,
    split_method="date",
    train_ratio=0.8,
    split_date=None,
    random_seed=42,
    nan_strategy="ffill_bfill",
    scale_features=True,
    feature_range=(0, 1),
    cache_dir="data_cache",
    use_cache=True,
):
    """
    General data loader for single-step / multistep forecasting.
    """
    df_raw = download_or_load_stock_data(
        company=company,
        start_date=dataset_start,
        end_date=dataset_end,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    df = df_raw.copy()
    df = _flatten_yfinance_columns(df)

    if "Adj Close" in feature_columns and "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            print("Warning: 'Adj Close' not found; using 'Close' as proxy for 'Adj Close'.")
        else:
            raise ValueError("Adj Close requested but neither 'Adj Close' nor 'Close' exists in downloaded data.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in downloaded data: {missing}. Available: {df.columns.tolist()}")

    df = df[feature_columns].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

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

        X_all, y_all, end_positions = [], [], []

        max_i = n - (forecast_horizon + k_steps - 1)

        for i in range(prediction_days, max_i):
            X_all.append(X_scaled[i - prediction_days:i, :])

            target_start = i + forecast_horizon - 1
            target_end = target_start + (k_steps - 1)

            if k_steps == 1:
                y_all.append(y_scaled[target_start])
            else:
                y_all.append(y_scaled[target_start:target_end + 1])

            end_positions.append(target_end)

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        end_positions = np.array(end_positions)

        train_mask = end_positions < split_idx
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[~train_mask], y_all[~train_mask]

        test_target_pos = end_positions[~train_mask]
        test_target_dates = df_clean.index[test_target_pos]

    else:
        raise ValueError("Random split is not used in this C.6 workflow. Use date or ratio split.")

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
# Helper functions (Task C.4/C.5 style)
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
    Dynamic recurrent model builder.
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
    arr = np.asarray(values)
    if target_scaler is None:
        return arr.copy()

    original_shape = arr.shape
    flat = arr.reshape(-1, 1)
    inv = target_scaler.inverse_transform(flat)
    return inv.reshape(original_shape)


# ------------------------------------------------------------------------------
# Helper functions (Task C.6 ensemble)
# ------------------------------------------------------------------------------

def fit_sarima(train_series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 5)):
    """
    Fit SARIMA on a 1D training series and return the fitted model.
    """
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    return fitted


def forecast_sarima(fitted_model, steps):
    """
    Forecast next 'steps' points.
    """
    return fitted_model.forecast(steps=steps)


def ensemble_weighted(pred_dl, pred_sarima, w_dl=0.5):
    """
    Weighted ensemble:
      y_hat = w_dl * DL + (1 - w_dl) * SARIMA
    """
    pred_dl = np.asarray(pred_dl).reshape(-1)
    pred_sarima = np.asarray(pred_sarima).reshape(-1)
    return w_dl * pred_dl + (1 - w_dl) * pred_sarima


def plot_ensemble_vs_actual(
    actual_prices,
    dl_pred,
    sarima_pred,
    ens_pred,
    company,
    label,
    save_path=None,
    show=False
):
    """
    Plot actual, DL, SARIMA, ensemble on one graph.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color="black", linewidth=2, label=f"Actual {company} Price")
    plt.plot(dl_pred, label="DL prediction")
    plt.plot(sarima_pred, label="SARIMA prediction")
    plt.plot(ens_pred, label="Ensemble prediction")
    plt.title(f"{company} Ensemble Prediction vs Actual | {label}")
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


def plot_mae_bar_for_best(best_row, save_path=None, show=True):
    """
    Bar chart for DL_MAE vs SARIMA_MAE vs ENS_MAE for best config.
    """
    labels = ["DL_MAE", "SARIMA_MAE", "ENS_MAE"]
    values = [best_row["DL_MAE"], best_row["SARIMA_MAE"], best_row["ENS_MAE"]]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Best Configuration MAE Comparison")
    plt.ylabel("MAE")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_weight_sensitivity(results_df, save_path=None, show=True):
    """
    Plot ensemble weight sensitivity: w_dl vs ENS_MAE
    Uses only GRU rows if available, otherwise all rows sorted by w_dl.
    """
    df = results_df.copy()

    if "DL" in df.columns and (df["DL"] == "GRU").any():
        df = df[df["DL"] == "GRU"].copy()

    df = df.sort_values("w_dl")

    plt.figure(figsize=(8, 5))
    plt.plot(df["w_dl"], df["ENS_MAE"], marker="o")
    plt.title("Ensemble Weight Sensitivity")
    plt.xlabel("w_dl")
    plt.ylabel("ENS_MAE")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_ensemble_predictions(
    actual_prices,
    prediction_map,
    company,
    save_path=None,
    show=True
):
    """
    Combined graph: actual + several ensemble predictions.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color="black", linewidth=2, label=f"Actual {company} Price")

    for label, preds in prediction_map.items():
        plt.plot(preds, label=label)

    plt.title(f"{company} Ensemble Comparison (All Selected Configurations)")
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


def run_task6_experiment(
    config,
    x_train,
    y_train,
    x_test,
    y_true,
    train_close,
    target_scaler,
    input_shape,
    output_dir
):
    """
    Run one ensemble experiment:
    - train DL
    - fit SARIMA
    - forecast both
    - ensemble them
    - compute metrics
    - save plot
    """
    tf.keras.backend.clear_session()

    dl_model = build_recurrent_model(
        layer_name=config["dl_type"],
        input_shape=input_shape,
        num_layers=config["num_layers"],
        units=config["units"],
        dropout=config["dropout"],
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
    )

    exp_name = (
        f"{config['dl_type']}_E{config['epochs']}_B{config['batch']}"
        f"_ARIMA{config['order']}_S{config['seasonal']}_W{config['w']}"
    )

    print("\n" + "=" * 100)
    print("Running experiment:", exp_name)
    print("=" * 100)

    t0 = time.perf_counter()
    dl_model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch"], verbose=0)
    dl_train_time = time.perf_counter() - t0

    dl_pred_scaled = dl_model.predict(x_test, verbose=0)
    dl_pred = inverse_transform_target(dl_pred_scaled, target_scaler).reshape(-1)

    dl_mae, dl_mse, dl_rmse = regression_metrics(y_true, dl_pred)

    t1 = time.perf_counter()
    sarima_fitted = fit_sarima(train_close, order=config["order"], seasonal_order=config["seasonal"])
    sarima_fit_time = time.perf_counter() - t1

    sarima_pred = forecast_sarima(sarima_fitted, steps=len(y_true))
    sarima_pred = np.asarray(sarima_pred).reshape(-1)

    sa_mae, sa_mse, sa_rmse = regression_metrics(y_true, sarima_pred)

    ens_pred = ensemble_weighted(dl_pred, sarima_pred, w_dl=config["w"])
    en_mae, en_mse, en_rmse = regression_metrics(y_true, ens_pred)

    plot_path = os.path.join(output_dir, f"{exp_name}_pred_vs_actual.png")
    plot_ensemble_vs_actual(
        actual_prices=y_true,
        dl_pred=dl_pred,
        sarima_pred=sarima_pred,
        ens_pred=ens_pred,
        company=COMPANY,
        label=exp_name,
        save_path=plot_path,
        show=False,
    )

    return {
        "DL": config["dl_type"],
        "epochs": config["epochs"],
        "batch": config["batch"],
        "SARIMA_order": config["order"],
        "SARIMA_seasonal": config["seasonal"],
        "w_dl": config["w"],
        "DL_MAE": dl_mae,
        "DL_MSE": dl_mse,
        "DL_RMSE": dl_rmse,
        "SARIMA_MAE": sa_mae,
        "SARIMA_MSE": sa_mse,
        "SARIMA_RMSE": sa_rmse,
        "ENS_MAE": en_mae,
        "ENS_MSE": en_mse,
        "ENS_RMSE": en_rmse,
        "DL_time_s": dl_train_time,
        "SARIMA_fit_s": sarima_fit_time,
        "plot_path": plot_path,
        "experiment_name": exp_name,
        "ens_pred": ens_pred,
    }


# ------------------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------------------

set_global_seed(42)

COMPANY = "CBA.AX"
DATASET_START = "2020-01-01"
DATASET_END = "2024-07-02"
print(f"Dataset range: {DATASET_START} → {DATASET_END}")

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
BASE_OUTPUT_DIR = "outputs_c6"
VIS_DIR = os.path.join(BASE_OUTPUT_DIR, "visualizations")
EXP_DIR = os.path.join(BASE_OUTPUT_DIR, "ensemble_experiments")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Quick split-method evidence (single-step 5-feature base evidence)
# ------------------------------------------------------------------------------



for method in ["date", "ratio"]:
    Xtr, ytr, Xte, yte, *_ = load_and_process_multi_feature_dataset(
        company=COMPANY,
        dataset_start=DATASET_START,
        dataset_end=DATASET_END,
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        target_column="Close",
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

# ------------------------------------------------------------------------------
# Main Task C.6 data load
# Use multivariate single-step input for ensemble experiments
# ------------------------------------------------------------------------------

FEATURE_COLUMNS_TASK6 = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

x_train, y_train, x_test, y_test, scalers, df_raw, df_clean, test_dates = load_and_process_multi_feature_dataset(
    company=COMPANY,
    dataset_start=DATASET_START,
    dataset_end=DATASET_END,
    feature_columns=FEATURE_COLUMNS_TASK6,
    target_column="Close",
    prediction_days=PREDICTION_DAYS,
    forecast_horizon=1,
    k_steps=1,
    split_method=SPLIT_METHOD,
    train_ratio=TRAIN_RATIO,
    split_date=SPLIT_DATE,
    random_seed=42,
    nan_strategy=NAN_STRATEGY,
    scale_features=SCALE_FEATURES,
    cache_dir=CACHE_DIR,
    use_cache=USE_CACHE,
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
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
# Prepare train/test Close series for ensemble alignment
# ------------------------------------------------------------------------------

target_scaler = scalers["target_scaler"]
y_true = inverse_transform_target(y_test, target_scaler).reshape(-1)

first_test_date = test_dates[0]
first_test_pos = df_clean.index.get_loc(first_test_date)
train_close = df_clean["Close"].iloc[:first_test_pos]

n_features = x_train.shape[2]
input_shape = (PREDICTION_DAYS, n_features)

# ------------------------------------------------------------------------------
# Task C.6 experiment configurations
# ------------------------------------------------------------------------------

TASK6_EXPERIMENTS = [
    {"dl_type": "LSTM", "epochs": 25, "batch": 64, "order": (1, 1, 1), "seasonal": (1, 0, 1, 5), "w": 0.5, "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2},
    {"dl_type": "GRU",  "epochs": 25, "batch": 64, "order": (1, 1, 1), "seasonal": (1, 0, 1, 5), "w": 0.5, "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2},
    {"dl_type": "LSTM", "epochs": 25, "batch": 64, "order": (2, 1, 2), "seasonal": (1, 0, 1, 5), "w": 0.5, "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2},
    {"dl_type": "RNN",  "epochs": 25, "batch": 64, "order": (1, 1, 1), "seasonal": (1, 0, 1, 5), "w": 0.5, "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2},
    {"dl_type": "GRU",  "epochs": 25, "batch": 64, "order": (1, 1, 1), "seasonal": (1, 0, 1, 5), "w": 0.8, "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2},
    {"dl_type": "GRU",  "epochs": 25, "batch": 64, "order": (1, 1, 1), "seasonal": (1, 0, 1, 5), "w": 0.2, "num_layers": 3, "units": [50, 50, 50], "dropout": 0.2},
]

results = []
prediction_map = {}

for cfg in TASK6_EXPERIMENTS:
    result = run_task6_experiment(
        config=cfg,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_true=y_true,
        train_close=train_close,
        target_scaler=target_scaler,
        input_shape=input_shape,
        output_dir=EXP_DIR,
    )
    results.append(result)
    prediction_map[result["experiment_name"]] = result["ens_pred"]

results_df = pd.DataFrame([
    {
        "DL": r["DL"],
        "epochs": r["epochs"],
        "batch": r["batch"],
        "SARIMA_order": r["SARIMA_order"],
        "SARIMA_seasonal": r["SARIMA_seasonal"],
        "w_dl": r["w_dl"],
        "DL_MAE": r["DL_MAE"],
        "SARIMA_MAE": r["SARIMA_MAE"],
        "ENS_MAE": r["ENS_MAE"],
        "DL_time_s": r["DL_time_s"],
        "SARIMA_fit_s": r["SARIMA_fit_s"],
        "experiment_name": r["experiment_name"],
        "plot_path": r["plot_path"],
    }
    for r in results
]).sort_values("ENS_MAE").reset_index(drop=True)

results_csv_path = os.path.join(EXP_DIR, "ensemble_experiment_results.csv")
results_df.to_csv(results_csv_path, index=False)

print("\nSaved ensemble results table:", results_csv_path)
print(results_df)
print("Best config:", results_df.iloc[0].to_dict())

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

best_row = results_df.iloc[0]

best_mae_bar_path = os.path.join(EXP_DIR, "best_ensemble_mae_bar.png")
plot_mae_bar_for_best(best_row, save_path=best_mae_bar_path, show=True)

weight_sensitivity_path = os.path.join(EXP_DIR, "ensemble_weight_sensitivity.png")
plot_weight_sensitivity(results_df, save_path=weight_sensitivity_path, show=True)

combined_plot_path = os.path.join(EXP_DIR, "all_ensemble_predictions_comparison.png")
plot_all_ensemble_predictions(
    actual_prices=y_true,
    prediction_map=prediction_map,
    company=COMPANY,
    save_path=combined_plot_path,
    show=True
)

print("Saved best MAE bar chart:", best_mae_bar_path)
print("Saved ensemble weight sensitivity chart:", weight_sensitivity_path)
print("Saved combined ensemble comparison graph:", combined_plot_path)

# ------------------------------------------------------------------------------
# Final best-model interpretation
# ------------------------------------------------------------------------------

print("\nBest experiment summary:")
print("DL:", best_row["DL"])
print("SARIMA order:", best_row["SARIMA_order"])
print("Seasonal order:", best_row["SARIMA_seasonal"])
print("w_dl:", best_row["w_dl"])
print("DL_MAE:", best_row["DL_MAE"])
print("SARIMA_MAE:", best_row["SARIMA_MAE"])
print("ENS_MAE:", best_row["ENS_MAE"])
print("DL_time_s:", best_row["DL_time_s"])
print("SARIMA_fit_s:", best_row["SARIMA_fit_s"])