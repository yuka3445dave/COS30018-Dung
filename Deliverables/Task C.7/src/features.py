import pandas as pd

def build_price_features(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic price-derived features for classification.
    Uses only information available up to day t.
    """
    df = df_price.copy()
    df["date"] = df.index.strftime("%Y-%m-%d")

    # 1-day return
    df["ret_1d"] = df["Close"].pct_change()

    # moving averages
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()

    # volatility (std of returns)
    df["vol_5"] = df["ret_1d"].rolling(5).std()

    # volume change
    df["volchg_1d"] = df["Volume"].pct_change()

    # label: UP/DOWN tomorrow
    df["close_next"] = df["Close"].shift(-1)
    df["label_up"] = (df["close_next"] > df["Close"]).astype(int)

    # Drop rows with NaNs caused by rolling/pct_change/shift
    df = df.dropna().reset_index(drop=True)

    return df

def merge_sentiment_features(df_feat: pd.DataFrame, df_daily_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sentiment onto price features. Fill missing sentiment with neutral 0.
    """
    df = df_feat.merge(df_daily_sent, on="date", how="left")
    df["sent_mean"] = df["sent_mean"].fillna(0.0)
    df["sent_sum"] = df["sent_sum"].fillna(0.0)
    df["sent_count"] = df["sent_count"].fillna(0)
    return df

def merge_sentiment_features_both(
    df_feat: pd.DataFrame,
    df_daily_vader: pd.DataFrame,
    df_daily_finbert: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge BOTH VADER and FinBERT daily sentiment features into the price feature table.
    Fill missing values with neutral defaults.
    """
    df = df_feat.merge(df_daily_vader, on="date", how="left")
    df = df.merge(df_daily_finbert, on="date", how="left")

    # VADER fill
    df["sent_mean"] = df["sent_mean"].fillna(0.0)
    df["sent_sum"] = df["sent_sum"].fillna(0.0)
    df["sent_count"] = df["sent_count"].fillna(0)

    # FinBERT fill
    df["finbert_mean"] = df["finbert_mean"].fillna(0.0)
    df["finbert_sum"] = df["finbert_sum"].fillna(0.0)
    df["finbert_count"] = df["finbert_count"].fillna(0)

    return df