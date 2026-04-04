import os
import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_sources import (
    download_or_load_stock_data,
    fetch_alpha_vantage_news_windows,
    fetch_finviz_headlines,
    standardize_finviz_df,
    standardize_av_df,
    filter_news_by_keywords,
    combine_and_dedupe_news,
    finviz_quick_check,
)

from sentiment import (
    add_vader_sentiment,
    aggregate_daily_sentiment,
    add_finbert_sentiment,
    aggregate_daily_finbert,
)

from features import (
    build_price_features,
    merge_sentiment_features,
    merge_sentiment_features_both,
)

from evaluation import (
    time_split,
    eval_classifier,
)

from models import (
    predict_direction,
)

from plots import (
    plot_close_with_sentiment,
    plot_confusion_matrix,
)


# -----------------------------
# Quick sanity test (Step 0 output)
# -----------------------------
if __name__ == "__main__":
    COMPANY = "CBA.AX"
    START = "2025-01-01"
    END = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 1) Price data test
    df = download_or_load_stock_data(
        company=COMPANY,
        start_date=START,
        end_date=END,
        cache_dir="data_cache",
        use_cache=True,
        force_download=False,
    )
    print("Downloaded df shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # 2) Alpha Vantage fetch (ONE call per run, cached)
    TOPICS = "finance,economy_macro,economy_monetary"

    df_av = fetch_alpha_vantage_news_windows(
        topics=TOPICS,
        start_date=START,
        end_date=END,
        window_days=90,      # ~4 windows/year
        limit=200,
        use_cache=True,
        sort="EARLIEST"
    )
    print("Alpha Vantage windows total rows:", len(df_av))
    print(df_av.head())

    keywords = ["australia", "australian", "rba", "aud", "asx", "sydney", "melbourne", "commonwealth", "bank"]

    NEWS_TICKER = "EWA"
    finviz_csv = "news_cache/finviz_EWA.csv"

    df_finviz = fetch_finviz_headlines(NEWS_TICKER, finviz_csv, use_cache=True)
    print("\nFinViz sample:")
    print(df_finviz.head())

    df_finviz_std = standardize_finviz_df(df_finviz)
    df_av_std = standardize_av_df(df_av)
    df_av_std = filter_news_by_keywords(df_av_std, keywords)
    print("Alpha Vantage (topics) after keyword filter:", len(df_av_std))

    df_news_all = combine_and_dedupe_news(df_finviz_std, df_av_std)

    # --- IMPORTANT: filter combined news to match the price date range ---
    start_dt = pd.to_datetime(START)
    end_dt = pd.to_datetime(END)

    df_news_all["date"] = pd.to_datetime(df_news_all["date"], errors="coerce")
    df_news_all = df_news_all.dropna(subset=["date"])

    df_news_all = df_news_all[(df_news_all["date"] >= start_dt) & (df_news_all["date"] <= end_dt)].copy()
    df_news_all["date"] = df_news_all["date"].dt.strftime("%Y-%m-%d")

    print("\n[After filtering] Combined news rows:", len(df_news_all))
    print("[After filtering] News date range:", df_news_all["date"].min(), "->", df_news_all["date"].max())

    df_scored = add_vader_sentiment(df_news_all)
    df_scored.to_csv("news_cache/news_scored_vader.csv", index=False)
    print("Saved headline-level VADER scores to: news_cache/news_scored_vader.csv")
    df_daily_sent = aggregate_daily_sentiment(df_scored)
    print("\nVADER score stats:")
    print(df_scored["vader_compound"].describe())


    price_dates = set(df.index.strftime("%Y-%m-%d"))
    sent_dates = set(df_daily_sent["date"].tolist())
    print("\nPrice days:", len(price_dates), "| Sentiment days:", len(sent_dates), "| Overlap:", len(price_dates & sent_dates))

    print("\nDaily sentiment rows:", len(df_daily_sent))
    print(df_daily_sent.head())

    df_daily_sent.to_csv("news_cache/sentiment_daily_vader.csv", index=False)
    print("Saved daily sentiment to: news_cache/sentiment_daily_vader.csv")

    

    # -----------------------------
    # FinBERT (Independent Research) + daily aggregation
    # -----------------------------
    df_finbert_scored = add_finbert_sentiment(
        df_news_all,
        cache_csv="news_cache/news_scored_finbert.csv",
        use_cache=True,
        batch_size=16,
        model_name="ProsusAI/finbert"
    )

    df_daily_finbert = aggregate_daily_finbert(df_finbert_scored)
    df_daily_finbert.to_csv("news_cache/sentiment_daily_finbert.csv", index=False)
    print("Saved daily FinBERT sentiment to: news_cache/sentiment_daily_finbert.csv")

    print("\nFinBERT daily rows:", len(df_daily_finbert))
    print(df_daily_finbert.head())

    # -----------------------------
    # Step 5: Build classification dataset
    # -----------------------------
    df_feat = build_price_features(df)  # df is your price dataframe
    
    df_cls = merge_sentiment_features_both(df_feat, df_daily_sent, df_daily_finbert)

    print("\nClassification dataset shape:", df_cls.shape)
    print("Label distribution (0=DOWN,1=UP):")
    print(df_cls["label_up"].value_counts(normalize=True))

    # Time split
    train_df, test_df = time_split(df_cls, train_ratio=0.8)

    # Feature sets
    price_cols = ["ret_1d", "ma_5", "ma_10", "vol_5", "volchg_1d"]
    vader_cols = ["sent_mean", "sent_sum", "sent_count"]
    finbert_cols = ["finbert_mean", "finbert_sum", "finbert_count"]

    X_train_vader = train_df[price_cols + vader_cols].values
    X_test_vader  = test_df[price_cols + vader_cols].values

    X_train_finbert = train_df[price_cols + finbert_cols].values
    X_test_finbert  = test_df[price_cols + finbert_cols].values

    X_train_both = train_df[price_cols + vader_cols + finbert_cols].values
    X_test_both  = test_df[price_cols + vader_cols + finbert_cols].values

    X_train_base = train_df[price_cols].values
    X_test_base = test_df[price_cols].values

    # X_train_sent = train_df[price_cols + sent_cols].values
    # X_test_sent = test_df[price_cols + sent_cols].values

    y_train = train_df["label_up"].values
    y_test = test_df["label_up"].values

    # Baseline model: Logistic Regression
    lr_base = LogisticRegression(max_iter=2000)
    lr_base.fit(X_train_base, y_train)
    pred_base = lr_base.predict(X_test_base)
    m_base = eval_classifier(y_test, pred_base, title="LogReg (Price-only)")

    # --- Logistic Regression comparisons ---
    lr_vader = LogisticRegression(max_iter=2000)
    lr_vader.fit(X_train_vader, y_train)
    pred_lr_vader = lr_vader.predict(X_test_vader)
    m_lr_vader = eval_classifier(y_test, pred_lr_vader, title="LogReg (Price + VADER)")

    lr_finbert = LogisticRegression(max_iter=2000)
    lr_finbert.fit(X_train_finbert, y_train)
    pred_lr_finbert = lr_finbert.predict(X_test_finbert)
    m_lr_finbert = eval_classifier(y_test, pred_lr_finbert, title="LogReg (Price + FinBERT)")

    lr_both = LogisticRegression(max_iter=2000)
    lr_both.fit(X_train_both, y_train)
    pred_lr_both = lr_both.predict(X_test_both)
    m_lr_both = eval_classifier(y_test, pred_lr_both, title="LogReg (Price + VADER + FinBERT)")

    print("\n=== Comparison (LogReg) ===")
    print("Price-only F1:", round(m_base["f1"], 4),
        "| +VADER F1:", round(m_lr_vader["f1"], 4),
        "| +FinBERT F1:", round(m_lr_finbert["f1"], 4),
        "| +Both F1:", round(m_lr_both["f1"], 4))

    

    # --- RandomForest comparisons (baseline + VADER + FinBERT + BOTH) ---
    rf_base = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_base.fit(X_train_base, y_train)
    pred_rf_base = rf_base.predict(X_test_base)
    m_rf_base = eval_classifier(y_test, pred_rf_base, title="RandomForest (Price-only)")

    rf_vader = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )
    rf_vader.fit(X_train_vader, y_train)
    pred_rf_vader = rf_vader.predict(X_test_vader)
    m_rf_vader = eval_classifier(y_test, pred_rf_vader, title="RandomForest (Price + VADER)")

    # -----------------------------
    # Final demo prediction (use the chosen model)
    # -----------------------------
    print("\nFinal demo model: RandomForest (Price + VADER)")

    latest = test_df.iloc[-1]
    X_latest = latest[price_cols + vader_cols].values.astype(float)

    pred, prob_up = predict_direction(rf_vader, X_latest, threshold=0.5)

    print("\nNext trading day direction prediction:")
    print("Predicted label (1=UP, 0=DOWN):", pred)
    print("Probability of UP:", round(prob_up, 4))
    print("Last test day (date):", latest["date"])
    print("Close on last test day:", latest["Close"])
    print("Actual next-day label (1=UP,0=DOWN):", latest["label_up"])

    rf_finbert = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_finbert.fit(X_train_finbert, y_train)
    pred_rf_finbert = rf_finbert.predict(X_test_finbert)
    m_rf_finbert = eval_classifier(y_test, pred_rf_finbert, title="RandomForest (Price + FinBERT)")

    # rf_both = RandomForestClassifier(n_estimators=300, random_state=42)
    # rf_both.fit(X_train_both, y_train)
    # pred_rf_both = rf_both.predict(X_test_both)
    # m_rf_both = eval_classifier(y_test, pred_rf_both, title="RandomForest (Price + VADER + FinBERT)")

    # print("\n=== Comparison (RandomForest) ===")
    # print("Price-only F1:", round(m_rf_base["f1"], 4),
    #     "| +VADER F1:", round(m_rf_vader["f1"], 4),
    #     "| +FinBERT F1:", round(m_rf_finbert["f1"], 4),
    #     "| +Both F1:", round(m_rf_both["f1"], 4))

  
    # # -----------------------------
    # # Final demo prediction (ONE block only)
    # # -----------------------------
    # print("\nFinal demo model: RandomForest (Price + VADER)")

    # latest = test_df.iloc[-1]
    # X_latest = latest[price_cols + vader_cols].values.astype(float)

    # pred, prob_up = predict_direction(rf_vader, X_latest, threshold=0.5)

    # print("\nNext trading day direction prediction:")
    # print("Predicted label (1=UP, 0=DOWN):", pred)
    # print("Probability of UP:", round(prob_up, 4))
    # print("Last test day (date):", latest["date"])
    # print("Close on last test day:", latest["Close"])
    # print("Actual next-day label (1=UP,0=DOWN):", latest["label_up"])

    # Simple comparison
    
    print("Price-only F1:", round(m_base["f1"], 4),
      "| +VADER F1:", round(m_lr_vader["f1"], 4),
      "| +FinBERT F1:", round(m_lr_finbert["f1"], 4),
      "| +Both F1:", round(m_lr_both["f1"], 4))

    print("\nCombined news rows:", len(df_news_all))
    
    # -----------------------------
    # Save combined news (report evidence)
    # -----------------------------
    print("\nCombined news rows:", len(df_news_all))
    print("News date range:", df_news_all["date"].min(), "->", df_news_all["date"].max())
    print(df_news_all.head())

    df_news_all.to_csv("news_cache/news_combined_EWA.csv", index=False)
    print("Saved combined news to: news_cache/news_combined_EWA.csv")

    plot_close_with_sentiment(
        df_feat,
        df_daily_sent,
        out_png="figures/close_vs_sentiment.png"
    )

    plot_confusion_matrix(
        m_rf_vader["cm"],
        "RandomForest (Price + VADER)",
        "figures/cm_rf_vader.png"
    )

    # -----------------------------
    # Final demo prediction (ONE block only)
    # Choose ONE final model: RandomForest (Price + VADER)
    # -----------------------------
    print("\nFinal demo model: RandomForest (Price + VADER)")

    latest = test_df.iloc[-1]
    X_latest = latest[price_cols + vader_cols].values.astype(float)

    pred, prob_up = predict_direction(rf_vader, X_latest, threshold=0.5)
    direction = "UP" if pred == 1 else "DOWN"

    print("\nNext trading day direction prediction:")
    print("Direction:", direction)
    print("Predicted label (1=UP, 0=DOWN):", pred)
    print("Probability of UP:", round(prob_up, 4))
    print("Last test day (date):", latest["date"])
    print("Close on last test day:", latest["Close"])
    print("Actual next-day label (1=UP,0=DOWN):", latest["label_up"])