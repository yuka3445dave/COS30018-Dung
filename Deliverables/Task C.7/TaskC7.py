import os
import re
import csv
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Utils
# -----------------------------
def _safe_filename(name: str) -> str:
    """Make a string safe to use as a filename."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns (especially when downloading multiple tickers).
    This function flattens them into single-level columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Example: ('Close', 'CBA.AX') -> 'Close'
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


# -----------------------------
# Task C.7 - Data download + cache (from C.5)
# -----------------------------
def download_or_load_stock_data(
    company: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data_cache",
    use_cache: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV stock data from yfinance OR load from local CSV cache.

    Why caching matters:
    - yfinance depends on network and can be slow.
    - caching saves time when rerunning experiments.

    If use_cache=True:
      - If the CSV exists and force_download=False, load it.
      - Otherwise download and save it.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_name = _safe_filename(f"{company}_{start_date}_{end_date}.csv")
    cache_path = os.path.join(cache_dir, cache_name)

    if use_cache and (not force_download) and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded from cache: {cache_path}")
    else:
        end_plus_1 = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(company, start=start_date, end=end_plus_1, progress=False)
        df = _flatten_yfinance_columns(df)

        if use_cache:
            df.to_csv(cache_path)
            print(f"Saved to cache: {cache_path}")

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Keep only standard OHLCV columns if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    return df


def finviz_quick_check(ticker: str, timeout: int = 20):
    """
    Quick check:
    1) Can we open FinViz quote page?
    2) Does it have a news table with at least 1 headline?
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36"
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    status = r.status_code

    if status != 200:
        print(f"[{ticker}] FAIL: HTTP {status} -> {url}")
        return False

    soup = BeautifulSoup(r.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")

    if news_table is None:
        print(f"[{ticker}] FAIL: Page exists but no news table found -> {url}")
        return False

    rows = news_table.find_all("tr")
    if len(rows) == 0:
        print(f"[{ticker}] FAIL: News table exists but empty -> {url}")
        return False

    # Print first 3 headlines as proof
    print(f"[{ticker}] OK: Found {len(rows)} news rows on FinViz.")
    for row in rows[:3]:
        cols = row.find_all("td")
        if len(cols) >= 2:
            dt = cols[0].get_text(strip=True)
            headline = cols[1].get_text(" ", strip=True)
            print("  -", dt, "|", headline[:120])
    return True


def fetch_finviz_headlines(ticker: str, out_csv: str, timeout: int = 20, use_cache: bool = True):
    """
    Scrape FinViz news table from quote page and save to CSV.
    Output columns: date, time, headline, source, link

    FinViz date/time format:
    - First row of a day: "Apr-30-25 02:40AM"
    - Subsequent rows same day: "03:10AM" (date omitted)
    We carry forward the last seen date.
    """
    if use_cache and os.path.exists(out_csv):
        print(f"Loaded FinViz headlines from cache: {out_csv}")
        return pd.read_csv(out_csv)
    
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36"
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    if news_table is None:
        raise RuntimeError(f"No FinViz news table found for {ticker}: {url}")

    rows = news_table.find_all("tr")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    data = []
    current_date_str = None  # FinViz uses like "Apr-30-25"

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        dt_text = cols[0].get_text(" ", strip=True)
        # headline + link
        a = cols[1].find("a")
        headline = a.get_text(" ", strip=True) if a else cols[1].get_text(" ", strip=True)
        link = a["href"] if (a and a.has_attr("href")) else ""

        # source (usually in <span>)
        source_span = cols[1].find("span")
        source = source_span.get_text(strip=True) if source_span else ""

        parts = dt_text.split()
        if len(parts) == 2:
            current_date_str = parts[0]   # e.g. "Apr-30-25"
            time_str = parts[1]           # e.g. "02:40AM"
        else:
            time_str = parts[0]           # e.g. "03:10AM"
            if current_date_str is None:
                continue

        data.append([current_date_str, time_str, headline, source, link])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date_finviz", "time", "headline", "source", "link"])
        writer.writerows(data)

    print(f"Saved {len(data)} FinViz headlines to {out_csv}")
    return pd.DataFrame(data, columns=["date_finviz", "time", "headline", "source", "link"])

def av_news_quick_check(ticker: str | None = None, limit=50, topics: str | None = None):
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("ERROR: ALPHAVANTAGE_API_KEY is missing in this Python process.")
        return False

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "limit": limit,
        "apikey": api_key,
        "sort": "LATEST",
    }
    if ticker:
        params["tickers"] = ticker
    if topics:
        params["topics"] = topics

    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    # Print real error/throttle messages (most important for debugging)
    if "Error Message" in data:
        print(f"[AV] Error Message for {ticker or topics}: {data['Error Message']}")
        return False
    if "Information" in data:
        print(f"[AV] Information for {ticker or topics}: {data['Information']}")
        return False
    if "Note" in data:
        print(f"[AV] Note for {ticker or topics}: {data['Note']}")
        return False

    feed = data.get("feed", [])
    print(f"[{ticker or topics}] feed items:", len(feed))
    if feed:
        print("  sample:", feed[0].get("time_published"), "|", feed[0].get("title"))
        return True

    print(f"[{ticker or topics}] No feed returned. Keys:", list(data.keys())[:10])
    return False

def fetch_alpha_vantage_news(
    tickers=None,
    topics=None,
    time_from=None,   # "YYYYMMDDTHHMM"
    time_to=None,     # "YYYYMMDDTHHMM"
    sort="LATEST",    # "LATEST" / "EARLIEST" / "RELEVANCE"
    limit=50,
    cache_path="news_cache/av_news.json",
    use_cache=True,
    timeout=30
):
    """
    Fetch Alpha Vantage NEWS_SENTIMENT feed and cache to JSON to avoid rate limit.
    Returns a DataFrame with: date, headline, source, url
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded Alpha Vantage news from cache: {cache_path}")
    else:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY not found in environment.")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": api_key,
            "limit": limit,
            "sort": sort,
        }
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        r = requests.get(url, params=params, timeout=timeout)
        data = r.json()

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage Error Message: {data['Error Message']}")
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage Information: {data['Information']}")
        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage Note: {data['Note']}")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved Alpha Vantage news to cache: {cache_path}")

        time.sleep(1.2)

    feed = data.get("feed", [])
    rows = []
    for item in feed:
        tpub = item.get("time_published", "")
        if len(tpub) >= 8:
            date = f"{tpub[:4]}-{tpub[4:6]}-{tpub[6:8]}"
        else:
            date = None

        rows.append({
            "date": date,
            "headline": item.get("title", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        })

    df_news = pd.DataFrame(rows).dropna(subset=["date"])
    print(f"Alpha Vantage feed items parsed: {len(df_news)}")
    return df_news

def parse_finviz_date(date_finviz: str) -> str:
    """
    Convert FinViz date like 'Apr-30-25' -> '2025-04-30'
    """
    return datetime.strptime(date_finviz, "%b-%d-%y").strftime("%Y-%m-%d")

def standardize_finviz_df(df_finviz: pd.DataFrame) -> pd.DataFrame:
    df = df_finviz.copy()
    df["date"] = df["date_finviz"].apply(parse_finviz_date)
    # normalize source like "(Seeking Alpha)" -> "Seeking Alpha"
    df["source"] = df["source"].astype(str).str.strip().str.strip("()")
    df["url"] = df["link"]
    df["origin"] = "finviz"
    return df[["date", "headline", "source", "url", "origin"]]

def standardize_av_df(df_av: pd.DataFrame) -> pd.DataFrame:
    df = df_av.copy()
    df["origin"] = "alphavantage"
    # already has date/headline/source/url
    return df[["date", "headline", "source", "url", "origin"]]

def combine_and_dedupe_news(df_finviz_std: pd.DataFrame, df_av_std: pd.DataFrame) -> pd.DataFrame:
    df_all = pd.concat([df_finviz_std, df_av_std], ignore_index=True)
    # Basic cleaning + dedupe
    df_all["headline_clean"] = (
        df_all["headline"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df_all = df_all.drop_duplicates(subset=["date", "headline_clean"]).drop(columns=["headline_clean"])
    df_all = df_all.sort_values(["date", "origin"]).reset_index(drop=True)
    return df_all

def add_vader_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df = df_news.copy()
    df["vader_compound"] = df["headline"].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

def aggregate_daily_sentiment(df_news_scored: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df_news_scored.groupby("date")
        .agg(
            sent_mean=("vader_compound", "mean"),
            sent_sum=("vader_compound", "sum"),
            sent_count=("vader_compound", "count"),
        )
        .reset_index()
        .sort_values("date")
    )
    return daily

def add_finbert_sentiment(
    df_news: pd.DataFrame,
    cache_csv: str = "news_cache/news_scored_finbert.csv",
    use_cache: bool = True,
    batch_size: int = 16,
    model_name: str = "ProsusAI/finbert"
) -> pd.DataFrame:
    """
    Run FinBERT on each headline and store per-headline sentiment probabilities.
    Creates finbert_score = P(pos) - P(neg) in [-1, 1].
    Caches output to CSV to avoid rerunning model inference.
    """
    os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)

    if use_cache and os.path.exists(cache_csv):
        print(f"Loaded FinBERT scored headlines from cache: {cache_csv}")
        return pd.read_csv(cache_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # label order in ProsusAI/finbert is usually: [negative, neutral, positive]
    id2label = model.config.id2label
    # Build mapping robustly
    label_to_index = {v.lower(): k for k, v in id2label.items()}

    headlines = df_news["headline"].astype(str).tolist()

    rows = []
    with torch.no_grad():
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()

            for j, text in enumerate(batch):
                p_neg = float(probs[j][label_to_index.get("negative", 0)])
                p_neu = float(probs[j][label_to_index.get("neutral", 1)])
                p_pos = float(probs[j][label_to_index.get("positive", 2)])

                rows.append({
                    "date": df_news.iloc[i + j]["date"],
                    "headline": text,
                    "source": df_news.iloc[i + j].get("source", ""),
                    "url": df_news.iloc[i + j].get("url", ""),
                    "origin": df_news.iloc[i + j].get("origin", ""),
                    "finbert_neg": p_neg,
                    "finbert_neu": p_neu,
                    "finbert_pos": p_pos,
                    "finbert_score": p_pos - p_neg
                })

    df_scored = pd.DataFrame(rows)
    df_scored.to_csv(cache_csv, index=False)
    print(f"Saved FinBERT scored headlines to: {cache_csv}")
    return df_scored


def aggregate_daily_finbert(df_finbert_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate FinBERT scores per day.
    """
    daily = (
        df_finbert_scored.groupby("date")
        .agg(
            finbert_mean=("finbert_score", "mean"),
            finbert_sum=("finbert_score", "sum"),
            finbert_count=("finbert_score", "count"),
        )
        .reset_index()
        .sort_values("date")
    )
    return daily


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

def filter_news_by_keywords(df_news: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    pattern = "|".join([re.escape(k) for k in keywords])
    mask = df_news["headline"].astype(str).str.contains(pattern, case=False, regex=True)
    return df_news[mask].copy()

def fetch_alpha_vantage_news_windows(
    topics: str,
    start_date: str,
    end_date: str,
    window_days: int = 90,
    limit: int = 200,
    cache_dir: str = "news_cache/av_windows",
    use_cache: bool = True,
    sort: str = "EARLIEST",
):
    """
    Fetch Alpha Vantage news in multiple time windows to increase coverage.
    Each window is cached to JSON to avoid rate limits on re-run.
    """
    os.makedirs(cache_dir, exist_ok=True)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    all_parts = []
    cur = start_dt

    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=window_days), end_dt)

        # Alpha Vantage wants YYYYMMDDTHHMM
        time_from = cur.strftime("%Y%m%dT0000")
        time_to = nxt.strftime("%Y%m%dT2359")

        cache_path = os.path.join(cache_dir, f"av_{time_from}_{time_to}.json")

        df_part = fetch_alpha_vantage_news(
            tickers=None,
            topics=topics,
            time_from=time_from,
            time_to=time_to,
            sort=sort,
            limit=limit,
            cache_path=cache_path,
            use_cache=use_cache,
            timeout=30,
        )
        all_parts.append(df_part)

        # Only sleep if we actually hit the network (simple approach: always sleep a bit)
        time.sleep(1.2)

        cur = nxt + pd.Timedelta(days=1)

    df_all = pd.concat(all_parts, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["date", "headline"])
    return df_all

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

def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Time-based split (no leakage).
    """
    n = len(df)
    split = int(n * train_ratio)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    return train_df, test_df

def eval_classifier(y_true, y_pred, title="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {title} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix:\n", cm)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

def predict_direction(model, X_row, threshold=0.5):
    """
    Returns predicted label and probability of UP (class 1).
    """
    prob_up = float(model.predict_proba(X_row.reshape(1, -1))[0][1])
    pred = 1 if prob_up >= threshold else 0
    return pred, prob_up

def plot_close_with_sentiment(df_price_feat: pd.DataFrame, df_daily_sent: pd.DataFrame, out_png="figures/close_vs_sentiment.png"):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    dfp = df_price_feat.copy()
    dfp["date_dt"] = pd.to_datetime(dfp["date"])
    dfs = df_daily_sent.copy()
    dfs["date_dt"] = pd.to_datetime(dfs["date"])

    merged = dfp.merge(dfs[["date_dt", "sent_mean"]], on="date_dt", how="left")

    plt.figure()
    plt.plot(merged["date_dt"], merged["Close"], label="Close")
    # sentiment markers only where available
    mask = merged["sent_mean"].notna() & (merged["sent_mean"] != 0)
    colors = merged.loc[mask, "sent_mean"].apply(lambda x: "green" if x > 0 else "red")
    plt.scatter(merged.loc[mask, "date_dt"], merged.loc[mask, "Close"], s=20, c=colors)
    plt.title("CBA.AX Close Price with Sentiment Days (VADER)")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot: {out_png}")


def plot_confusion_matrix(cm, title, out_png):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["DOWN", "UP"])
    plt.yticks([0, 1], ["DOWN", "UP"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot: {out_png}")

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