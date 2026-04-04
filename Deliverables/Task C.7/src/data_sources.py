import os
import re
import csv
import json
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup

from c7_utils import _safe_filename, _flatten_yfinance_columns

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

def standardize_av_df(df_av: pd.DataFrame) -> pd.DataFrame:
    df = df_av.copy()
    df["origin"] = "alphavantage"
    # already has date/headline/source/url
    return df[["date", "headline", "source", "url", "origin"]]

def filter_news_by_keywords(df_news: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    pattern = "|".join([re.escape(k) for k in keywords])
    mask = df_news["headline"].astype(str).str.contains(pattern, case=False, regex=True)
    return df_news[mask].copy()

def combine_and_dedupe_news(df_finviz_std: pd.DataFrame, df_av_std: pd.DataFrame) -> pd.DataFrame:
    df_all = pd.concat([df_finviz_std, df_av_std], ignore_index=True)
    # Basic cleaning + dedupe
    df_all["headline_clean"] = (
        df_all["headline"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df_all = df_all.drop_duplicates(subset=["date", "headline_clean"]).drop(columns=["headline_clean"])
    df_all = df_all.sort_values(["date", "origin"]).reset_index(drop=True)
    return df_all