import os
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



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