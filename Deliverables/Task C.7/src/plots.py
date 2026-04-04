import os
import pandas as pd
import matplotlib.pyplot as plt

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