import re
import pandas as pd

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