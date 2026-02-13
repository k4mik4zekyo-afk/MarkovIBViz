"""
volatility.py — Compute ATR on resampled bars.

Per Agent.md:
- ATR window: Configurable (default: 5 bars)
- Uses Wilder's smoothing method

ATR (Average True Range):
True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
ATR = Wilder's smoothed moving average of TR over period
"""

import pandas as pd
import numpy as np
import os

from config import CONFIG

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output", "volatility_data.csv")


def compute_atr(df, period=None):
    """Compute ATR using Wilder's smoothing method.

    Args:
        df: DataFrame with high, low, close columns
        period: ATR lookback period (default from config)
    
    Adds column: atr_{period}
    """
    if period is None:
        period = CONFIG['atr_window']
    
    df = df.copy()

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    n = len(df)
    tr = np.zeros(n)

    # First bar: TR = high - low (no previous close)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        prev_close = close[i - 1]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )

    # Wilder's smoothed ATR
    atr = np.full(n, np.nan)

    # First ATR value at index period-1
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    col_name = f"atr_{period}"
    df[col_name] = atr
    
    # Also add as atr_14 for backward compatibility if period != 14
    if period != 14:
        df["atr_14"] = atr

    non_null = np.count_nonzero(~np.isnan(atr))
    print(f"Computed ATR({period}) for {non_null} / {len(df)} bars")

    return df


def save_volatility(df, output_path=None, period=None):
    """Save volatility data to CSV."""
    if period is None:
        period = CONFIG['atr_window']
    if output_path is None:
        output_path = OUTPUT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    col_name = f"atr_{period}"
    cols = ["datetime", col_name]
    if "atr_14" in df.columns and "atr_14" != col_name:
        cols.append("atr_14")
    
    df[cols].to_csv(output_path, index=False)
    print(f"Saved volatility data to {output_path}")


if __name__ == "__main__":
    from load_data import load_and_validate
    df = load_and_validate()
    df = compute_atr(df)
    save_volatility(df)
