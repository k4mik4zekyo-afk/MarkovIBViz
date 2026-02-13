"""
vwap_engine.py — Compute Anchored VWAP and standard deviation bands.

VWAP = cumulative(price * volume) / cumulative(volume)
Standard deviation bands at +/- 1 sigma and +/- 2 sigma.

The typical price used is (High + Low + Close) / 3.
VWAP resets at each session start (6:30am PT).
"""

import pandas as pd
import numpy as np
import os

from load_data import load_and_validate

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output", "vwap_data.csv")


def compute_session_vwap(df):
    """Compute session-anchored VWAP with standard deviation bands.

    VWAP anchors to the session start (6:30am PT). Each new session_date
    resets the cumulative calculation.

    Adds columns: vwap, vwap_upper1, vwap_lower1, vwap_upper2, vwap_lower2
    """
    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

    vwap = np.full(len(df), np.nan)
    upper1 = np.full(len(df), np.nan)
    lower1 = np.full(len(df), np.nan)
    upper2 = np.full(len(df), np.nan)
    lower2 = np.full(len(df), np.nan)

    session_dates = df["session_date"].values
    tp = df["typical_price"].values
    vol = df["volume"].values

    # Detect session boundaries
    session_starts = np.where(
        np.concatenate([[True], session_dates[1:] != session_dates[:-1]])
    )[0]
    session_ends = np.concatenate([session_starts[1:], [len(df)]])

    for start, end in zip(session_starts, session_ends):
        cum_tp_vol = 0.0
        cum_vol = 0.0
        sq_devs = []

        for i in range(start, end):
            cum_tp_vol += tp[i] * vol[i]
            cum_vol += vol[i]

            if cum_vol > 0:
                v = cum_tp_vol / cum_vol
                vwap[i] = v
                dev = tp[i] - v
                sq_devs.append(dev ** 2)

                if len(sq_devs) > 1:
                    std = np.sqrt(np.mean(sq_devs))
                else:
                    std = 0.0

                upper1[i] = v + std
                lower1[i] = v - std
                upper2[i] = v + 2 * std
                lower2[i] = v - 2 * std
            else:
                vwap[i] = tp[i]
                upper1[i] = tp[i]
                lower1[i] = tp[i]
                upper2[i] = tp[i]
                lower2[i] = tp[i]

    df["vwap"] = vwap
    df["vwap_upper1"] = upper1
    df["vwap_lower1"] = lower1
    df["vwap_upper2"] = upper2
    df["vwap_lower2"] = lower2

    df = df.drop(columns=["typical_price"])

    non_null = np.count_nonzero(~np.isnan(vwap))
    print(f"Computed VWAP for {non_null} / {len(df)} bars")

    return df


def save_vwap(df, output_path=None):
    """Save VWAP data to CSV."""
    if output_path is None:
        output_path = OUTPUT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[["datetime", "vwap", "vwap_upper1", "vwap_lower1", "vwap_upper2", "vwap_lower2"]].to_csv(
        output_path, index=False
    )
    print(f"Saved VWAP data to {output_path}")


if __name__ == "__main__":
    df = load_and_validate()
    df = compute_session_vwap(df)
    save_vwap(df)
