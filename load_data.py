"""
load_data.py — Load and validate OHLCV data with configurable resampling.

Per Agent.md:
- Input data can be 1-minute or 5-minute OHLCV
- Resampled to configurable timeframe (5m, 15m, 30m)
- Data filtered to 2023-2025 only
- Sessions are well-defined and continuous

Timestamps are in Pacific Time (PST/PDT). DST transitions are handled by
using wall-clock time for all session and phase boundaries.

Session definition: 6:30am PT to 6:25am PT next day (~24 hours).
This captures US RTH, afternoon Globex, Asia session, and London session.

Phase assignment for each bar:
- 'pre_ib':  6:30am to 7:25am (forms the Initial Balance; discovery vs prior VAH/VAL)
- 'post_ib': 7:30am to 12:55pm (discovery vs IBH/IBL)
- 'overnight': all other times (included in volume profile, not in episode analysis)
- 'prior_rth': 6:30am - 1:00pm (previous day, for prior RTH value)
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import time, timedelta, date as date_type

from config import CONFIG, SESSION_TIMES

# Try multiple potential data files
DATA_FILES = [
    os.path.join(os.path.dirname(__file__), "MNQ_1min_2023_2025.csv"),
    os.path.join(os.path.dirname(__file__), "MNQ_5min_2021Jan_2026Jan.csv"),
]
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output", "cleaned_data.csv")

# Column mappings for different data formats
COLUMN_MAPS = {
    'format1': {
        "DateTime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume(from bar)": "volume",
    },
    'format2': {
        "datetime": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
}

# Session boundary: 6:30am PT
SESSION_START_TIME = time(*SESSION_TIMES['rth_start'])

# Phase boundaries
PRE_IB_START = time(*SESSION_TIMES['pre_ib_start'])
PRE_IB_END = time(*SESSION_TIMES['pre_ib_end'])
POST_IB_START = time(*SESSION_TIMES['post_ib_start'])
POST_IB_END = time(*SESSION_TIMES['post_ib_end'])
OVERNIGHT_START = time(*SESSION_TIMES['overnight_start'])
OVERNIGHT_END = time(*SESSION_TIMES['overnight_end'])


def detect_input_timeframe(df):
    """Detect the timeframe of input data based on timestamp differences."""
    if len(df) < 2:
        return '5m'
    
    diffs = df['datetime'].diff().dropna()
    median_diff = diffs.median()
    
    if median_diff <= pd.Timedelta(minutes=2):
        return '1m'
    elif median_diff <= pd.Timedelta(minutes=7):
        return '5m'
    elif median_diff <= pd.Timedelta(minutes=20):
        return '15m'
    else:
        return '30m'


def resample_to_timeframe(df, target_timeframe):
    """Resample data to target timeframe.
    
    OHLCV Aggregation Rules:
    - O (Open): First bar's open
    - H (High): Maximum of all bar highs
    - L (Low): Minimum of all bar lows
    - C (Close): Last bar's close
    - V (Volume): Sum of all bar volumes
    """
    if target_timeframe == '1m':
        return df
    
    df = df.copy()
    df = df.set_index('datetime')
    
    # Resample to target timeframe
    resampled = df.resample(target_timeframe, label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled = resampled.reset_index()
    
    return resampled


def assign_session_date(dt):
    """Assign a bar to its session date based on the 6:30am PT boundary.

    Bars at or after 6:30am belong to that calendar date's session.
    Bars before 6:30am belong to the previous calendar date's session.
    This groups overnight bars with the preceding RTH session.
    """
    if dt.time() >= SESSION_START_TIME:
        return dt.date()
    else:
        return (dt - timedelta(days=1)).date()


def assign_phase(dt):
    """Assign the trading phase for a bar based on its time.

    Returns:
        'pre_ib':   6:30am - 7:25am (IB formation period)
        'post_ib':  7:30am - 12:55pm (main session)
        'overnight': all other times (volume profile only)
    """
    t = dt.time()
    if PRE_IB_START <= t <= PRE_IB_END:
        return "pre_ib"
    elif POST_IB_START <= t <= POST_IB_END:
        return "post_ib"
    else:
        return "overnight"


def assign_prior_rth_flag(dt):
    """Check if bar is in prior RTH window (6:30am - 1:00pm).
    
    Used to compute prior RTH value area separately.
    """
    t = dt.time()
    rth_start = time(*SESSION_TIMES['prior_rth_start'])
    rth_end = time(*SESSION_TIMES['prior_rth_end'])
    return rth_start <= t <= rth_end


def load_and_validate(filepath=None, target_timeframe=None):
    """Load CSV, rename columns, parse datetime, resample, validate integrity.

    Args:
        filepath: Path to data file (auto-detected if None)
        target_timeframe: Target timeframe for resampling (from config if None)
    
    Returns:
        pd.DataFrame with columns: datetime, open, high, low, close, volume,
        session_date, phase
        Sorted by datetime ascending with no duplicate timestamps.
    """
    if target_timeframe is None:
        target_timeframe = CONFIG['timeframe']
    
    # Find data file
    if filepath is None:
        filepath = None
        for f in DATA_FILES:
            if os.path.exists(f):
                filepath = f
                break
        if filepath is None:
            raise FileNotFoundError(f"No data file found. Tried: {DATA_FILES}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Try different column mappings
    column_map = None
    for fmt_name, fmt_map in COLUMN_MAPS.items():
        if all(c in df.columns for c in fmt_map.keys()):
            column_map = fmt_map
            break
    
    if column_map is None:
        # Try case-insensitive match
        df.columns = df.columns.str.strip()
        lower_cols = {c.lower(): c for c in df.columns}
        
        standard_cols = {'datetime': None, 'open': None, 'high': None, 
                        'low': None, 'close': None, 'volume': None}
        for std_col in standard_cols:
            for orig_col, mapped_col in lower_cols.items():
                if std_col in orig_col:
                    standard_cols[std_col] = mapped_col
                    break
        
        if all(v is not None for v in standard_cols.values()):
            column_map = {v: k for k, v in standard_cols.items()}
        else:
            raise ValueError(f"Cannot map columns. Available: {list(df.columns)}")

    # Rename to standard names
    df = df.rename(columns=column_map)

    # Keep only the columns we need
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df[required].copy()

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # Remove duplicate timestamps (keep first occurrence)
    dup_count = df["datetime"].duplicated().sum()
    if dup_count > 0:
        print(f"Warning: Removed {dup_count} duplicate timestamps")
        df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)

    # Detect input timeframe and resample if needed
    input_tf = detect_input_timeframe(df)
    print(f"Detected input timeframe: {input_tf}")
    
    if input_tf != target_timeframe:
        print(f"Resampling from {input_tf} to {target_timeframe}...")
        df = resample_to_timeframe(df, target_timeframe)
        print(f"Resampled to {len(df)} bars")

    # Validate ordering — timestamps must be strictly increasing
    diffs = df["datetime"].diff().dropna()
    if (diffs <= pd.Timedelta(0)).any():
        raise ValueError("Timestamps are not strictly increasing after dedup")

    # Validate OHLCV integrity
    invalid_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    invalid_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    if invalid_high > 0:
        print(f"Warning: {invalid_high} bars where high < max(open, close)")
    if invalid_low > 0:
        print(f"Warning: {invalid_low} bars where low > min(open, close)")

    # Validate no null prices
    null_count = df[["open", "high", "low", "close"]].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Found {null_count} null price values")

    # Validate volume is non-negative
    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        raise ValueError(f"Found {neg_vol} negative volume values")

    # Assign session date and phase
    df["session_date"] = df["datetime"].apply(assign_session_date)
    df["phase"] = df["datetime"].apply(assign_phase)
    
    # Add flag for prior RTH bars (used in session_profile.py)
    df["is_prior_rth"] = df["datetime"].apply(assign_prior_rth_flag)

    # Keep legacy 'date' column as calendar date for compatibility
    df["date"] = df["datetime"].dt.date

    # Filter to configured year range
    start_year = CONFIG['data_start_year']
    end_year = CONFIG['data_end_year']
    df["year"] = df["datetime"].dt.year
    original_len = len(df)
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    df = df.drop(columns=["year"]).reset_index(drop=True)
    
    if len(df) < original_len:
        print(f"Filtered to {start_year}-{end_year}: {original_len} -> {len(df)} bars")

    n_sessions = df["session_date"].nunique()
    phase_counts = df["phase"].value_counts()
    print(f"Loaded {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"Sessions (6:30am-6:25am): {n_sessions}")
    print(f"Phases: pre_ib={phase_counts.get('pre_ib', 0)}, "
          f"post_ib={phase_counts.get('post_ib', 0)}, "
          f"overnight={phase_counts.get('overnight', 0)}")

    return df


def save_cleaned(df, output_path=None):
    """Save cleaned data to CSV."""
    if output_path is None:
        output_path = OUTPUT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    df = load_and_validate()
    save_cleaned(df)
