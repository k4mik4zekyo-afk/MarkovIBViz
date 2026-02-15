"""
smart_range.py — Single-session Opening Range fib-grid trade simulator.

For a single trading day, computes conditional win/loss statistics of
ATR-scaled 2:1 trades triggered every time price touches a fib location
inside the Opening Range.  Overlapping trades are allowed.

This is an intraday event study — not a backtester.

Usage:
    python smart_range.py                       # runs built-in sample days
    python smart_range.py 2025-10-27            # single date
    python smart_range.py 2025-10-27 2025-10-28 # multiple dates
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import time, timedelta

from config import CONFIG, SESSION_TIMES

# ── paths ────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "MNQ_5min_2021Jan_2026Jan.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── defaults (can be overridden via function args) ───────────────────
DEFAULT_OPENING_WINDOW = ("06:30", "06:40")   # 15-min OR
DEFAULT_END_TIME       = "13:00"              # RTH close
DEFAULT_FIB_STEP       = 0.05                 # 10 % fib increments
DEFAULT_ATR_PERIOD     = 3                    # bars for ATR lookback
DEFAULT_ATR_MULTIPLIER = 1.0                  # risk = mult × ATR
DEFAULT_RR_RATIO       = 2.0                  # reward : risk


# =====================================================================
#  DATA HELPERS
# =====================================================================

def load_raw_csv(filepath: str = DATA_FILE) -> pd.DataFrame:
    """Load the raw 5-min CSV and normalise column names / types."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "DateTime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume(from bar)": "volume",
    })
    # Keep only required cols (drop ATR column from CSV if present)
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", dayfirst=False)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def extract_single_day(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Return bars for a single calendar date (midnight-to-midnight).

    For the OR analysis we only need bars from the RTH open onward,
    but we keep all bars for that calendar date so ATR has prior context.
    """
    target = pd.Timestamp(target_date).date()
    mask = df["datetime"].dt.date == target
    day = df[mask].copy().reset_index(drop=True)
    if day.empty:
        raise ValueError(f"No data found for {target_date}")
    return day


# =====================================================================
#  ATR — COMPUTED STRICTLY FROM PAST DATA
# =====================================================================

def compute_rolling_atr(day_data: pd.DataFrame, period: int) -> np.ndarray:
    """Return an array of ATR values (one per bar) using Wilder's smoothing.

    atr[i] is computed from bars 0..i (i.e. NO forward-looking bias).
    The first `period-1` entries are NaN.
    """
    high  = day_data["high"].values
    low   = day_data["low"].values
    close = day_data["close"].values
    n     = len(day_data)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_c = close[i - 1]
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - prev_c),
                     abs(low[i] - prev_c))

    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# =====================================================================
#  CORE ENGINE
# =====================================================================

def analyze_single_day(
    day_data: pd.DataFrame,
    opening_window: tuple[str, str] = DEFAULT_OPENING_WINDOW,
    end_time: str = DEFAULT_END_TIME,
    fib_step: float = DEFAULT_FIB_STEP,
    atr_period: int = DEFAULT_ATR_PERIOD,
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
    rr_ratio: float = DEFAULT_RR_RATIO,
) -> pd.DataFrame:
    """Analyse a single trading session and return fib-level statistics.

    Parameters
    ----------
    day_data : DataFrame
        OHLCV bars for a single trading day (one calendar date).
    opening_window : tuple of ("HH:MM", "HH:MM")
        Start and end of the Opening Range window.
    end_time : str
        Last time at which a trade may still be open ("HH:MM").
    fib_step : float
        Step size for the fib grid (0.10 → 0 %, 10 %, … 100 %).
    atr_period : int
        Number of bars for ATR computation.
    atr_multiplier : float
        Risk = atr_multiplier × ATR.
    rr_ratio : float
        Reward-to-risk ratio (default 2:1).

    Returns
    -------
    DataFrame with columns:
        fib_percent, entry_price,
        long_trades, long_wins, long_losses, long_inconclusive, long_win_rate,
        short_trades, short_wins, short_losses, short_inconclusive, short_win_rate,
        long_ev, short_ev
    """

    # ── 1. Parse time boundaries ────────────────────────────────────
    or_start = _parse_time(opening_window[0])
    or_end   = _parse_time(opening_window[1])
    session_end = _parse_time(end_time)

    # ── 2. Compute Opening Range ────────────────────────────────────
    or_mask = (
        (day_data["datetime"].dt.time >= or_start) &
        (day_data["datetime"].dt.time <= or_end)
    )
    or_bars = day_data[or_mask]
    if or_bars.empty:
        raise ValueError("No bars found in the opening window — check date / times.")

    or_high  = or_bars["high"].max()
    or_low   = or_bars["low"].min()
    or_range = or_high - or_low

    print(f"  Opening Range: {or_low:.2f} – {or_high:.2f}  (range {or_range:.2f})")

    if or_range == 0:
        raise ValueError("Opening Range has zero width — cannot build fib grid.")

    # ── 3. Generate fib grid ────────────────────────────────────────
    fib_percents = np.arange(0, 1 + fib_step / 2, fib_step)
    fib_percents = np.round(fib_percents, 6)            # avoid float drift
    entry_prices = or_low + fib_percents * or_range

    # ── 4. Compute rolling ATR (past-only) ──────────────────────────
    atr_arr = compute_rolling_atr(day_data, atr_period)

    # ── 5. Build bar arrays for the scan window ─────────────────────
    scan_mask = (
        (day_data["datetime"].dt.time > or_end) &
        (day_data["datetime"].dt.time <= session_end)
    )
    scan_idx = day_data.index[scan_mask].values          # integer indices
    if len(scan_idx) == 0:
        raise ValueError("No bars to scan after the opening window.")

    highs  = day_data["high"].values
    lows   = day_data["low"].values
    closes = day_data["close"].values

    # ── 6. Trade scanning ───────────────────────────────────────────
    results = []

    for fp, ep in zip(fib_percents, entry_prices):
        long_wins = long_losses = long_inconclusive = 0
        short_wins = short_losses = short_inconclusive = 0

        for idx in scan_idx:
            bar_high = highs[idx]
            bar_low  = lows[idx]

            # Did this bar touch the entry price?
            if not (bar_low <= ep <= bar_high):
                continue

            # ATR from the PREVIOUS bar (strictly past data)
            if idx == 0 or np.isnan(atr_arr[idx - 1]):
                continue
            atr_val = atr_arr[idx - 1]
            risk   = atr_multiplier * atr_val
            reward = rr_ratio * risk

            if risk == 0:
                continue

            # --- LONG trade ---
            long_target = ep + reward
            long_stop   = ep - risk
            lw, ll, li  = _simulate_trade(
                scan_idx, idx, highs, lows, session_end,
                day_data["datetime"].values,
                target=long_target, stop=long_stop, direction="long",
            )
            long_wins += lw
            long_losses += ll
            long_inconclusive += li

            # --- SHORT trade ---
            short_target = ep - reward
            short_stop   = ep + risk
            sw, sl, si   = _simulate_trade(
                scan_idx, idx, highs, lows, session_end,
                day_data["datetime"].values,
                target=short_target, stop=short_stop, direction="short",
            )
            short_wins += sw
            short_losses += sl
            short_inconclusive += si

        # Win rate: exclude inconclusive from denominator
        long_denom  = long_wins + long_losses
        short_denom = short_wins + short_losses
        long_wr  = (long_wins / long_denom * 100) if long_denom else np.nan
        short_wr = (short_wins / short_denom * 100) if short_denom else np.nan

        # Optional EV
        long_ev  = _compute_ev(long_wr, rr_ratio)
        short_ev = _compute_ev(short_wr, rr_ratio)

        results.append({
            "fib_percent":        round(fp * 100, 1),
            "entry_price":        round(ep, 2),
            "long_trades":        int(long_wins + long_losses + long_inconclusive),
            "long_wins":          int(long_wins),
            "long_losses":        int(long_losses),
            "long_inconclusive":  int(long_inconclusive),
            "long_win_rate":      round(long_wr, 2) if not np.isnan(long_wr) else np.nan,
            "short_trades":       int(short_wins + short_losses + short_inconclusive),
            "short_wins":         int(short_wins),
            "short_losses":       int(short_losses),
            "short_inconclusive": int(short_inconclusive),
            "short_win_rate":     round(short_wr, 2) if not np.isnan(short_wr) else np.nan,
            "long_ev":            round(long_ev, 4) if not np.isnan(long_ev) else np.nan,
            "short_ev":           round(short_ev, 4) if not np.isnan(short_ev) else np.nan,
        })

    return pd.DataFrame(results)


# =====================================================================
#  TRADE SIMULATION (forward-walk from trigger bar)
# =====================================================================

def _simulate_trade(
    scan_idx: np.ndarray,
    trigger_idx: int,
    highs: np.ndarray,
    lows: np.ndarray,
    session_end: time,
    timestamps: np.ndarray,
    target: float,
    stop: float,
    direction: str,
) -> tuple[int, int, int]:
    """Walk forward from trigger bar and resolve the trade.

    Returns (wins, losses, inconclusive) — each is 0 or 1.
    """
    # Start from the NEXT bar after the trigger
    start_pos = np.searchsorted(scan_idx, trigger_idx, side="right")

    for pos in range(start_pos, len(scan_idx)):
        bar_i = scan_idx[pos]
        bar_ts = pd.Timestamp(timestamps[bar_i])

        # Past session end → classify as inconclusive (expired)
        if bar_ts.time() > session_end:
            return (0, 0, 1)

        bar_high = highs[bar_i]
        bar_low  = lows[bar_i]

        if direction == "long":
            hit_target = bar_high >= target
            hit_stop   = bar_low <= stop
        else:  # short
            hit_target = bar_low <= target
            hit_stop   = bar_high >= stop

        if hit_target and hit_stop:
            return (0, 0, 1)          # inconclusive — both hit same bar
        elif hit_target:
            return (1, 0, 0)
        elif hit_stop:
            return (0, 1, 0)

    # Ran out of bars → classify as inconclusive (session expired)
    return (0, 0, 1)


# =====================================================================
#  HELPERS
# =====================================================================

def _parse_time(t_str: str) -> time:
    """Parse 'HH:MM' → datetime.time."""
    parts = t_str.split(":")
    return time(int(parts[0]), int(parts[1]))


def _compute_ev(win_rate_pct, rr_ratio: float) -> float:
    """EV = rr_ratio × P(win) − 1 × P(loss).  Returns NaN if win_rate is NaN."""
    if np.isnan(win_rate_pct):
        return np.nan
    p_win = win_rate_pct / 100.0
    return rr_ratio * p_win - 1.0 * (1.0 - p_win)


def _print_report(date_str: str, df: pd.DataFrame) -> None:
    """Pretty-print the results table for a single day."""
    print(f"\n{'=' * 90}")
    print(f"  SMART RANGE REPORT — {date_str}")
    print(f"{'=' * 90}")
    print(f"{'Fib%':>6}  {'Entry':>10}  │ {'L_Trd':>5} {'L_Win':>5} {'L_WR%':>6} {'L_EV':>7}"
          f"  │ {'S_Trd':>5} {'S_Win':>5} {'S_WR%':>6} {'S_EV':>7}")
    print(f"{'─' * 90}")
    for _, r in df.iterrows():
        lwr = f"{r['long_win_rate']:.1f}" if not np.isnan(r['long_win_rate']) else "  –"
        swr = f"{r['short_win_rate']:.1f}" if not np.isnan(r['short_win_rate']) else "  –"
        lev = f"{r['long_ev']:.3f}" if not np.isnan(r['long_ev']) else "   –"
        sev = f"{r['short_ev']:.3f}" if not np.isnan(r['short_ev']) else "   –"
        print(f"{r['fib_percent']:>5.1f}%  {r['entry_price']:>10.2f}  │ "
              f"{int(r['long_trades']):>5}  {int(r['long_wins']):>4}  {lwr:>6}  {lev:>7}"
              f"  │ {int(r['short_trades']):>5}  {int(r['short_wins']):>4}  {swr:>6}  {sev:>7}")
    print(f"{'─' * 90}\n")


# =====================================================================
#  CLI ENTRY POINT
# =====================================================================

def main():
    """Run the smart-range analysis on one or more dates."""
    # Determine target dates
    if len(sys.argv) > 1:
        dates = sys.argv[1:]
    else:
        # Built-in sample days
        dates = ["2025-10-27", "2025-10-28"]

    print("Loading data …")
    raw = load_raw_csv()
    print(f"  {len(raw):,} bars loaded.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for date_str in dates:
        print(f"▸ Analysing {date_str} …")
        try:
            day = extract_single_day(raw, date_str)
            result = analyze_single_day(day)
            _print_report(date_str, result)

            # Save to CSV
            out_path = os.path.join(OUTPUT_DIR, f"smart_range_{date_str}.csv")
            result.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}\n")

        except Exception as e:
            print(f"  ⚠ Skipped {date_str}: {e}\n")


if __name__ == "__main__":
    main()
