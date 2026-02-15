"""
compare_breakout_rules.py — Single-session ORB entry rule comparison engine.

For a single trading day, compares multiple breakout entry rules at the
100% (OR_high) and 0% (OR_low) levels of the Opening Range.

This is an intraday event study — not a trading strategy or PnL backtest.

Usage:
    python compare_breakout_rules.py                       # runs built-in sample days
    python compare_breakout_rules.py 2025-10-27            # single date
    python compare_breakout_rules.py 2025-10-27 2025-10-28 # multiple dates
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import time
from enum import Enum

from config import CONFIG, SESSION_TIMES

# ── paths ────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "MNQ_5min_2021Jan_2026Jan.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── defaults (can be overridden via function args) ───────────────────
DEFAULT_OPENING_WINDOW = ("06:30", "06:45")   # 15 minutes OR
DEFAULT_END_TIME       = "13:00"              # RTH close
DEFAULT_ATR_PERIOD     = 3                    # bars for ATR lookback
DEFAULT_ATR_MULTIPLIER = 1.0                  # risk = mult × ATR
DEFAULT_RR_RATIO       = 2.0                  # reward : risk


# =====================================================================
#  DATA HELPERS (reusing from smart_range.py)
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
#  STATE MACHINE TYPES
# =====================================================================

class BreakState(Enum):
    """State for Option C (Break → Pullback → Reclaim)."""
    NONE = "none"
    BREAK = "break"
    PULLBACK = "pullback"


# =====================================================================
#  ENTRY RULE IMPLEMENTATIONS
# =====================================================================

def base_rule(
    scan_idx: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    or_high: float,
    or_low: float,
    direction: str,
) -> list[int]:
    """Base Case — Immediate Breakout Touch.
    
    Long: If high >= OR_high → enter at OR_high
    Short: If low <= OR_low → enter at OR_low
    
    Returns list of bar indices where entry occurred.
    """
    entries = []
    for idx in scan_idx:
        if direction == "long":
            if highs[idx] >= or_high:
                entries.append(idx)
        else:  # short
            if lows[idx] <= or_low:
                entries.append(idx)
    return entries


def wick_rejection_rule(
    scan_idx: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    or_high: float,
    or_low: float,
    direction: str,
) -> list[int]:
    """Option A — Wick Rejection.
    
    Long: Candle trades above OR_high, closes below OR_high, has upper wick
          → Enter at OR_high at close
    Short: Candle trades below OR_low, closes above OR_low, has lower wick
          → Enter at OR_low at close
    
    Returns list of bar indices where entry occurred.
    """
    entries = []
    for idx in scan_idx:
        if direction == "long":
            # Trades above, closes below, has upper wick
            if (highs[idx] > or_high and 
                closes[idx] < or_high and 
                highs[idx] > closes[idx]):
                entries.append(idx)
        else:  # short
            # Trades below, closes above, has lower wick
            if (lows[idx] < or_low and 
                closes[idx] > or_low and 
                lows[idx] < closes[idx]):
                entries.append(idx)
    return entries


def close_reclaim_rule(
    scan_idx: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    or_high: float,
    or_low: float,
    direction: str,
) -> list[int]:
    """Option B — Close Reclaim.
    
    Long: Candle trades below OR_high, closes above OR_high
          → Enter at close
    Short: Candle trades above OR_low, closes below OR_low
          → Enter at close
    
    Returns list of bar indices where entry occurred.
    """
    entries = []
    for idx in scan_idx:
        if direction == "long":
            # Trades below, closes above
            if lows[idx] < or_high and closes[idx] > or_high:
                entries.append(idx)
        else:  # short
            # Trades above, closes below
            if highs[idx] > or_low and closes[idx] < or_low:
                entries.append(idx)
    return entries


def break_pullback_reclaim_rule(
    scan_idx: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    or_high: float,
    or_low: float,
    direction: str,
) -> list[int]:
    """Option C — Break → Pullback → Reclaim Sequence.
    
    Long state machine:
    1. Break: high > OR_high
    2. Pullback: subsequent candle low < OR_high
    3. Reclaim: subsequent candle closes above OR_high → ENTER at close
    
    Short mirror logic:
    1. Break: low < OR_low
    2. Pullback: subsequent candle high > OR_low
    3. Reclaim: subsequent candle closes below OR_low → ENTER at close
    
    State machine resets after entry or full invalidation.
    
    Returns list of bar indices where entry occurred.
    """
    entries = []
    state = BreakState.NONE
    
    for idx in scan_idx:
        if direction == "long":
            if state == BreakState.NONE:
                # Looking for initial break
                if highs[idx] > or_high:
                    state = BreakState.BREAK
            elif state == BreakState.BREAK:
                # Looking for pullback
                if lows[idx] < or_high:
                    state = BreakState.PULLBACK
                # Otherwise stay in BREAK state
            elif state == BreakState.PULLBACK:
                # Looking for reclaim
                if closes[idx] > or_high:
                    entries.append(idx)
                    # Reset state machine after entry
                    state = BreakState.NONE
                # Otherwise continue waiting for reclaim
        
        else:  # short
            if state == BreakState.NONE:
                # Looking for initial break
                if lows[idx] < or_low:
                    state = BreakState.BREAK
            elif state == BreakState.BREAK:
                # Looking for pullback
                if highs[idx] > or_low:
                    state = BreakState.PULLBACK
                # Otherwise stay in BREAK state
            elif state == BreakState.PULLBACK:
                # Looking for reclaim
                if closes[idx] < or_low:
                    entries.append(idx)
                    # Reset state machine after entry
                    state = BreakState.NONE
    
    return entries


# =====================================================================
#  TRADE SIMULATION (reusing from smart_range.py with modifications)
# =====================================================================

def simulate_trade(
    scan_idx: np.ndarray,
    entry_idx: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    session_end: time,
    timestamps: np.ndarray,
    target: float,
    stop: float,
    direction: str,
) -> tuple[int, int, int]:
    """Walk forward from entry bar and resolve the trade.
    
    Returns (wins, losses, inconclusive) — each is 0 or 1.
    """
    # Start from the NEXT bar after entry
    start_pos = np.searchsorted(scan_idx, entry_idx, side="right")
    
    for pos in range(start_pos, len(scan_idx)):
        bar_i = scan_idx[pos]
        bar_ts = pd.Timestamp(timestamps[bar_i])
        
        # Past session end → classify as loss (as per spec)
        if bar_ts.time() > session_end:
            return (0, 1, 0)
        
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
    
    # Ran out of bars → classify as loss (as per spec)
    return (0, 1, 0)


# =====================================================================
#  CORE COMPARISON ENGINE
# =====================================================================

def compare_breakout_rules(
    day_data: pd.DataFrame,
    opening_window: tuple[str, str] = DEFAULT_OPENING_WINDOW,
    end_time: str = DEFAULT_END_TIME,
    atr_period: int = DEFAULT_ATR_PERIOD,
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
    rr_ratio: float = DEFAULT_RR_RATIO,
) -> pd.DataFrame:
    """Compare breakout entry rules for a single trading session.
    
    Parameters
    ----------
    day_data : DataFrame
        OHLCV bars for a single trading day (one calendar date).
    opening_window : tuple of ("HH:MM", "HH:MM")
        Start and end of the Opening Range window.
    end_time : str
        Last time at which a trade may still be open ("HH:MM").
    atr_period : int
        Number of bars for ATR computation.
    atr_multiplier : float
        Risk = atr_multiplier × ATR.
    rr_ratio : float
        Reward-to-risk ratio (default 2:1).
    
    Returns
    -------
    DataFrame with columns:
        rule, direction, trades, wins, losses, inconclusive, win_rate
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
        raise ValueError("Opening Range has zero width — cannot analyze breakout.")
    
    # ── 3. Compute rolling ATR (past-only) ──────────────────────────
    atr_arr = compute_rolling_atr(day_data, atr_period)
    
    # ── 4. Build bar arrays for the scan window ─────────────────────
    scan_mask = (
        (day_data["datetime"].dt.time > or_end) &
        (day_data["datetime"].dt.time <= session_end)
    )
    scan_idx = day_data.index[scan_mask].values
    if len(scan_idx) == 0:
        raise ValueError("No bars to scan after the opening window.")
    
    highs  = day_data["high"].values
    lows   = day_data["low"].values
    closes = day_data["close"].values
    timestamps = day_data["datetime"].values
    
    # ── 5. Define rules to compare ──────────────────────────────────
    rules = {
        "base": base_rule,
        "wick_rejection": wick_rejection_rule,
        "close_reclaim": close_reclaim_rule,
        "break_pullback_reclaim": break_pullback_reclaim_rule,
    }
    
    results = []
    
    # ── 6. Run each rule for both directions ────────────────────────
    for rule_name, rule_func in rules.items():
        for direction in ["long", "short"]:
            # Get entry indices from rule
            if rule_name == "base":
                entry_indices = rule_func(
                    scan_idx, highs, lows, or_high, or_low, direction
                )
            else:
                entry_indices = rule_func(
                    scan_idx, highs, lows, closes, or_high, or_low, direction
                )
            
            wins = losses = inconclusive = 0
            
            # Simulate each entry
            for entry_idx in entry_indices:
                # ATR from PREVIOUS bar (strictly past data)
                if entry_idx == 0 or np.isnan(atr_arr[entry_idx - 1]):
                    continue
                
                atr_val = atr_arr[entry_idx - 1]
                risk    = atr_multiplier * atr_val
                reward  = rr_ratio * risk
                
                if risk == 0:
                    continue
                
                # Determine entry price based on rule
                if rule_name in ["base", "wick_rejection"]:
                    entry_price = or_high if direction == "long" else or_low
                else:  # close_reclaim and break_pullback_reclaim
                    entry_price = closes[entry_idx]
                
                # Set targets and stops
                if direction == "long":
                    target = entry_price + reward
                    stop   = entry_price - risk
                else:  # short
                    target = entry_price - reward
                    stop   = entry_price + risk
                
                # Simulate trade
                w, l, i = simulate_trade(
                    scan_idx, entry_idx, entry_price,
                    highs, lows, session_end, timestamps,
                    target, stop, direction
                )
                wins += w
                losses += l
                inconclusive += i
            
            # Calculate win rate
            total_trades = wins + losses + inconclusive
            denom = wins + losses
            win_rate = (wins / denom * 100) if denom > 0 else np.nan
            
            results.append({
                "rule": rule_name,
                "direction": direction,
                "trades": int(total_trades),
                "wins": int(wins),
                "losses": int(losses),
                "inconclusive": int(inconclusive),
                "win_rate": round(win_rate, 2) if not np.isnan(win_rate) else np.nan,
            })
    
    return pd.DataFrame(results)


# =====================================================================
#  HELPERS
# =====================================================================

def _parse_time(t_str: str) -> time:
    """Parse 'HH:MM' → datetime.time."""
    parts = t_str.split(":")
    return time(int(parts[0]), int(parts[1]))


def _print_report(date_str: str, df: pd.DataFrame) -> None:
    """Pretty-print the comparison results for a single day."""
    print(f"\n{'=' * 80}")
    print(f"  BREAKOUT RULES COMPARISON — {date_str}")
    print(f"{'=' * 80}")
    print(f"{'Rule':<25} {'Dir':<6} │ {'Trades':>6} {'Wins':>6} {'Losses':>6} {'WR%':>8}")
    print(f"{'─' * 80}")
    
    for _, r in df.iterrows():
        wr_str = f"{r['win_rate']:.1f}" if not np.isnan(r['win_rate']) else "  –"
        print(f"{r['rule']:<25} {r['direction']:<6} │ "
              f"{int(r['trades']):>6}  {int(r['wins']):>6}  {int(r['losses']):>6}  {wr_str:>8}")
    print(f"{'─' * 80}\n")


# =====================================================================
#  CLI ENTRY POINT
# =====================================================================

def main():
    """Run the breakout rules comparison on one or more dates."""
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
            result = compare_breakout_rules(day)
            _print_report(date_str, result)
            
            # Save to CSV
            out_path = os.path.join(OUTPUT_DIR, f"breakout_comparison_{date_str}.csv")
            result.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}\n")
        
        except Exception as e:
            print(f"  ⚠ Skipped {date_str}: {e}\n")


if __name__ == "__main__":
    main()
