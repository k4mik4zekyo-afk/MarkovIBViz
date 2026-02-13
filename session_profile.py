"""
session_profile.py — Compute session profiles with SEPARATE Prior RTH and Overnight.

Per Agent.md spec:
- Session: 6:30am PT to 6:25am PT next day (~24hr window)
- Prior RTH: 6:30am-1:00pm PT of the PREVIOUS session
- Overnight: 1:00pm PT previous to 6:25am PT current (includes Globex, Asia, London)
- Initial Balance: 6:30am-7:25am PT (pre_ib phase)
- Post-IB: 7:30am-1:00pm PT

Weekend/Holiday Handling:
- For Sundays (session starts Sunday evening), use Friday's RTH as prior
- Holiday adjustments map specific dates to their proper prior RTH date
- Contract roll dates are also adjusted for low volume

Compute VAH/VAL/POC separately for Prior RTH and Overnight using volume profiles.
"""

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

from config import CONFIG

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "session_profiles.csv")

# Tick size for MNQ (Micro E-mini Nasdaq) is 0.25
TICK_SIZE = 0.25
VALUE_AREA_PCT = 0.70

# Holiday and special date adjustments
# Maps session_date -> prior RTH date to use (skipping holidays/low volume days)
HOLIDAY_ADJUSTMENTS = {
    # 2025 Holidays
    date(2025, 12, 1): date(2025, 11, 26),   # Thanksgiving (Nov 27) -> skip Thursday and Friday
    date(2025, 9, 2): date(2025, 8, 29),     # Labor Day (Sept 1) -> use Friday Aug 29
    date(2025, 7, 7): date(2025, 7, 2),      # July 4th holiday meant July 3rd was short day
    date(2025, 6, 20): date(2025, 6, 18),    # June 19 (Juneteenth) -> use Wednesday June 18
    date(2025, 5, 27): date(2025, 5, 23),    # May 26 (Memorial Day) -> use Friday May 23
    date(2025, 2, 18): date(2025, 2, 14),    # Presidents Day (Feb 17) -> use Friday Feb 14
    date(2025, 1, 21): date(2025, 1, 17),    # MLK Day (Jan 20) -> use Friday Jan 17
    # Contract switches (low volume days)
    date(2025, 12, 16): date(2025, 12, 12),  # Q4 contract switch on 12/15
    date(2025, 9, 16): date(2025, 9, 12),    # Q3 contract switch on 9/15
    date(2025, 3, 18): date(2025, 3, 14),    # Q1 contract switch on 3/17
    
    # 2024 Holidays
    date(2024, 12, 2): date(2024, 11, 27),   # Thanksgiving (Nov 28)
    date(2024, 9, 3): date(2024, 8, 30),     # Labor Day (Sept 2)
    date(2024, 7, 8): date(2024, 7, 3),      # July 4th
    date(2024, 6, 20): date(2024, 6, 18),    # Juneteenth (June 19)
    date(2024, 5, 28): date(2024, 5, 24),    # Memorial Day (May 27)
    date(2024, 2, 20): date(2024, 2, 16),    # Presidents Day (Feb 19)
    date(2024, 1, 16): date(2024, 1, 12),    # MLK Day (Jan 15)
    # Contract switches 2024
    date(2024, 12, 16): date(2024, 12, 12),
    date(2024, 9, 16): date(2024, 9, 12),
    date(2024, 3, 18): date(2024, 3, 14),
    
    # 2023 Holidays
    date(2023, 11, 27): date(2023, 11, 22),  # Thanksgiving (Nov 23)
    date(2023, 9, 5): date(2023, 9, 1),      # Labor Day (Sept 4)
    date(2023, 7, 5): date(2023, 6, 30),     # July 4th
    date(2023, 6, 20): date(2023, 6, 16),    # Juneteenth (June 19)
    date(2023, 5, 30): date(2023, 5, 26),    # Memorial Day (May 29)
    date(2023, 2, 21): date(2023, 2, 17),    # Presidents Day (Feb 20)
    date(2023, 1, 17): date(2023, 1, 13),    # MLK Day (Jan 16)
    # Contract switches 2023
    date(2023, 12, 18): date(2023, 12, 14),
    date(2023, 9, 18): date(2023, 9, 14),
    date(2023, 3, 20): date(2023, 3, 16),
}


def build_volume_profile(bars_df, tick_size=TICK_SIZE):
    """Build a volume-at-price histogram for given bars.

    For each bar, distribute volume evenly across the price range [low, high]
    at tick_size increments.

    Returns:
        pd.Series indexed by price level, values are volume.
    """
    if bars_df.empty:
        return pd.Series(dtype=float)
    
    profile = {}
    for _, bar in bars_df.iterrows():
        low = bar["low"]
        high = bar["high"]
        vol = bar.get("volume", 1)  # Default volume if missing
        
        prices = np.arange(
            np.floor(low / tick_size) * tick_size,
            np.ceil(high / tick_size) * tick_size + tick_size / 2,
            tick_size,
        )
        prices = prices[(prices >= low) & (prices <= high)]
        if len(prices) == 0:
            continue
        vol_per_level = vol / len(prices)
        for p in prices:
            p_rounded = round(p, 4)
            profile[p_rounded] = profile.get(p_rounded, 0.0) + vol_per_level

    return pd.Series(profile).sort_index()


def compute_value_area(volume_profile, value_area_pct=VALUE_AREA_PCT):
    """Compute VAH, VAL, POC from a volume profile.

    Standard algorithm:
    1. Find POC (price level with highest volume).
    2. Starting from POC, expand one level at a time in the direction
       with more volume until value_area_pct of total volume is enclosed.

    Returns:
        dict with keys: poc, vah, val
    """
    if volume_profile.empty:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan}

    total_volume = volume_profile.sum()
    poc_price = volume_profile.idxmax()

    prices = volume_profile.index.values
    poc_idx = np.searchsorted(prices, poc_price)

    low_idx = poc_idx
    high_idx = poc_idx
    captured_volume = volume_profile.iloc[poc_idx] if len(volume_profile) > 0 else 0

    target_volume = total_volume * value_area_pct

    while captured_volume < target_volume:
        can_go_low = low_idx > 0
        can_go_high = high_idx < len(prices) - 1

        if not can_go_low and not can_go_high:
            break

        low_vol = volume_profile.iloc[low_idx - 1] if can_go_low else -1
        high_vol = volume_profile.iloc[high_idx + 1] if can_go_high else -1

        if low_vol >= high_vol:
            low_idx -= 1
            captured_volume += volume_profile.iloc[low_idx]
        else:
            high_idx += 1
            captured_volume += volume_profile.iloc[high_idx]

    val = prices[low_idx]
    vah = prices[high_idx]

    return {"poc": poc_price, "vah": vah, "val": val}


def get_prior_session_date(session_date, session_dates):
    """Determine the correct prior RTH session date, handling weekends and holidays.
    
    Logic:
    1. Check HOLIDAY_ADJUSTMENTS for explicit overrides
    2. For Sundays, use Friday (skip Saturday which has no RTH)
    3. Otherwise use the previous session date in the list
    
    Args:
        session_date: The current session date (date object or convertible)
        session_dates: Sorted list of all session dates
    
    Returns:
        The prior session date to use for RTH, or None if no prior exists
    """
    # Convert to date object if needed
    if isinstance(session_date, pd.Timestamp):
        sd = session_date.date()
    elif isinstance(session_date, str):
        sd = pd.to_datetime(session_date).date()
    else:
        sd = session_date
    
    # Check holiday adjustments first
    if sd in HOLIDAY_ADJUSTMENTS:
        target_date = HOLIDAY_ADJUSTMENTS[sd]
        # Find this date in session_dates
        for sess_date in session_dates:
            if isinstance(sess_date, pd.Timestamp):
                sess_d = sess_date.date()
            elif isinstance(sess_date, str):
                sess_d = pd.to_datetime(sess_date).date()
            else:
                sess_d = sess_date
            if sess_d == target_date:
                return sess_date
        # If exact date not found, find closest prior
        return None
    
    # Find current index in session_dates
    idx = None
    for i, sess_date in enumerate(session_dates):
        if isinstance(sess_date, pd.Timestamp):
            sess_d = sess_date.date()
        elif isinstance(sess_date, str):
            sess_d = pd.to_datetime(sess_date).date()
        else:
            sess_d = sess_date
        if sess_d == sd:
            idx = i
            break
    
    if idx is None or idx == 0:
        return None
    
    # Check if current session is a Sunday (weekday 6)
    if sd.weekday() == 6:  # Sunday
        # Look for Friday (2 days back from Sunday)
        friday_date = sd - timedelta(days=2)
        for sess_date in session_dates:
            if isinstance(sess_date, pd.Timestamp):
                sess_d = sess_date.date()
            elif isinstance(sess_date, str):
                sess_d = pd.to_datetime(sess_date).date()
            else:
                sess_d = sess_date
            if sess_d == friday_date:
                return sess_date
        # Friday not found, fall back to previous in list
    
    # Default: use previous session in the sorted list
    return session_dates[idx - 1]


def get_prior_rth_bars(df, session_date, session_dates):
    """Get RTH bars from the appropriate PRIOR session (6:30am-1:00pm PT).
    
    Per Agent.md: Prior RTH = pre_ib + post_ib phases of PREVIOUS session.
    Handles weekends (use Friday for Sunday) and holidays via HOLIDAY_ADJUSTMENTS.
    """
    prior_date = get_prior_session_date(session_date, session_dates)
    
    if prior_date is None:
        return pd.DataFrame()
    
    prior_bars = df[
        (df["session_date"] == prior_date) & 
        (df["phase"].isin(["pre_ib", "post_ib"]))
    ]
    return prior_bars


def get_overnight_bars(df, session_date, session_dates):
    """Get overnight bars between previous RTH close and current RTH open.
    
    Per Agent.md: Overnight = 1:00pm PT previous to 6:25am PT current.
    This captures Globex afternoon, Asia, and London sessions.
    
    Using the 'overnight' phase assigned in load_data.py.
    """
    overnight_bars = df[
        (df["session_date"] == session_date) & 
        (df["phase"] == "overnight")
    ]
    return overnight_bars


def compute_session_profiles(df):
    """Compute session profiles with SEPARATE Prior RTH and Overnight value areas.

    Per Agent.md spec:
    - prior_rth_vah/val/poc: From previous RTH session (6:30am-1:00pm PT yesterday)
    - overnight_vah/val/poc: From overnight session (1:00pm PT yesterday to 6:25am PT today)
    - ibh/ibl/ib_range: Initial Balance (6:30am-7:25am PT, first hour)

    Returns:
        pd.DataFrame with one row per session.
    """
    sessions = []
    session_dates = sorted(df["session_date"].unique())

    for sd in session_dates:
        session_df = df[df["session_date"] == sd]
        if len(session_df) == 0:
            continue

        # Full-session value area (legacy, still useful)
        full_vol_profile = build_volume_profile(session_df)
        full_va = compute_value_area(full_vol_profile)

        # Prior RTH value area (PREVIOUS session's RTH)
        prior_rth_bars = get_prior_rth_bars(df, sd, session_dates)
        prior_rth_profile = build_volume_profile(prior_rth_bars)
        prior_rth_va = compute_value_area(prior_rth_profile)
        
        # Prior RTH high/low
        prior_rth_high = prior_rth_bars["high"].max() if not prior_rth_bars.empty else np.nan
        prior_rth_low = prior_rth_bars["low"].min() if not prior_rth_bars.empty else np.nan

        # Overnight value area (Globex/Asia/London between sessions)
        overnight_bars = get_overnight_bars(df, sd, session_dates)
        overnight_profile = build_volume_profile(overnight_bars)
        overnight_va = compute_value_area(overnight_profile)
        
        # Overnight high/low
        overnight_high = overnight_bars["high"].max() if not overnight_bars.empty else np.nan
        overnight_low = overnight_bars["low"].min() if not overnight_bars.empty else np.nan

        # Initial Balance: bars in the pre_ib phase (6:30am - 7:25am)
        ib_bars = session_df[session_df["phase"] == "pre_ib"]
        if len(ib_bars) > 0:
            ibh = ib_bars["high"].max()
            ibl = ib_bars["low"].min()
            ib_range = ibh - ibl
        else:
            ibh = np.nan
            ibl = np.nan
            ib_range = np.nan

        # RTH bars (pre_ib + post_ib)
        rth_bars = session_df[session_df["phase"].isin(["pre_ib", "post_ib"])]
        if len(rth_bars) > 0:
            rth_open = rth_bars["open"].iloc[0]
            rth_close = rth_bars["close"].iloc[-1]
            rth_high = rth_bars["high"].max()
            rth_low = rth_bars["low"].min()
        else:
            rth_open = np.nan
            rth_close = np.nan
            rth_high = np.nan
            rth_low = np.nan

        sessions.append({
            "session_date": sd,
            # Full session stats (legacy)
            "session_high": session_df["high"].max(),
            "session_low": session_df["low"].min(),
            "poc": full_va["poc"],
            "vah": full_va["vah"],
            "val": full_va["val"],
            "total_volume": session_df["volume"].sum(),
            # Prior RTH value area (SEPARATE per Agent.md)
            "prior_rth_vah": prior_rth_va["vah"],
            "prior_rth_val": prior_rth_va["val"],
            "prior_rth_poc": prior_rth_va["poc"],
            "prior_rth_high": prior_rth_high,
            "prior_rth_low": prior_rth_low,
            # Overnight value area (SEPARATE per Agent.md)
            "overnight_vah": overnight_va["vah"],
            "overnight_val": overnight_va["val"],
            "overnight_poc": overnight_va["poc"],
            "overnight_high": overnight_high,
            "overnight_low": overnight_low,
            # Initial Balance
            "ibh": ibh,
            "ibl": ibl,
            "ib_range": ib_range,
            # RTH stats
            "rth_open": rth_open,
            "rth_close": rth_close,
            "rth_high": rth_high,
            "rth_low": rth_low,
        })

    profiles = pd.DataFrame(sessions)

    # Normalized volume
    mean_vol = profiles["total_volume"].mean()
    profiles["normalized_volume"] = profiles["total_volume"] / mean_vol if mean_vol > 0 else 0.0

    # Legacy prior-session columns (shifted by 1 for full-session)
    profiles["prior_vah"] = profiles["vah"].shift(1)
    profiles["prior_val"] = profiles["val"].shift(1)
    profiles["prior_poc"] = profiles["poc"].shift(1)
    profiles["prior_high"] = profiles["session_high"].shift(1)
    profiles["prior_low"] = profiles["session_low"].shift(1)
    profiles["prior_normalized_volume"] = profiles["normalized_volume"].shift(1)

    print(f"Computed profiles for {len(profiles)} sessions (with separate Prior RTH & Overnight)")
    return profiles


def merge_profiles_to_bars(df, profiles):
    """Merge session profile data onto bar-level data.

    Each bar gets:
    - Prior RTH VAH/VAL/POC (for pre-IB discovery)
    - Overnight VAH/VAL/POC (for context)
    - IBH/IBL (for post-IB discovery)
    - Legacy prior_vah/val/poc (backward compatibility)
    """
    merge_cols = [
        "session_date",
        # Prior RTH (SEPARATE)
        "prior_rth_vah", "prior_rth_val", "prior_rth_poc", 
        "prior_rth_high", "prior_rth_low",
        # Overnight (SEPARATE)
        "overnight_vah", "overnight_val", "overnight_poc",
        "overnight_high", "overnight_low",
        # Legacy (full-session shifted)
        "prior_vah", "prior_val", "prior_poc", "prior_high", "prior_low",
        "prior_normalized_volume",
        # IB
        "ibh", "ibl", "ib_range",
        # RTH
        "rth_open", "rth_close",
    ]
    available_cols = [c for c in merge_cols if c in profiles.columns]
    df = df.merge(profiles[available_cols], on="session_date", how="left")
    return df


def save_profiles(profiles, output_path=None):
    """Save session profiles to CSV."""
    if output_path is None:
        output_path = OUTPUT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profiles.to_csv(output_path, index=False)
    print(f"Saved session profiles to {output_path}")


def save_prior_rth_profiles(profiles, output_path=None):
    """Save Prior RTH specific profile columns."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "session_profiles_prior_rth.csv")
    
    cols = ["session_date", "prior_rth_vah", "prior_rth_val", "prior_rth_poc", 
            "prior_rth_high", "prior_rth_low"]
    cols = [c for c in cols if c in profiles.columns]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profiles[cols].to_csv(output_path, index=False)
    print(f"Saved Prior RTH profiles to {output_path}")


def save_overnight_profiles(profiles, output_path=None):
    """Save Overnight specific profile columns."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "session_profiles_overnight.csv")
    
    cols = ["session_date", "overnight_vah", "overnight_val", "overnight_poc",
            "overnight_high", "overnight_low"]
    cols = [c for c in cols if c in profiles.columns]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profiles[cols].to_csv(output_path, index=False)
    print(f"Saved Overnight profiles to {output_path}")


if __name__ == "__main__":
    from load_data import load_and_validate
    
    df = load_and_validate()
    profiles = compute_session_profiles(df)
    save_profiles(profiles)
    save_prior_rth_profiles(profiles)
    save_overnight_profiles(profiles)
    
    df_merged = merge_profiles_to_bars(df, profiles)
    print(f"Bars with prior RTH profiles: {df_merged['prior_rth_vah'].notna().sum()} / {len(df_merged)}")
    print(f"Bars with overnight profiles: {df_merged['overnight_vah'].notna().sum()} / {len(df_merged)}")
    print(f"Bars with IB data: {df_merged['ibh'].notna().sum()} / {len(df_merged)}")
