"""
opening_range.py — Compute Opening Range (OR) metrics and classifications.

Per Agent.md:
- Opening Range: First N minutes of RTH (default: 15 minutes from 6:30am)
- ORH = Highest high during window
- ORL = Lowest low during window
- OR_range = ORH - ORL

OR Position Classifications:
- above_vah: OR formed above value area high
- within_value: OR within VAH-VAL range
- below_val: OR formed below value area low
- at_vah: OR straddling VAH
- at_val: OR straddling VAL
"""

import pandas as pd
import numpy as np
import os
from datetime import time, timedelta

from config import CONFIG, SESSION_TIMES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def compute_opening_range(df, profiles, opening_window_minutes=None):
    """Compute Opening Range for each session.
    
    Args:
        df: Bar-level DataFrame with datetime, high, low, session_date
        profiles: Session profiles with prior_rth_vah/val, overnight_vah/val
        opening_window_minutes: Duration of OR window (default from config)
    
    Returns:
        DataFrame with OR metrics per session
    """
    if opening_window_minutes is None:
        opening_window_minutes = CONFIG['opening_window_minutes']
    
    or_data = []
    session_dates = sorted(df["session_date"].unique())
    
    rth_start = time(*SESSION_TIMES['rth_start'])
    or_end_delta = timedelta(minutes=opening_window_minutes)
    
    for sd in session_dates:
        session_df = df[df["session_date"] == sd]
        if session_df.empty:
            continue
        
        # Get session profile
        profile = profiles[profiles["session_date"] == sd]
        if profile.empty:
            continue
        profile = profile.iloc[0]
        
        # Get bars in OR window (first N minutes from 6:30am)
        or_end_time = (pd.Timestamp.combine(sd, rth_start) + or_end_delta).time()
        or_mask = (
            (session_df["datetime"].dt.time >= rth_start) &
            (session_df["datetime"].dt.time < or_end_time)
        )
        or_bars = session_df[or_mask]
        
        if len(or_bars) == 0:
            continue
        
        # Compute OR metrics
        orh = or_bars["high"].max()
        orl = or_bars["low"].min()
        or_range = orh - orl
        
        # Get reference values
        prior_rth_vah = profile.get("prior_rth_vah", np.nan)
        prior_rth_val = profile.get("prior_rth_val", np.nan)
        prior_rth_poc = profile.get("prior_rth_poc", np.nan)
        overnight_vah = profile.get("overnight_vah", np.nan)
        overnight_val = profile.get("overnight_val", np.nan)
        overnight_poc = profile.get("overnight_poc", np.nan)
        ibh = profile.get("ibh", np.nan)
        ibl = profile.get("ibl", np.nan)
        ib_range = profile.get("ib_range", np.nan)
        
        # Classify OR vs Prior RTH Value
        or_vs_prior_rth = classify_or_position(orh, orl, prior_rth_vah, prior_rth_val)
        
        # Classify OR vs Overnight Value
        or_vs_overnight = classify_or_position(orh, orl, overnight_vah, overnight_val)
        
        # Check if price retested prior RTH value area during pre-IB
        pre_ib_bars = session_df[session_df["phase"] == "pre_ib"]
        retest_prior_rth = False
        if len(pre_ib_bars) > 0 and pd.notna(prior_rth_vah) and pd.notna(prior_rth_val):
            pre_ib_low = pre_ib_bars["low"].min()
            pre_ib_high = pre_ib_bars["high"].max()
            if pre_ib_low <= prior_rth_vah and pre_ib_high >= prior_rth_val:
                retest_prior_rth = True
        
        # Price at IB formation (7:30am) relative to prior RTH
        ib_bar = session_df[session_df["datetime"].dt.time == time(7, 30)]
        if len(ib_bar) > 0:
            price_at_ib = ib_bar["close"].iloc[0]
            if pd.notna(prior_rth_vah) and pd.notna(prior_rth_val):
                if price_at_ib > prior_rth_vah:
                    price_at_ib_vs_prior_rth = "above_vah"
                elif price_at_ib < prior_rth_val:
                    price_at_ib_vs_prior_rth = "below_val"
                else:
                    price_at_ib_vs_prior_rth = "within_value"
            else:
                price_at_ib_vs_prior_rth = "unknown"
        else:
            price_at_ib_vs_prior_rth = "unknown"
        
        or_data.append({
            "session_date": sd,
            "opening_window_minutes": opening_window_minutes,
            "orh": orh,
            "orl": orl,
            "or_range": or_range,
            "prior_rth_vah": prior_rth_vah,
            "prior_rth_val": prior_rth_val,
            "prior_rth_poc": prior_rth_poc,
            "overnight_vah": overnight_vah,
            "overnight_val": overnight_val,
            "overnight_poc": overnight_poc,
            "or_vs_prior_rth": or_vs_prior_rth,
            "or_vs_overnight": or_vs_overnight,
            "retest_prior_rth_during_preib": retest_prior_rth,
            "price_at_ib_vs_prior_rth": price_at_ib_vs_prior_rth,
            "ibh": ibh,
            "ibl": ibl,
            "ib_range": ib_range,
        })
    
    or_df = pd.DataFrame(or_data)
    print(f"Computed Opening Range for {len(or_df)} sessions")
    return or_df


def classify_or_position(orh, orl, vah, val):
    """Classify OR position relative to a value area."""
    if pd.isna(vah) or pd.isna(val):
        return "unknown"
    
    if orl > vah:
        return "above_vah"
    elif orh < val:
        return "below_val"
    elif orl >= val and orh <= vah:
        return "within_value"
    elif orl < vah <= orh:
        return "at_vah"
    elif orl <= val < orh:
        return "at_val"
    else:
        return "straddling"


def merge_or_to_bars(df, or_df):
    """Merge OR data onto bar-level data."""
    merge_cols = ["session_date", "orh", "orl", "or_range"]
    available_cols = [c for c in merge_cols if c in or_df.columns]
    df = df.merge(or_df[available_cols], on="session_date", how="left")
    return df


def save_opening_range(or_df, output_path=None):
    """Save Opening Range analysis to CSV."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "opening_range_analysis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    or_df.to_csv(output_path, index=False)
    print(f"Saved Opening Range analysis to {output_path}")


if __name__ == "__main__":
    from load_data import load_and_validate
    from session_profile import compute_session_profiles
    
    df = load_and_validate()
    profiles = compute_session_profiles(df)
    or_df = compute_opening_range(df, profiles)
    save_opening_range(or_df)
