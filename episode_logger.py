"""
episode_logger.py — Persist episode-level data for both phases with Agent.md schema.

Episode Schema per Agent.md:
- session_date, episode_id, state_type, direction, failure_count_at_start
- start_time, end_time, duration_minutes
- max_extension_points, max_extension_atr, max_extension_ib, max_extension_or
- max_retracement_points
- acceptance_achieved, time_in_acceptance_minutes
- terminal_outcome (A, A*, R, B)
- phase, avg_atr_during_episode, ib_range, or_range, reference_boundary
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _enrich_episodes(ep_list):
    """Convert episode list to DataFrame and add computed fields."""
    if not ep_list:
        return pd.DataFrame()
    
    ep_df = pd.DataFrame(ep_list)
    
    # Add episode_id
    ep_df = ep_df.reset_index(drop=True)
    ep_df["episode_id"] = ep_df.apply(
        lambda r: f"{r['session_date']}_{r.name:03d}", axis=1
    )
    
    # Discovery state label (D0, D1, D2, D3, D4)
    if "failure_count_at_start" in ep_df.columns:
        ep_df["discovery_state"] = "D" + ep_df["failure_count_at_start"].fillna(0).astype(int).astype(str)
    else:
        ep_df["discovery_state"] = "D0"
    
    # Ensure datetime columns
    if "start_time" in ep_df.columns:
        ep_df["start_time"] = pd.to_datetime(ep_df["start_time"])
    if "end_time" in ep_df.columns:
        ep_df["end_time"] = pd.to_datetime(ep_df["end_time"])
    
    return ep_df


def build_episode_log(df, profiles, or_df=None):
    """Build and persist the full episode log.
    
    This function imports state_engine dynamically to avoid circular imports.
    """
    from state_engine import encode_discovery_states
    from opening_range import merge_or_to_bars
    
    # Merge OR data if available
    if or_df is not None:
        df = merge_or_to_bars(df, or_df)
    
    pre_ib_episodes, post_ib_episodes = encode_discovery_states(df)
    
    pre_ib_df = _enrich_episodes(pre_ib_episodes)
    post_ib_df = _enrich_episodes(post_ib_episodes)
    
    return pre_ib_df, post_ib_df


def build_daily_summary(pre_ib_df, post_ib_df, profiles):
    """Build a daily summary comparing each day to prior session's value profile."""
    rows = []
    for _, p in profiles.iterrows():
        sd = p["session_date"]
        prior_vah = p.get("prior_rth_vah", p.get("prior_vah", np.nan))
        prior_val = p.get("prior_rth_val", p.get("prior_val", np.nan))
        rth_close = p.get("rth_close", np.nan)
        
        # Classify day relative to prior value
        if pd.notna(prior_vah) and pd.notna(prior_val) and pd.notna(rth_close):
            if rth_close > prior_vah:
                day_class = "above_value"
            elif rth_close < prior_val:
                day_class = "below_value"
            else:
                day_class = "within_value"
        else:
            day_class = "unknown"
        
        # Count episodes per phase
        pre_mask = pre_ib_df["session_date"] == sd if not pre_ib_df.empty else pd.Series(dtype=bool)
        post_mask = post_ib_df["session_date"] == sd if not post_ib_df.empty else pd.Series(dtype=bool)
        
        pre_sess = pre_ib_df[pre_mask] if pre_mask.any() else pd.DataFrame()
        post_sess = post_ib_df[post_mask] if post_mask.any() else pd.DataFrame()
        
        def _outcome_counts(ep):
            if ep.empty:
                return 0, 0, 0, 0
            vc = ep["terminal_outcome"].value_counts()
            return vc.get("A", 0), vc.get("A*", 0), vc.get("R", 0), vc.get("B", 0)
        
        pre_a, pre_a_star, pre_r, pre_b = _outcome_counts(pre_sess)
        post_a, post_a_star, post_r, post_b = _outcome_counts(post_sess)
        
        rows.append({
            "session_date": sd,
            "prior_rth_vah": prior_vah,
            "prior_rth_val": prior_val,
            "prior_rth_poc": p.get("prior_rth_poc", p.get("prior_poc", np.nan)),
            "overnight_vah": p.get("overnight_vah", np.nan),
            "overnight_val": p.get("overnight_val", np.nan),
            "ibh": p.get("ibh", np.nan),
            "ibl": p.get("ibl", np.nan),
            "ib_range": p.get("ib_range", np.nan),
            "rth_open": p.get("rth_open", np.nan),
            "rth_close": rth_close,
            "rth_high": p.get("rth_high", np.nan),
            "rth_low": p.get("rth_low", np.nan),
            "day_classification": day_class,
            "pre_ib_total": len(pre_sess),
            "pre_ib_A": pre_a,
            "pre_ib_A_star": pre_a_star,
            "pre_ib_R": pre_r,
            "pre_ib_B": pre_b,
            "post_ib_total": len(post_sess),
            "post_ib_A": post_a,
            "post_ib_A_star": post_a_star,
            "post_ib_R": post_r,
            "post_ib_B": post_b,
        })
    
    return pd.DataFrame(rows)


def save_episode_log(ep_df, output_path):
    """Save an episode log to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ep_df.to_csv(output_path, index=False)
    print(f"Saved episode log ({len(ep_df)} episodes) to {output_path}")


def save_episode_summary(ep_df, output_path):
    """Save episode summary statistics to CSV."""
    if ep_df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print(f"Saved empty episode summary to {output_path}")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Group by direction and discovery_state
    disc_eps = ep_df[ep_df["state_type"] == "discovery"]
    if disc_eps.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    
    summary = disc_eps.groupby(["direction", "discovery_state", "terminal_outcome"]).agg({
        "session_date": "count",
        "duration_minutes": ["mean", "std"],
        "max_extension_points": ["mean", "std"],
        "max_extension_atr": ["mean", "std"],
    }).reset_index()
    
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.rename(columns={"session_date_count": "count"})
    
    summary.to_csv(output_path, index=False)
    print(f"Saved episode summary to {output_path}")


if __name__ == "__main__":
    print("episode_logger.py - run via run_pipeline.py")
