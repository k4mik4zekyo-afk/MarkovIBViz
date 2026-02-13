"""
session_logger.py — Compute session-level outcome classifications.

Per Agent.md Session Classifications:

Directional:
- net_discovery_up: Session ended with accepted upside discovery
- net_discovery_down: Session ended with accepted downside discovery
- two_sided_discovery: Both upside and downside discovery accepted

Failed Directional:
- failed_up_balance: Any upside rejection attempts, remained in value area
- failed_down_balance: Any downside rejection attempts, remained in value area
- failed_two_sided_balance: Rejected both directions, remained in value

Rotation:
- failed_discovery_up: Rejected downside attempts, then rotated to upside acceptance
- failed_discovery_down: Rejected upside attempts, then rotated to downside acceptance

Pure Balance:
- balanced: No acceptances either direction, stayed in value, minimal rejections
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def compute_session_outcomes(pre_ib_episodes, post_ib_episodes, profiles):
    """Compute session-level outcome classifications.
    
    Args:
        pre_ib_episodes: DataFrame of pre-IB episodes
        post_ib_episodes: DataFrame of post-IB episodes
        profiles: Session profiles DataFrame
    
    Returns:
        DataFrame with session-level outcomes
    """
    session_data = []
    
    for _, profile in profiles.iterrows():
        sd = profile["session_date"]
        
        # Get episodes for this session
        pre_eps = pre_ib_episodes[pre_ib_episodes["session_date"] == sd] if not pre_ib_episodes.empty else pd.DataFrame()
        post_eps = post_ib_episodes[post_ib_episodes["session_date"] == sd] if not post_ib_episodes.empty else pd.DataFrame()
        all_eps = pd.concat([pre_eps, post_eps], ignore_index=True) if not pre_eps.empty or not post_eps.empty else pd.DataFrame()
        
        # Filter to discovery episodes only
        if not all_eps.empty:
            disc_eps = all_eps[all_eps["state_type"] == "discovery"]
        else:
            disc_eps = pd.DataFrame()
        
        # Count attempts and outcomes by direction
        up_attempts = len(disc_eps[disc_eps["direction"] == "up"]) if not disc_eps.empty else 0
        down_attempts = len(disc_eps[disc_eps["direction"] == "down"]) if not disc_eps.empty else 0
        
        up_acceptances = len(disc_eps[(disc_eps["direction"] == "up") & (disc_eps["terminal_outcome"].isin(["A", "A*"]))]) if not disc_eps.empty else 0
        down_acceptances = len(disc_eps[(disc_eps["direction"] == "down") & (disc_eps["terminal_outcome"].isin(["A", "A*"]))]) if not disc_eps.empty else 0
        
        up_rejections = len(disc_eps[(disc_eps["direction"] == "up") & (disc_eps["terminal_outcome"] == "R")]) if not disc_eps.empty else 0
        down_rejections = len(disc_eps[(disc_eps["direction"] == "down") & (disc_eps["terminal_outcome"] == "R")]) if not disc_eps.empty else 0
        
        # Max extensions
        up_eps = disc_eps[disc_eps["direction"] == "up"] if not disc_eps.empty else pd.DataFrame()
        down_eps = disc_eps[disc_eps["direction"] == "down"] if not disc_eps.empty else pd.DataFrame()
        
        max_ext_up = up_eps["max_extension"].max() if not up_eps.empty and "max_extension" in up_eps.columns else 0
        max_ext_down = down_eps["max_extension"].max() if not down_eps.empty and "max_extension" in down_eps.columns else 0
        
        # Time in states (approximation using episode durations)
        balance_eps = all_eps[all_eps["state_type"] == "balance"] if not all_eps.empty and "state_type" in all_eps.columns else pd.DataFrame()
        time_in_balance = balance_eps["duration_minutes"].sum() if not balance_eps.empty and "duration_minutes" in balance_eps.columns else 0
        time_in_discovery = disc_eps["duration_minutes"].sum() if not disc_eps.empty and "duration_minutes" in disc_eps.columns else 0
        
        # Final failure counts
        if not disc_eps.empty and "failure_count_at_start" in disc_eps.columns:
            final_up = disc_eps[disc_eps["direction"] == "up"]["failure_count_at_start"].max() if up_attempts > 0 else 0
            final_down = disc_eps[disc_eps["direction"] == "down"]["failure_count_at_start"].max() if down_attempts > 0 else 0
            # Add 1 for each rejection
            final_up = (final_up if pd.notna(final_up) else 0) + up_rejections
            final_down = (final_down if pd.notna(final_down) else 0) + down_rejections
        else:
            final_up = up_rejections
            final_down = down_rejections
        
        # Classify session
        session_class = classify_session(
            up_acceptances, down_acceptances,
            up_rejections, down_rejections,
            up_attempts, down_attempts
        )
        
        # Get closing price and reference levels
        closing_price = profile.get("rth_close", np.nan)
        prior_rth_vah = profile.get("prior_rth_vah", np.nan)
        prior_rth_val = profile.get("prior_rth_val", np.nan)
        overnight_vah = profile.get("overnight_vah", np.nan)
        overnight_val = profile.get("overnight_val", np.nan)
        ibh = profile.get("ibh", np.nan)
        ibl = profile.get("ibl", np.nan)
        
        # Classify close relative to references
        close_vs_prior_rth = classify_close(closing_price, prior_rth_vah, prior_rth_val)
        close_vs_overnight = classify_close(closing_price, overnight_vah, overnight_val)
        close_vs_ib = classify_close(closing_price, ibh, ibl)
        
        session_data.append({
            "session_date": sd,
            "session_classification": session_class,
            "discovery_up_attempts": up_attempts,
            "discovery_down_attempts": down_attempts,
            "discovery_up_acceptances": up_acceptances,
            "discovery_down_acceptances": down_acceptances,
            "discovery_up_rejections": up_rejections,
            "discovery_down_rejections": down_rejections,
            "max_extension_up_points": max_ext_up,
            "max_extension_down_points": max_ext_down,
            "time_in_balance_minutes": time_in_balance,
            "time_in_discovery_minutes": time_in_discovery,
            "final_failure_count_up": min(final_up, 4),
            "final_failure_count_down": min(final_down, 4),
            "closing_price": closing_price,
            "prior_rth_vah": prior_rth_vah,
            "prior_rth_val": prior_rth_val,
            "overnight_vah": overnight_vah,
            "overnight_val": overnight_val,
            "ibh": ibh,
            "ibl": ibl,
            "close_vs_prior_rth": close_vs_prior_rth,
            "close_vs_overnight": close_vs_overnight,
            "close_vs_ib": close_vs_ib,
        })
    
    session_df = pd.DataFrame(session_data)
    print(f"Computed session outcomes for {len(session_df)} sessions")
    return session_df


def classify_session(up_acc, down_acc, up_rej, down_rej, up_att, down_att):
    """Classify session based on discovery outcomes."""
    has_up_acceptance = up_acc > 0
    has_down_acceptance = down_acc > 0
    has_up_rejection = up_rej > 0
    has_down_rejection = down_rej > 0
    
    # Directional
    if has_up_acceptance and has_down_acceptance:
        return "two_sided_discovery"
    elif has_up_acceptance and not has_down_acceptance:
        if has_down_rejection:
            return "failed_discovery_up"  # Rotation
        return "net_discovery_up"
    elif has_down_acceptance and not has_up_acceptance:
        if has_up_rejection:
            return "failed_discovery_down"  # Rotation
        return "net_discovery_down"
    
    # Failed Directional
    if has_up_rejection and has_down_rejection:
        return "failed_two_sided_balance"
    elif has_up_rejection:
        return "failed_up_balance"
    elif has_down_rejection:
        return "failed_down_balance"
    
    # Pure Balance
    return "balanced"


def classify_close(close_price, high_ref, low_ref):
    """Classify closing price relative to reference range."""
    if pd.isna(close_price) or pd.isna(high_ref) or pd.isna(low_ref):
        return "unknown"
    
    if close_price > high_ref:
        return "above"
    elif close_price < low_ref:
        return "below"
    else:
        return "within"


def save_session_outcomes(session_df, output_path=None):
    """Save session outcomes to CSV."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "session_outcomes.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    session_df.to_csv(output_path, index=False)
    print(f"Saved session outcomes to {output_path}")


if __name__ == "__main__":
    # This would be run after episode generation
    print("session_logger.py - run via run_pipeline.py")
