"""
transition_counter.py — Count context→outcome transitions for Phase 2 Markov matrix.

Per Agent.md Transition Count Schema:
- direction: up | down
- failure_count: {0, 1, 2, 3, 4+}
- extension_bin: 0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0+ (ATR units)
- phase: pre_ib | post_ib
- count_acceptance, count_acceptance_incomplete, count_rejection
- total_observations, avg_duration_minutes
"""

import pandas as pd
import numpy as np
import os

from config import CONFIG, EXTENSION_BINS, EXTENSION_BIN_LABELS, FAILURE_COUNT_CAP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def bin_extension(ext_atr):
    """Bin extension (in ATR units) into predefined categories."""
    if pd.isna(ext_atr):
        return "unknown"
    
    for i, (low, high) in enumerate(zip(EXTENSION_BINS[:-1], EXTENSION_BINS[1:])):
        if low <= ext_atr < high:
            return EXTENSION_BIN_LABELS[i]
    
    return EXTENSION_BIN_LABELS[-1]  # 2.0+


def cap_failure_count(fc):
    """Cap failure count at 4."""
    if pd.isna(fc):
        return 0
    return min(int(fc), FAILURE_COUNT_CAP)


def compute_transition_counts(episodes_df, phase):
    """Compute context→outcome transition counts.
    
    Args:
        episodes_df: Episode DataFrame with direction, failure_count_at_start,
                     extension_atr, terminal_outcome, duration_minutes
        phase: 'pre_ib' or 'post_ib'
    
    Returns:
        DataFrame with transition counts per context
    """
    if episodes_df.empty:
        return pd.DataFrame()
    
    # Filter to discovery episodes only
    disc_eps = episodes_df[episodes_df["state_type"] == "discovery"].copy()
    if disc_eps.empty:
        return pd.DataFrame()
    
    # Cap failure count
    fc_col = "failure_count_at_start" if "failure_count_at_start" in disc_eps.columns else "failure_count"
    disc_eps["failure_count_capped"] = disc_eps[fc_col].apply(cap_failure_count)
    
    # Bin extension
    ext_col = "extension_atr" if "extension_atr" in disc_eps.columns else "max_extension"
    disc_eps["extension_bin"] = disc_eps[ext_col].apply(bin_extension)
    
    # Group by context
    context_cols = ["direction", "failure_count_capped", "extension_bin"]
    
    transition_data = []
    
    for (direction, fc, ext_bin), group in disc_eps.groupby(context_cols):
        outcomes = group["terminal_outcome"].value_counts()
        
        count_a = outcomes.get("A", 0)
        count_a_star = outcomes.get("A*", 0)
        count_r = outcomes.get("R", 0)
        total = len(group)
        
        avg_duration = group["duration_minutes"].mean() if "duration_minutes" in group.columns else np.nan
        
        transition_data.append({
            "direction": direction,
            "failure_count": fc,
            "extension_bin": ext_bin,
            "phase": phase,
            "count_acceptance": count_a,
            "count_acceptance_incomplete": count_a_star,
            "count_rejection": count_r,
            "total_observations": total,
            "avg_duration_minutes": avg_duration,
        })
    
    trans_df = pd.DataFrame(transition_data)
    print(f"Computed {len(trans_df)} transition contexts for {phase}")
    return trans_df


def save_transition_counts(trans_df, output_path):
    """Save transition counts to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trans_df.to_csv(output_path, index=False)
    print(f"Saved transition counts to {output_path}")


if __name__ == "__main__":
    print("transition_counter.py - run via run_pipeline.py")
