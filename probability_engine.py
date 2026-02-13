"""
probability_engine.py — Compute conditional probabilities from transition counts.

Per Agent.md spec:
- Convert context→outcome transition counts into P(outcome | context)
- Aggregate across sessions for reliable probabilities
- Support stratification by context features (failure_count, extension_bin, etc.)
"""

import pandas as pd
import numpy as np
import os

from config import CONFIG, FAILURE_COUNT_CAP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def compute_outcome_probabilities(transition_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """
    Convert transition counts into conditional probabilities P(outcome | context).
    
    Args:
        transition_df: DataFrame with context columns + outcome column + count
        phase: 'pre_ib' or 'post_ib'
    
    Returns:
        DataFrame with probabilities for each context→outcome combination
    """
    if transition_df.empty:
        return pd.DataFrame()
    
    # Identify context columns (everything except 'outcome' and 'count')
    context_cols = [c for c in transition_df.columns if c not in ['outcome', 'count', 'terminal_outcome']]
    outcome_col = 'terminal_outcome' if 'terminal_outcome' in transition_df.columns else 'outcome'
    
    if not context_cols:
        # Simple overall probability
        total = transition_df['count'].sum()
        probs = transition_df.copy()
        probs['probability'] = probs['count'] / total
        probs['phase'] = phase
        return probs
    
    # Group by context and compute probabilities within each context
    results = []
    for ctx, group in transition_df.groupby(context_cols, dropna=False):
        total = group['count'].sum()
        for _, row in group.iterrows():
            prob_row = row.to_dict()
            prob_row['probability'] = row['count'] / total if total > 0 else 0.0
            prob_row['context_count'] = total
            prob_row['phase'] = phase
            results.append(prob_row)
    
    return pd.DataFrame(results)


def compute_transition_matrix(episodes_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """
    Compute state-to-state transition matrix for discovery episodes.
    
    Per Agent.md: For each failure_count level (0-4+), compute P(outcome | failure_count).
    
    Args:
        episodes_df: DataFrame of episodes with failure_count and terminal_outcome
        phase: 'pre_ib' or 'post_ib'
    
    Returns:
        DataFrame with transition probabilities
    """
    if episodes_df.empty:
        return pd.DataFrame()
    
    # Only include discovery episodes
    discovery = episodes_df[episodes_df['state_type'] == 'discovery'].copy()
    if discovery.empty:
        discovery = episodes_df.copy()  # Fall back if no state_type column
    
    fc_col = 'failure_count_at_start' if 'failure_count_at_start' in discovery.columns else 'failure_count'
    
    # Cap failure count at 4+
    discovery['fc_capped'] = discovery[fc_col].clip(upper=FAILURE_COUNT_CAP)
    
    # Count transitions
    trans = discovery.groupby(['fc_capped', 'terminal_outcome']).size().reset_index(name='count')
    
    # Convert to probabilities
    results = []
    for fc, group in trans.groupby('fc_capped'):
        total = group['count'].sum()
        for _, row in group.iterrows():
            results.append({
                'failure_count': int(fc),
                'terminal_outcome': row['terminal_outcome'],
                'count': row['count'],
                'probability': row['count'] / total if total > 0 else 0.0,
                'context_count': total,
                'phase': phase,
            })
    
    return pd.DataFrame(results)


def compute_extension_probabilities(episodes_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """
    Compute P(outcome | extension_bin) using ATR-normalized extensions.
    
    Per Agent.md Extension bins:
    - 0.0-0.5 ATR
    - 0.5-1.0 ATR
    - 1.0-1.5 ATR
    - 1.5-2.0 ATR
    - 2.0+ ATR
    
    Args:
        episodes_df: DataFrame with extension and atr columns
        phase: 'pre_ib' or 'post_ib'
    
    Returns:
        DataFrame with extension-based probabilities
    """
    if episodes_df.empty:
        return pd.DataFrame()
    
    discovery = episodes_df[episodes_df['state_type'] == 'discovery'].copy()
    if discovery.empty:
        discovery = episodes_df.copy()
    
    # Get extension column
    ext_col = None
    for col in ['max_extension', 'extension', 'peak_extension']:
        if col in discovery.columns:
            ext_col = col
            break
    
    atr_col = None
    for col in [f'atr_{CONFIG["atr_window"]}', 'atr', 'atr_5', 'atr_14']:
        if col in discovery.columns:
            atr_col = col
            break
    
    if ext_col is None:
        return pd.DataFrame()
    
    # Bin extensions by ATR
    if atr_col and atr_col in discovery.columns:
        discovery['extension_atr'] = discovery[ext_col] / discovery[atr_col].replace(0, np.nan)
    else:
        discovery['extension_atr'] = discovery[ext_col]  # Use raw extension if no ATR
    
    bins = [0, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
    discovery['extension_bin'] = pd.cut(
        discovery['extension_atr'], bins=bins, labels=labels, include_lowest=True
    )
    
    # Count outcomes by extension bin
    trans = discovery.groupby(['extension_bin', 'terminal_outcome']).size().reset_index(name='count')
    
    # Convert to probabilities
    results = []
    for ext_bin, group in trans.groupby('extension_bin'):
        total = group['count'].sum()
        for _, row in group.iterrows():
            results.append({
                'extension_bin': str(ext_bin),
                'terminal_outcome': row['terminal_outcome'],
                'count': row['count'],
                'probability': row['count'] / total if total > 0 else 0.0,
                'context_count': total,
                'phase': phase,
            })
    
    return pd.DataFrame(results)


def compute_or_position_probabilities(episodes_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """
    Compute P(outcome | or_position) for opening range relative positions.
    
    Per Agent.md OR positions:
    - above_vah
    - within_value
    - below_val
    - at_vah
    - at_val
    
    Args:
        episodes_df: DataFrame with or_position column
        phase: 'pre_ib' or 'post_ib'
    
    Returns:
        DataFrame with OR position probabilities
    """
    if episodes_df.empty:
        return pd.DataFrame()
    
    if 'or_position' not in episodes_df.columns:
        return pd.DataFrame()
    
    discovery = episodes_df[episodes_df['state_type'] == 'discovery'].copy()
    if discovery.empty:
        discovery = episodes_df.copy()
    
    trans = discovery.groupby(['or_position', 'terminal_outcome']).size().reset_index(name='count')
    
    results = []
    for pos, group in trans.groupby('or_position'):
        total = group['count'].sum()
        for _, row in group.iterrows():
            results.append({
                'or_position': str(pos),
                'terminal_outcome': row['terminal_outcome'],
                'count': row['count'],
                'probability': row['count'] / total if total > 0 else 0.0,
                'context_count': total,
                'phase': phase,
            })
    
    return pd.DataFrame(results)


def compute_all_probabilities(
    pre_ib_df: pd.DataFrame, 
    post_ib_df: pd.DataFrame,
    pre_trans: pd.DataFrame = None,
    post_trans: pd.DataFrame = None,
) -> dict:
    """
    Compute all probability tables for both phases.
    
    Returns:
        Dictionary with probability tables for each stratification
    """
    results = {}
    
    # Transition matrices (by failure_count)
    results['pre_ib_transition'] = compute_transition_matrix(pre_ib_df, 'pre_ib')
    results['post_ib_transition'] = compute_transition_matrix(post_ib_df, 'post_ib')
    
    # Extension-based probabilities
    results['pre_ib_extension'] = compute_extension_probabilities(pre_ib_df, 'pre_ib')
    results['post_ib_extension'] = compute_extension_probabilities(post_ib_df, 'post_ib')
    
    # OR position probabilities
    results['pre_ib_or_position'] = compute_or_position_probabilities(pre_ib_df, 'pre_ib')
    results['post_ib_or_position'] = compute_or_position_probabilities(post_ib_df, 'post_ib')
    
    return results


def save_probability_tables(prob_tables: dict, output_dir: str = None):
    """Save all probability tables to CSV files."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in prob_tables.items():
        if df is not None and not df.empty:
            output_path = os.path.join(output_dir, f"{name}_probabilities.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {name} probability table to {output_path}")


def print_probability_summary(prob_tables: dict):
    """Print a summary of computed probabilities."""
    print("\n" + "=" * 60)
    print("PROBABILITY SUMMARY")
    print("=" * 60)
    
    for name, df in prob_tables.items():
        if df is None or df.empty:
            print(f"\n{name}: No data")
            continue
        
        phase = df['phase'].iloc[0] if 'phase' in df.columns else 'unknown'
        print(f"\n{name} ({phase}):")
        print("-" * 40)
        
        if 'failure_count' in df.columns:
            # Transition matrix format
            pivot = df.pivot_table(
                index='failure_count',
                columns='terminal_outcome',
                values='probability',
                aggfunc='first',
            ).fillna(0)
            print(pivot.round(3).to_string())
            
        elif 'extension_bin' in df.columns:
            # Extension format
            pivot = df.pivot_table(
                index='extension_bin',
                columns='terminal_outcome',
                values='probability',
                aggfunc='first',
            ).fillna(0)
            print(pivot.round(3).to_string())
            
        elif 'or_position' in df.columns:
            # OR position format
            pivot = df.pivot_table(
                index='or_position',
                columns='terminal_outcome',
                values='probability',
                aggfunc='first',
            ).fillna(0)
            print(pivot.round(3).to_string())
        
        else:
            print(df.head(10).to_string())
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test with existing episode data
    pre_ib_path = os.path.join(OUTPUT_DIR, "episodes_pre_ib.csv")
    post_ib_path = os.path.join(OUTPUT_DIR, "episodes_post_ib.csv")
    
    if os.path.exists(pre_ib_path) and os.path.exists(post_ib_path):
        pre_ib_df = pd.read_csv(pre_ib_path)
        post_ib_df = pd.read_csv(post_ib_path)
        
        prob_tables = compute_all_probabilities(pre_ib_df, post_ib_df)
        save_probability_tables(prob_tables)
        print_probability_summary(prob_tables)
    else:
        print(f"Episode files not found. Run the pipeline first.")
