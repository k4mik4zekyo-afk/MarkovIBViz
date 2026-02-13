"""
run_pipeline.py — Run the full Phase 1 pipeline end-to-end.

Per Agent.md execution order:
1. Load and validate data (resample if needed)
2. Compute session profiles (Prior RTH & Overnight separately)
3. Compute Opening Range analysis
4. Compute VWAP (anchored to 6:30am)
5. Compute ATR (configurable window)
6. Encode discovery states and log episodes
7. Compute session-level outcomes
8. Compute transition counts
9. Run tests
10. Generate README

Configuration is read from config.py
"""

import sys
import os
import time

from config import CONFIG

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("MARKOV DISCOVERY MODEL — PHASE 1 PIPELINE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Timeframe: {CONFIG['timeframe']}")
    print(f"  ATR Window: {CONFIG['atr_window']}")
    print(f"  Opening Window: {CONFIG['opening_window_minutes']} minutes")
    print(f"  Acceptance Threshold: {CONFIG['acceptance_threshold_minutes']} minutes")
    print(f"  Data Range: {CONFIG['data_start_year']}-{CONFIG['data_end_year']}")
    print("=" * 60)
    
    # ---- Step 1: Load and validate data ----
    print("\n[1/10] Loading and validating data...")
    from load_data import load_and_validate, save_cleaned
    df = load_and_validate()
    save_cleaned(df)
    
    # ---- Step 2: Compute session profiles ----
    print("\n[2/10] Computing session profiles (Prior RTH & Overnight separately)...")
    from session_profile import (compute_session_profiles, merge_profiles_to_bars, 
                                  save_profiles, save_prior_rth_profiles, save_overnight_profiles)
    profiles = compute_session_profiles(df)
    save_profiles(profiles)
    save_prior_rth_profiles(profiles)
    save_overnight_profiles(profiles)
    df = merge_profiles_to_bars(df, profiles)
    
    # ---- Step 3: Compute Opening Range ----
    print("\n[3/10] Computing Opening Range analysis...")
    from opening_range import compute_opening_range, merge_or_to_bars, save_opening_range
    or_df = compute_opening_range(df, profiles)
    save_opening_range(or_df)
    df = merge_or_to_bars(df, or_df)
    
    # ---- Step 4: Compute VWAP ----
    print("\n[4/10] Computing session-anchored VWAP...")
    from vwap_engine import compute_session_vwap, save_vwap
    df = compute_session_vwap(df)
    save_vwap(df)
    
    # ---- Step 5: Compute ATR ----
    print(f"\n[5/10] Computing ATR({CONFIG['atr_window']})...")
    from volatility import compute_atr, save_volatility
    df = compute_atr(df)
    save_volatility(df)
    
    # ---- Step 6: Encode discovery states and log episodes ----
    print("\n[6/10] Encoding two-phase discovery states...")
    from state_engine import encode_discovery_states, save_episodes
    pre_ib_episodes, post_ib_episodes = encode_discovery_states(df)
    save_episodes(pre_ib_episodes, os.path.join(OUTPUT_DIR, "episodes_pre_ib.csv"))
    save_episodes(post_ib_episodes, os.path.join(OUTPUT_DIR, "episodes_post_ib.csv"))
    
    # Enrich and save episode logs
    from episode_logger import (_enrich_episodes, save_episode_log, save_episode_summary,
                                 build_daily_summary)
    import pandas as pd
    
    pre_ib_df = _enrich_episodes(pre_ib_episodes)
    post_ib_df = _enrich_episodes(post_ib_episodes)
    
    save_episode_log(pre_ib_df, os.path.join(OUTPUT_DIR, "episode_log_pre_ib.csv"))
    save_episode_log(post_ib_df, os.path.join(OUTPUT_DIR, "episode_log_post_ib.csv"))
    save_episode_summary(pre_ib_df, os.path.join(OUTPUT_DIR, "episode_summary_pre_ib.csv"))
    save_episode_summary(post_ib_df, os.path.join(OUTPUT_DIR, "episode_summary_post_ib.csv"))
    
    # Daily summary
    daily = build_daily_summary(pre_ib_df, post_ib_df, profiles)
    daily.to_csv(os.path.join(OUTPUT_DIR, "daily_summary.csv"), index=False)
    print(f"Saved daily summary ({len(daily)} days)")
    
    # ---- Step 7: Compute session-level outcomes ----
    print("\n[7/10] Computing session-level outcomes...")
    from session_logger import compute_session_outcomes, save_session_outcomes
    session_outcomes = compute_session_outcomes(pre_ib_df, post_ib_df, profiles)
    save_session_outcomes(session_outcomes)
    
    # ---- Step 8: Compute transition counts ----
    print("\n[8/10] Computing transition counts...")
    from transition_counter import compute_transition_counts, save_transition_counts
    pre_trans = compute_transition_counts(pre_ib_df, "pre_ib")
    post_trans = compute_transition_counts(post_ib_df, "post_ib")
    save_transition_counts(pre_trans, os.path.join(OUTPUT_DIR, "transition_counts_pre_ib.csv"))
    save_transition_counts(post_trans, os.path.join(OUTPUT_DIR, "transition_counts_post_ib.csv"))
    
    # ---- Step 9: Compute probabilities ----
    print("\n[9/10] Computing empirical probabilities per phase...")
    from probability_engine import compute_all_probabilities, save_probability_tables, print_probability_summary
    prob_tables = compute_all_probabilities(pre_ib_df, post_ib_df, pre_trans, post_trans)
    save_probability_tables(prob_tables)
    print_probability_summary(prob_tables)
    
    # ---- Step 10: Run tests ----
    print("\n[10/10] Running tests...")
    
    # Pipeline tests (must pass)
    print("\n--- Pipeline Tests ---")
    from tests_pipeline import PipelineTestRunner
    pipeline_runner = PipelineTestRunner()
    pipeline_passed, pipeline_results = pipeline_runner.run_all()
    pipeline_runner.save_report()
    
    # ---- Generate README ----
    print("\nGenerating README.md...")
    try:
        from readme_generator import generate_readme
        generate_readme()
    except Exception as e:
        print(f"Warning: Could not generate README: {e}")
    
    # ---- Final Report ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Total bars: {len(df)}")
    print(f"Total sessions: {profiles.shape[0]}")
    print(f"Pre-IB episodes: {len(pre_ib_df)}")
    print(f"Post-IB episodes: {len(post_ib_df)}")
    
    if not pre_ib_df.empty:
        print(f"Pre-IB outcomes: {pre_ib_df['terminal_outcome'].value_counts().to_dict()}")
    if not post_ib_df.empty:
        print(f"Post-IB outcomes: {post_ib_df['terminal_outcome'].value_counts().to_dict()}")
    
    # Session outcome summary
    if not session_outcomes.empty:
        print(f"Session classifications: {session_outcomes['session_classification'].value_counts().to_dict()}")
    
    print(f"Pipeline tests: {'PASSED' if pipeline_passed else 'FAILED'}")
    print("=" * 60)
    
    if not pipeline_passed:
        print("\nFATAL: Pipeline tests failed. Outputs should not be trusted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
