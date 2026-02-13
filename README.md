# Value Discovery State Agent

> **This README is auto-generated from Agent.md. Do not edit manually.**

## Project Overview

This project analyzes auction behavior using 5-minute market data to empirically
estimate how markets transition between balance and discovery.

It implements a **two-phase discovery analysis**:
1. **Pre-IB (6:30-7:25am PT)**: Discovery vs prior session VAH/VAL
2. **Post-IB (7:30am-12:55pm PT)**: Discovery vs IBH/IBL, extensions normalized by IB range

Sessions span a full ~24-hour window (6:30am to 6:25am next day) to capture
US RTH, afternoon Globex, Asia, and London sessions for volume profile computation.

**Important: All results are empirical observations, not predictive trading signals.
No predictive trading decisions are made by this system.**

## Two-Phase Architecture

### Phase 1: Pre-IB (Initial Balance Formation)
- **Time**: 6:30am - 7:25am PT (12 five-minute bars)
- **Reference**: Prior session VAH / VAL
- **Purpose**: Assess early discovery behavior relative to prior day value

### Phase 2: Post-IB (Main Session)
- **Time**: 7:30am - 12:55pm PT
- **Reference**: IBH (Initial Balance High) / IBL (Initial Balance Low)
- **Extensions**: Normalized by IB range (IBH - IBL)

### Transition at 7:30am
At the IB boundary, active pre-IB episodes are re-evaluated against IBH/IBL:
- If still in discovery (price outside IB in same direction): carry over with reset extension
- If now in balance (price inside IB): episode rejected, failure count increments
- Failure counts carry across the transition

## Discovery States

Each discovery attempt is classified into one of three terminal outcomes:

| State | Name | Description |
|-------|------|-------------|
| **A** | Acceptance | Price extends and sustains outside reference, zero prior failures |
| **R** | Rejection | Price returns into reference boundary |
| **C** | Continuation | Price remains outside reference at end, had prior failures |

Discovery persistence is tracked as D0, D1, D2, ... based on the number of
failed attempts within a session.

## Data Requirements

- **Format**: 5-minute OHLCV data in CSV format
- **Required columns**: DateTime, Open, High, Low, Close, Volume(from bar)
- **Timestamps**: Pacific Time (PST/PDT)
- **Span**: Multiple years recommended for statistical significance
- **Expected file**: `MNQ_5min_2021Jan_2026Jan.csv` (Micro E-mini Nasdaq-100)

No external indicators need to be precomputed. All derived inputs (VAH/VAL/POC,
VWAP, ATR, IB) are computed by the pipeline scripts.

## How to Run

### Full Pipeline

```bash
python run_pipeline.py
```

This executes all scripts in dependency order, runs tests, and generates outputs.

### Individual Scripts

| Script | Purpose |
|--------|---------|
| `load_data.py` | Load, validate, assign session dates and phases |
| `session_profile.py` | Compute 24hr volume profile (VAH/VAL/POC) and IB |
| `vwap_engine.py` | Session-anchored VWAP + standard deviation bands |
| `volatility.py` | ATR(14) on 5-minute bars |
| `state_engine.py` | Two-phase discovery state machine |
| `episode_logger.py` | Persist episode-level data for both phases |
| `probability_engine.py` | Compute per-phase empirical probabilities |
| `tests_pipeline.py` | Verify pipeline executed correctly |
| `tests_structural.py` | Validate market structure assumptions per phase |
| `readme_generator.py` | Generate this README from Agent.md |

### Running Tests Only

```bash
python tests_pipeline.py    # Pipeline correctness tests (must pass)
python tests_structural.py  # Market structure validation tests (reported)
```

## Outputs

All outputs are written to the `output/` directory:

| File | Description |
|------|-------------|
| `cleaned_data.csv` | Validated bar-level data with session_date and phase |
| `session_profiles.csv` | Per-session VAH/VAL/POC, IBH/IBL/IB_range, RTH stats |
| `vwap_data.csv` | Session-anchored VWAP and bands |
| `volatility_data.csv` | ATR(14) series |
| `episodes_pre_ib.csv` | Raw pre-IB episode data |
| `episodes_post_ib.csv` | Raw post-IB episode data |
| `episode_log_pre_ib.csv` | Enriched pre-IB episode log |
| `episode_log_post_ib.csv` | Enriched post-IB episode log |
| `episode_summary_pre_ib.csv` | Pre-IB summary by state and outcome |
| `episode_summary_post_ib.csv` | Post-IB summary by state and outcome |
| `daily_summary.csv` | Daily classification vs prior session value |
| `pre_ib_probability_tables.csv` | Pre-IB transition probabilities |
| `pre_ib_transition_matrix.csv` | Pre-IB transition matrix |
| `pre_ib_extension_stats.csv` | Pre-IB extension statistics |
| `post_ib_probability_tables.csv` | Post-IB transition probabilities |
| `post_ib_transition_matrix.csv` | Post-IB transition matrix |
| `post_ib_extension_stats.csv` | Post-IB extension statistics (IB-normalized) |
| `pipeline_test_report.txt` | Pipeline test results |
| `structural_test_report.txt` | Structural test results (both phases) |

## Testing Philosophy

The system implements two categories of tests:

### Pipeline Tests (Must Pass)

These confirm the system is functioning correctly:
- Data loads without gaps or ordering errors
- Sessions are detected correctly (6:30am boundary)
- VAH/VAL/POC exist for all sessions
- IB computed for sessions with pre-IB bars
- VWAP and ATR series are non-null
- Episodes are logged in both phases (non-zero count)
- Probability tables are non-empty for both phases
- Output files are written successfully

**Failure indicates a bug or misconfiguration.**

### Structural Tests (Research Validity)

These validate the logical consistency of results, run separately for each phase:
- Acceptance probability decreases with failure count
- Rejection probability increases after repeated failures
- Extension distributions widen with persistence
- Upside and downside statistics are approximately symmetric
- Sample sizes per state exceed minimum thresholds

**Failure indicates invalid assumptions or insufficient data, not code errors.**

## Design Principles

- Explicit time frame (5-minute only)
- Deterministic state logic
- Two-phase analysis with IB as structural pivot
- Empirical probabilities over predictions
- Strong separation between computation and interpretation
- Tests before trust

## Disclaimer

This system produces empirical observations about market microstructure. Results
describe historical patterns and are not predictive of future market behavior.
No trading decisions should be based solely on these outputs.

---

*Generated from Agent.md by readme_generator.py*
