# Research Tools Usage Guide

This repository contains several research tools for analyzing market microstructure and trading patterns.

## Main Pipeline (Value Discovery State Agent)

The main pipeline analyzes market discovery behavior across two phases (Pre-IB and Post-IB).

```bash
# Run full pipeline
python run_pipeline.py

# Run tests
python tests_pipeline.py
python tests_structural.py
```

See [README.md](README.md) for complete documentation.

---

## Smart Range Analysis

**File:** `smart_range.py`

Single-session Opening Range fib-grid trade simulator. Computes conditional win/loss statistics for ATR-scaled 2:1 trades at fibonacci levels inside the Opening Range.

### Usage

```bash
# Run with default sample days
python smart_range.py

# Run with specific date
python smart_range.py 2025-10-27

# Run with multiple dates
python smart_range.py 2025-10-27 2025-10-28
```

### Purpose
Maps the probability of winning at each fib % level, separately for long and short trades. Answers: "Should I wait for breakout or can I pre-empt inside the OR?"

### Output
- Console: Pretty-printed table showing win rates at each fib level
- CSV: `output/smart_range_YYYY-MM-DD.csv`

### Configuration
- Opening window: 06:30-06:40 (10 minutes)
- Session end: 13:00
- Fib step: 0.05 (5% increments)
- ATR period: 3 bars
- ATR multiplier: 1.0
- RR ratio: 2:1

See documentation in `smart_range.py` for details.

---

## Breakout Rules Comparison

**File:** `compare_breakout_rules.py`

Single-session ORB (Opening Range Breakout) entry rule comparison engine. Compares multiple breakout entry rules at the 100% (OR_high) and 0% (OR_low) levels.

### Usage

```bash
# Run with default sample days
python compare_breakout_rules.py

# Run with specific date
python compare_breakout_rules.py 2025-10-27

# Run with multiple dates
python compare_breakout_rules.py 2025-10-27 2025-10-28 2025-12-16
```

### Purpose
Compares 4 entry rules to answer:
- Is aggressive breakout superior?
- Does wick rejection improve edge?
- Does close reclaim matter more?
- Does structural sequencing (break→pullback→reclaim) outperform single-candle logic?

### Entry Rules Compared

1. **Base Case**: Immediate breakout touch
   - Long: high >= OR_high → enter at OR_high
   - Short: low <= OR_low → enter at OR_low

2. **Wick Rejection (Option A)**: Wait for rejection
   - Long: Trades above OR_high, closes below, has upper wick → enter at OR_high
   - Short: Trades below OR_low, closes above, has lower wick → enter at OR_low

3. **Close Reclaim (Option B)**: Wait for reclaim
   - Long: Trades below OR_high, closes above → enter at close
   - Short: Trades above OR_low, closes below → enter at close

4. **Break-Pullback-Reclaim (Option C)**: Full sequence
   - Long: Break above → Pullback below → Reclaim above → enter at close
   - Short: Break below → Pullback above → Reclaim below → enter at close

### Output
- Console: Pretty-printed comparison table
- CSV: `output/breakout_comparison_YYYY-MM-DD.csv`

### Configuration
- Opening window: 06:30-06:45 (15 minutes)
- Session end: 13:00
- ATR period: 3 bars
- ATR multiplier: 1.0
- RR ratio: 2:1

See [BREAKOUT_COMPARISON.md](BREAKOUT_COMPARISON.md) for detailed documentation.

---

## Common Features

All research tools share:
- **No forward-looking bias**: ATR computed from previous bar only
- **Overlapping trades allowed**: Every touch triggers new trade event
- **Identical risk model**: ATR-scaled stops, configurable RR ratio
- **Session-based analysis**: Each day analyzed independently
- **CSV output**: Results saved for further analysis
- **5-minute data**: Uses `MNQ_5min_2021Jan_2026Jan.csv`

## Data Requirements

All tools require 5-minute OHLCV data:
- File: `MNQ_5min_2021Jan_2026Jan.csv`
- Columns: DateTime, Open, High, Low, Close, Volume(from bar)
- Timezone: Pacific Time

## Dependencies

```bash
pip install pandas numpy
```

## Comparison Matrix

| Feature | Main Pipeline | Smart Range | Breakout Comparison |
|---------|--------------|-------------|---------------------|
| **Analysis Type** | Multi-day state machine | Single-session fib grid | Single-session rule comparison |
| **Focus** | Discovery vs balance | OR interior levels | OR boundary breakouts |
| **Entry Points** | Boundary extensions | All fib levels (0-100%) | OR_high/OR_low only |
| **Rules Tested** | State transitions | 1 (immediate touch) | 4 (different entry logic) |
| **Output** | Episode logs, probabilities | Win rates by fib % | Win rates by rule |
| **Use Case** | Market regime analysis | OR pre-emption study | Entry timing optimization |

## Research Workflow

1. **Macro view**: Run main pipeline to understand overall market structure
2. **OR interior**: Use `smart_range.py` to study probabilities inside the range
3. **Breakout timing**: Use `compare_breakout_rules.py` to optimize boundary entry
4. **Aggregate**: Collect results across multiple dates for statistical significance

## Notes

- All tools are research-oriented, not trading systems
- No PnL tracking or capital modeling
- Results describe historical patterns only
- No predictive trading decisions should be based solely on these outputs

---

For questions or issues, refer to individual tool documentation or Agent.md.
