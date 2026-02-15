# Breakout Rules Comparison Engine

## Overview

The **compare_breakout_rules.py** script is a single-session research tool that evaluates and compares multiple breakout entry rules at the Opening Range (OR) boundaries.

This is **NOT** a trading strategy or PnL backtest. It is a controlled intraday event study designed to answer:
- Is aggressive breakout superior?
- Does wick rejection improve edge?
- Does acceptance reclaim matter more?
- Does structural sequencing outperform single-candle logic?

## Purpose

For a given trading day, the engine:
1. Computes the Opening Range (OR)
2. Evaluates breakout trades at:
   - **OR_high (100%)** for long continuation
   - **OR_low (0%)** for short continuation
3. Compares four distinct entry rules
4. Reports win/loss statistics for each rule and direction

## Entry Rules Compared

### 1. Base Case — Immediate Breakout Touch
**Long:** If `high >= OR_high` → enter at OR_high  
**Short:** If `low <= OR_low` → enter at OR_low

This is the most aggressive approach: enter immediately when price touches the boundary.

### 2. Option A — Wick Rejection
**Long:**
- Candle trades above OR_high
- Candle closes below OR_high
- Upper wick exists (`high > close`)
- → Enter at OR_high at close of that candle

**Short:**
- Candle trades below OR_low
- Candle closes above OR_low
- Lower wick exists (`low < close`)
- → Enter at OR_low at close of that candle

This approach waits for a rejection of the breakout attempt before entering.

### 3. Option B — Close Reclaim
**Long:**
- Candle trades below OR_high
- Candle closes above OR_high
- → Enter at close of that candle

**Short:**
- Candle trades above OR_low
- Candle closes below OR_low
- → Enter at close of that candle

This approach waits for price to reclaim the boundary level on a closing basis.

### 4. Option C — Break → Pullback → Reclaim Sequence
**Long state machine:**
1. **Break:** `high > OR_high`
2. **Pullback:** subsequent candle `low < OR_high`
3. **Reclaim:** subsequent candle closes above OR_high → **ENTER** at close

**Short:** Mirror logic with `low < OR_low`, then pullback, then reclaim

This approach requires a full sequence: break out, pull back, then reclaim before entering.

## Usage

### Basic Usage
```bash
# Run with default sample days (2025-10-27, 2025-10-28)
python compare_breakout_rules.py

# Run with a single date
python compare_breakout_rules.py 2025-10-27

# Run with multiple dates
python compare_breakout_rules.py 2025-10-27 2025-10-28 2025-12-16
```

### Requirements
- Python 3.7+
- pandas
- numpy

### Data Requirements
The script expects:
- 5-minute OHLCV data in `MNQ_5min_2021Jan_2026Jan.csv`
- Columns: `datetime`, `open`, `high`, `low`, `close`, `volume`

## Risk Model

All rules use **identical risk/reward parameters**:

```python
ATR = compute_rolling_atr(previous_bar, period=3)
risk = atr_multiplier × ATR  (default: 1.0)
reward = rr_ratio × risk     (default: 2:1)
```

**Long trade:**
- Stop = entry_price - risk
- Target = entry_price + reward

**Short trade:**
- Stop = entry_price + risk
- Target = entry_price - reward

**Key constraint:** ATR is computed using **previous bar only** — no forward-looking bias.

## Trade Resolution

Trades are simulated forward until one of these outcomes:
1. **Win:** Target hit before stop
2. **Loss:** Stop hit before target
3. **Inconclusive:** Both hit in same candle (excluded from win rate)
4. **Expired:** Session end_time reached → classified as **loss** (per spec)

**Overlapping trades are allowed** — every boundary touch triggers a new trade event.

## Output Format

### Console Output
```
================================================================================
  BREAKOUT RULES COMPARISON — 2025-10-27
================================================================================
Rule                      Dir    │ Trades   Wins Losses      WR%
────────────────────────────────────────────────────────────────────────────────
base                      long   │     70      69       1      98.6
base                      short  │      2       0       2       0.0
wick_rejection            long   │      2       2       0     100.0
wick_rejection            short  │      2       0       2       0.0
close_reclaim             long   │      5       4       1      80.0
close_reclaim             short  │      0       0       0         –
break_pullback_reclaim    long   │      2       2       0     100.0
break_pullback_reclaim    short  │      0       0       0         –
```

### CSV Output
Saved to `output/breakout_comparison_YYYY-MM-DD.csv`:

```csv
rule,direction,trades,wins,losses,inconclusive,win_rate
base,long,70,69,1,0,98.57
base,short,2,0,2,0,0.0
wick_rejection,long,2,2,0,0,100.0
...
```

**Columns:**
- `rule`: Entry rule name
- `direction`: "long" or "short"
- `trades`: Total number of trades triggered
- `wins`: Number of trades that hit target
- `losses`: Number of trades that hit stop or expired
- `inconclusive`: Number of trades where both target and stop hit in same bar
- `win_rate`: `wins / (wins + losses) × 100` (excludes inconclusive)

## Configuration

Default parameters (can be modified in function call):

```python
DEFAULT_OPENING_WINDOW = ("06:30", "06:45")   # 15-min OR
DEFAULT_END_TIME       = "13:00"              # RTH close
DEFAULT_ATR_PERIOD     = 3                    # bars for ATR lookback
DEFAULT_ATR_MULTIPLIER = 1.0                  # risk = mult × ATR
DEFAULT_RR_RATIO       = 2.0                  # reward : risk
```

## Interpretation

### Example Analysis (2025-10-27)

**Observations:**
1. **Base Case (long):** 70 trades, 98.6% win rate
   - Most aggressive, highest trade count
   - Very high success rate on this bullish day

2. **Wick Rejection (long):** 2 trades, 100% win rate
   - Much fewer trades (more selective)
   - Perfect win rate but small sample

3. **Close Reclaim (long):** 5 trades, 80% win rate
   - Moderate selectivity
   - Good win rate with reasonable sample

4. **Break-Pullback-Reclaim (long):** 2 trades, 100% win rate
   - Most selective (requires full sequence)
   - Perfect but smallest sample

**Key Insights:**
- On trending days (like 10-27), aggressive breakout (base) performs well
- Selective rules reduce trade count but may improve win rate
- Short trades failed on this bullish day (as expected)

### What This Tells You

This engine helps identify:
- **Trade-off between frequency and quality**: More selective rules = fewer trades but potentially higher win rates
- **Directional bias**: Strong directional days favor one side
- **Entry timing**: Does waiting for confirmation improve outcomes?
- **Market structure**: Which entry logic aligns with actual price behavior?

## Constraints

- **No capital modeling:** Each trade is independent
- **No position blocking:** Overlapping trades allowed
- **No PnL tracking:** Focus is on win rate, not dollars
- **No cross-day aggregation:** Each session analyzed independently
- **No forward-looking bias:** ATR uses past data only

## Design Consistency

This implementation follows the same architectural patterns as `smart_range.py`:
- Identical ATR computation (Wilder's smoothing)
- Same risk/reward calculation
- Same trade simulation logic
- Same data loading and validation
- Modular, testable code structure

## Next Steps

For multi-day analysis:
1. Run on multiple dates
2. Aggregate results by rule
3. Compare win rates and sample sizes
4. Identify which rules are most robust across different market regimes

## Questions This Answers

✅ Is aggressive breakout superior?  
✅ Does wick rejection improve edge?  
✅ Does acceptance reclaim matter more?  
✅ Does structural sequencing (Option C) outperform single-candle logic?

This is a **breakout entry quality comparison study**, not a complete trading system.
