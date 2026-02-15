ORB – Breakout Entry Logic Comparison Engine (Single Day)

You are building a single-session research engine in Python.

This engine compares multiple breakout entry rules at the 100% (long) and 0% (short) levels of the Opening Range.

This is NOT a trading strategy.
This is NOT a PnL backtest.
This is a controlled intraday event study.

Objective

For a given trading day:

Compute Opening Range (OR).

Evaluate breakout trades at:

OR_high (100%) for long continuation

OR_low (0%) for short continuation

Compare the following entry rules:

Base Case (Immediate Touch)

Option A (Wick Rejection)

Option B (Close Reclaim)

Option C (Break → Pullback → Reclaim Sequence)

All rules must use:

ATR(previous bar) × atr_multiplier as risk

rr_ratio (default 2:1)

Same end_time

Overlapping trades allowed

Identical simulation logic

The goal is to estimate:

P(win | entry_rule)

For each rule and direction.

Function Signature
def compare_breakout_rules(
    day_data: pd.DataFrame,
    opening_window: tuple[str, str],
    end_time: str,
    atr_period: int,
    atr_multiplier: float,
    rr_ratio: float = 2.0
) -> pd.DataFrame:

Input Requirements

day_data must contain exactly one trading session.

Required columns:

timestamp

open

high

low

close

Data must be sorted chronologically.

No cross-session lookahead allowed.

Step 1 – Compute Opening Range

Using opening_window:

OR_high = max(high)

OR_low = min(low)

OR_range = OR_high - OR_low

Step 2 – Risk Model (Identical Across Rules)

At entry:

ATR must be computed using previous bar only

risk = atr_multiplier × ATR

reward = rr_ratio × risk

Long:

stop = entry_price − risk

target = entry_price + reward

Short:

stop = entry_price + risk

target = entry_price − reward

Simulation continues until:

target hit → win

stop hit → loss

both in same candle → inconclusive

end_time reached before resolution → classify as loss (default)

Exclude inconclusive from win-rate denominator.

Overlapping trades are allowed.

Step 3 – Entry Rule Definitions

All rules operate on 5-minute bars.

1. Base Case – Immediate Breakout Touch

Long:

If high >= OR_high → enter long at OR_high

Short:

If low <= OR_low → enter short at OR_low

2. Option A – Wick Rejection

Long:

Candle trades above OR_high

Candle closes below OR_high

Upper wick exists (high > close)

Enter long at OR_high at close of that candle.

Short:

Candle trades below OR_low

Candle closes above OR_low

Lower wick exists (low < close)

Enter short at OR_low at close of that candle.

3. Option B – Close Reclaim

Long:

Candle trades below OR_high

Candle closes above OR_high

Enter long at close of that candle.

Short:

Candle trades above OR_low

Candle closes below OR_low

Enter short at close of that candle.

4. Option C – Break → Pullback → Reclaim Sequence

Long state machine:

Break:

high > OR_high

Pullback:

subsequent candle low < OR_high

Reclaim:

subsequent candle closes above OR_high

Enter long at close of reclaim candle.

Short mirror logic:

Break:

low < OR_low

Pullback:

subsequent candle high > OR_low

Reclaim:

subsequent candle closes below OR_low

Enter short at close of reclaim candle.

State machine must reset after entry or full invalidation.

Step 4 – Output

Return a DataFrame:

| rule | direction | trades | wins | losses | win_rate |

Where:

rule ∈ {base, A, B, C}

direction ∈ {long, short}

win_rate = wins / (wins + losses)

No ranking or strategy recommendation logic.

Constraints

No capital modeling

No position blocking

No portfolio logic

No hardcoded data source

No forward-looking bias

ATR must use only past data

Code Quality Requirements

Modular rule functions:

base_rule()

wick_rejection_rule()

reclaim_rule()

break_pullback_reclaim_rule()

Clean state machine for Option C

Clear docstrings

Type hints

Clean pandas implementation

Intent

This engine exists to answer:

Is aggressive breakout superior?

Does wick rejection improve edge?

Does acceptance reclaim matter more?

Does structural sequencing (Option C) outperform single-candle logic?

This is a breakout entry quality comparison study.