🧠 Agent Role
You are implementing a single-session research tool that evaluates conditional breakout efficiency inside an Opening Range for one specific trading day.
This is not a portfolio backtester.
This is not a multi-day strategy evaluator.
It is an intraday event study for a given session.

📌 Objective
For a single trading day, compute the conditional win/loss statistics of ATR-scaled 2:1 trades triggered every time price touches a fib location inside the Opening Range.
Overlapping trades are allowed.
The goal is to map: The probability of winning at a given fib %
Separately for long and short.

📥 Inputs
The function must accept:
    analyze_single_day(
        day_data: pd.DataFrame,
        opening_window: tuple[str, str],
        end_time: str,
        fib_step: float,
        atr_period: int,
        atr_multiplier: float,
        rr_ratio: float = 2.0
    )

day_data
    Must contain only one trading session of OHLC data.
    Columns required:
        timestamp
        open
        high
        low
        close
    No cross-session lookahead allowed.

The function will read from config.py and generate any global variables there as needed.

🏗 Algorithm Requirements
1️⃣ Compute Opening Range
Using opening_window:
    OR_high
    OR_low
    OR_range = OR_high - OR_low

2️⃣ Generate Fib Grid
fib_percents = np.arange(0, 1 + fib_step, fib_step)
entry_price = OR_low + fib_percent * OR_range

3️⃣ Trade Event Scanning (Overlapping Allowed)
For each fib level:
    Scan bars from end of opening_window until end_time.
    Whenever price touches entry_price:
        Compute ATR using previous bar only
        risk = atr_multiplier × ATR
        reward = rr_ratio × risk
    Simulate forward until:
        target hit → win
        stop hit → loss
        both hit in same candle → inconclusive
        end_time reached → classify explicitly (default = loss)
    Do NOT block additional triggers.
    Every touch is a separate event.

📊 Output
Return a DataFrame:
| fib_percent | long_trades | long_wins | long_win_rate | short_trades | short_wins | short_win_rate |
Win rate calculation:
    win_rate = wins / (wins + losses)
Exclude inconclusive from denominator, but note them separately.

🚫 Explicit Constraints
No capital modeling
No position blocking
No PnL tracking
No cross-day aggregation
No forward-looking bias
ATR must be computed strictly from past data

🎯 Interpretation Context
This engine exists to answer:
Should I wait for breakout?
Can I pre-empt inside the OR?
Where does directional resolution statistically favor continuation?
This is a structural probability surface, not a trading system.

🧪 Optional Extension
Agent may optionally compute:
EV = rr_ratio * P(win) - 1 * (1 - P(win))

But this is informational only.