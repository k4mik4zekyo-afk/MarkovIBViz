# Agent2.md
**Agent Name:** Markov Discovery Model - Value Area Auction Analysis

## Purpose

This Agent builds a **Markov-based predictive model** for market discovery episodes using **1-minute market data resampled to configurable timeframes** (5m, 15m, or 30m).

The system empirically estimates transition probabilities between auction states to answer: **Given current market context, what is P(Acceptance) vs P(Rejection) for this discovery attempt?**

**Phase 1** constructs the empirical foundation: episode outcome logging with context, session-level outcomes, and transition counting across 2023-2025 data.

**Phase 2** uses Phase 1 outputs to build a Markov transition matrix that quantifies discovery outcome probabilities conditioned on observable market context (failure_count, extension, phase).

**No trading decisions are made.** The model quantifies probabilities to inform future systematic strategies or serve as features for ML models.

---

## Project Scope & Phasing

### Phase 1 (Current Focus)
**Empirical Data Collection & Episode Tracking**

- Data loading (1-min → configurable resample) and validation
- Session profile computation (Prior RTH VAH/VAL/POC, Overnight VAH/VAL/POC, IB)
- Opening Range analysis
- VWAP calculation (anchored to 6:30am RTH open)
- ATR calculation (configurable window, default=5)
- **Episode outcome logging** with context features
- **Session-level outcome** classification
- Transition frequency tables
- Pipeline tests (must pass)
- README generation

**Deliverables:**
- Episode-level summaries with outcomes (A, A*, R, B)
- Session-level outcome classifications
- Transition count tables (context → outcome frequencies)

**Status:** ✅ In Progress

---

### Phase 2 (Future)
**Markov Model Construction & Validation**

#### Phase 2a: Transition Matrix
- Construct state → outcome transition probabilities
- Handle sparse states (minimum observation thresholds)
- Export P(Acceptance | Context), P(Rejection | Context)

#### Phase 2b: Session Outcome Prediction
- Aggregate episode-level predictions to session level
- Model P(Session ends in net discovery up/down/balanced)

#### Phase 2c: Model Validation
- Backtest on hold-out period (e.g., 2025 data)
- Measure calibration: do predicted probabilities match observed frequencies?
- Identify high-confidence contexts vs uncertain contexts

#### Phase 2d: Regime Detection (Optional)
- Sliding window model re-training
- Detect structural breaks in transition probabilities
- Adaptive weighting of recent vs historical data

#### Phase 2e: Alternative Models (Comparative)
- Random Forest classifier (same features)
- Logistic regression
- Compare to Markov baseline

**Status:** ⏸️ Deferred

---

## Operating Assumptions

- Input data is **1-minute OHLCV** (resampled to configurable timeframe)
- **Timeframe parameter**: 5m, 15m, or 30m (user-configurable, default: 5m)
- **ATR window**: Configurable (default: 5 bars on resampled data)
- **Opening window**: Configurable (default: 15 minutes from 6:30am)
- **Acceptance threshold**: 30 minutes sustained outside boundary
- **Two-sided threshold**: 30 minutes (if opposite boundary touched within 30 min, reject current episode)
- Data filtered to **2023-2025 only** (3-year analysis window, ~934 sessions)
- Sessions are well-defined and continuous
- No live trading decisions are made (research/feature engineering only)

---

## Core Concepts

### Three-State System

The market operates in three mutually exclusive states:

1. **Balance (B)**
   - Price remains within reference boundaries
   - Reference: Prior RTH VAH/VAL (pre-IB) or IBH/IBL (post-IB)
   - Episode continues until price breaks out

2. **Discovery Up (DU)**
   - Price extends above upper reference boundary
   - Tracks extension, retracement, failure history, acceptance status
   - Terminal outcomes: A (Acceptance), A* (Incomplete), R (Rejection)

3. **Discovery Down (DD)**
   - Price extends below lower reference boundary
   - Tracks extension, retracement, failure history, acceptance status
   - Terminal outcomes: A (Acceptance), A* (Incomplete), R (Rejection)

---

### Discovery Episode Mechanics

**Episode Start:**
- Triggered when a bar **closes** outside reference boundary AND the bar's high/low **touched** the boundary
- 30-minute acceptance timer starts immediately
- 30-minute two-sided rejection timer starts immediately

**Episode Tracking:**
- `failure_count_at_start`: Number of prior rejections in this direction
- `max_extension`: Furthest distance from boundary (updated each bar)
- `acceptance_achieved`: Flag set to TRUE when 30 minutes elapsed while outside boundary
- `time_in_acceptance`: Duration spent in acceptance state before rejection or session end

**Episode Termination:**

**A (Acceptance):**
- Episode sustained ≥30 minutes outside boundary
- Session or phase ended while still outside boundary
- **No condition on failure_count** - any episode can achieve acceptance

**A\* (Acceptance-Incomplete):**
- Session ended while in discovery state
- Had NOT yet reached 30-minute threshold
- Still outside boundary at session end
- **Does NOT increment failure_count**

**R (Rejection):**
- Price closed back inside reference boundary before 30 minutes elapsed
- **OR** opposite boundary touched within 30 minutes of episode start (two-sided rejection)
- Increments `failure_count` for this direction

**B (Balance):**
- Balance episode ended when discovery started or phase transitioned

---

### Two-Sided Rejection Logic

**Purpose:** Detect when market tests both sides rapidly, indicating indecision rather than directional discovery.

**Mechanism:**
1. Bar closes outside boundary (e.g., high touches IBH) → Discovery Up starts, 30-min timer starts
2. Within 30 minutes, opposite boundary touched (e.g., low touches IBL)
3. → Close current Discovery Up episode with outcome R (two-sided rejection)
4. → Log max_extension and retracement during the episode window
5. → Increment failure_count_up
6. → Immediately start Discovery Down episode (same bar or next bar when close occurs)

**Key points:**
- Timer based on **bar close** where touch occurred, not intrabar touch time
- Opposite boundary **touch** (high/low) triggers rejection, not close
- Current episode metrics logged before starting opposite direction
- VWAP cross does NOT clear the two-sided timer
- Two-sided rejection takes precedence over acceptance timer

**Example:**
```
10:00 - Bar closes, high touched IBH → Discovery Up starts, timer starts
10:15 - Bar's low touches IBL (15 min < 30 min threshold)
        → Close Discovery Up: outcome=R, duration=15 min, log metrics
        → Increment failure_count_up
10:15 - Same bar close or next bar → Discovery Down starts (if close < IBL)
```

---

### Failure Count Mechanics

**Counting:**
- `failure_count` increments on each Rejection (R outcome)
- Tracked separately for up and down directions
- Values: 0, 1, 2, 3, 4+
- **Caps at 4+** (we track "4 or more" as a single state)
- No limit on discovery attempts (market can keep trying)

**VWAP Reset (Option B - At Episode Start Only):**
- When **starting a new discovery episode**, check if VWAP has been crossed since last episode in that direction ended
- If crossed: `failure_count → 0` for that direction
- If not crossed: use existing `failure_count`
- VWAP cross **during an active episode** does NOT reset failure_count mid-episode

**Implementation requires tracking:**
- `vwap_crossed_since_last_up_episode`: Boolean flag
- `vwap_crossed_since_last_down_episode`: Boolean flag
- Flags set when price crosses VWAP while in Balance state
- Flags consumed when new episode starts, then reset to FALSE

**Example:**
```
Session start: failure_count_up = 0, failure_count_down = 0

10:00 - Discovery Up #1 → Rejection → failure_count_up = 1
10:20 - Discovery Up #2 → Rejection → failure_count_up = 2
10:40 - In Balance, price crosses VWAP → vwap_crossed_since_last_up_episode = TRUE
11:00 - Discovery Up #3 starts
        → Check flag: TRUE
        → Reset failure_count_up = 0
        → Set flag = FALSE
11:30 - Discovery Up → Acceptance (fresh attempt succeeded)
```

---

### Acceptance Definition

**Acceptance (A):** Discovery sustained for ≥30 continuous minutes outside reference boundary.

**Operationalization:**
- **At 5-minute timeframe:** ≥6 consecutive bars beyond boundary
- **At 15-minute timeframe:** ≥2 consecutive bars beyond boundary
- **At 30-minute timeframe:** ≥1 bar beyond boundary

**Key constraint:** Price must remain continuously outside boundary for full duration. If price closes back inside boundary at any point during the 30 minutes, episode ends as Rejection.

**Acceptance is independent of failure_count:**
- Episode with failure_count=0 can achieve acceptance
- Episode with failure_count=4+ can also achieve acceptance
- failure_count is a **context feature** for probability estimation, NOT a barrier to acceptance

**Why 30 minutes?**
- Exceeds typical "test and reject" timeframes
- Represents meaningful commitment by auction participants
- Half the duration of Initial Balance (60 minutes)
- Same threshold as two-sided rejection window (consistent logic)

---

### Episode Definition

An **episode** is a contiguous state sequence logged when it terminates.

**Episode-level attributes (logged at termination):**

```
session_date              Trading session date
episode_id                Unique identifier (session_date + sequence)
state_type                "balance" | "discovery"
direction                 None (balance) | "up" | "down" (discovery)
failure_count_at_start    {0, 1, 2, 3, 4+} when episode began
start_time                Episode start timestamp
end_time                  Episode end timestamp
duration_minutes          Total episode duration
max_extension_points      Furthest price move from boundary (raw points)
max_extension_atr         max_extension / avg_atr_during_episode
max_extension_ib          max_extension / IB_range (post-IB only, NULL pre-IB)
max_extension_or          max_extension / OR_range (all episodes)
max_retracement_points    Largest pullback within episode (raw points)
acceptance_achieved       TRUE if sustained ≥30 min, FALSE otherwise
time_in_acceptance_min    Duration in acceptance state (0 if not achieved)
terminal_outcome          A | A* | R | B
phase                     "pre_ib" | "post_ib"
avg_atr_during_episode    Average ATR across episode bars
ib_range                  Session's IB range (points)
or_range                  Session's OR range (ORH - ORL, points)
reference_boundary        Actual price level used (VAH/VAL/IBH/IBL)
```

**Terminal Outcomes:**

- **A (Acceptance):** Sustained ≥30 min, session/phase ended outside boundary
- **A\* (Incomplete):** Session ended in discovery, <30 min elapsed (does NOT increment failure_count)
- **R (Rejection):** Returned inside boundary OR opposite boundary touched within 30 min
- **B (Balance):** Balance episode ended when discovery started

---

### Session-Level Outcomes

In addition to episode tracking, each session receives a holistic classification.

**Session outcome categories:**

**Directional:**
- **net_discovery_up:** Session ended with accepted upside discovery
- **net_discovery_down:** Session ended with accepted downside discovery
- **two_sided_discovery:** Both upside and downside discovery accepted

**Failed Directional:**
- **failed_up_balance:** Any upside rejection attempts, remained in value area
- **failed_down_balance:** Any downside rejection attempts, remained in value area
- **failed_two_sided_balance:** Rejected both directions, remained in value (choppy session)

**Rotation:**
- **failed_discovery_up:** Rejected downside attempts, then rotated to upside acceptance
- **failed_discovery_down:** Rejected upside attempts, then rotated to downside acceptance

**Pure Balance:**
- **balanced:** No acceptances either direction, stayed in value, minimal rejection attempts

**Tracked metrics per session:**
- Number of discovery attempts (up/down)
- Number of acceptances (up/down)
- Number of rejections (up/down)
- Maximum extension achieved (up/down)
- Time spent in balance vs discovery states
- Final closing price relative to: prior RTH value, overnight value, IB
- Final failure_count (up/down) at session end

**Purpose:** Understand session character beyond individual episodes for regime analysis and strategy design.

---

## Two-Phase Architecture

### Pre-IB Phase (6:30am - 7:25am PT)
**Purpose:** Initial Balance formation period

- **Duration:** Varies by timeframe (12 bars @ 5m, 4 bars @ 15m, 2 bars @ 30m)
- **Reference boundaries:** Prior RTH VAH/VAL (primary), Overnight VAH/VAL (secondary)
- **State tracking:** Balance and Discovery episodes
- **Discovery attempts:** Unlimited (failure_count can be 4+)
- **Outcome types:** R (Rejection), A* (Incomplete if episode active at 7:30am)
  - Acceptance (A) only if episode sustained full 30 min before 7:30am
- **IB Calculation:** IBH = max(high), IBL = min(low) over pre-IB period
- **Opening Range Analysis:**
  - Compute ORH/ORL from first N minutes (default: 15 min, configurable)
  - Classify OR position relative to prior RTH VAH/VAL/POC
  - Classify OR position relative to overnight VAH/VAL/POC
  - Track if price retests prior RTH value area during pre-IB
  - Track where price is at 7:30am (IB formation) relative to prior RTH value

**Episode logging:** Episode-level summaries only (no bar-level tracking)

---

### Transition (7:30am)
**Purpose:** Re-evaluate active episodes against newly formed IB

**Balance episodes:**
- Close with outcome B

**Discovery episodes:**
- Evaluate position relative to new reference (IBH/IBL)
- **If still outside IB in same direction:**
  - Close pre-IB episode with outcome A* (incomplete) or A (if ≥30 min sustained)
  - Start new post-IB episode immediately
  - Extension resets (now measured from IBH/IBL, not VAH/VAL)
  - failure_count carries over (does NOT reset at phase transition)
- **If now inside IB:**
  - Close episode with outcome R (rejection)
  - Increment failure_count
  - Enter Balance state
- **If in opposite direction:**
  - Close episode as R
  - Evaluate for new discovery in that direction

**Key principle:** Phase transition does NOT reset failure counts. Pre-IB rejections inform post-IB context.

---

### Post-IB Phase (7:30am - 12:55pm PT)
**Purpose:** Main session discovery analysis

- **Duration:** 66 bars @ 5m (330 minutes)
- **Reference boundaries:** IBH/IBL (NOT prior VAH/VAL)
- **State tracking:** Balance and Discovery episodes
- **Discovery attempts:** Unlimited (failure_count can be 4+)
- **Outcome types:** All outcomes possible (A, A*, R)
- **Extension normalization:**
  - extension_atr: Points / avg_atr_during_episode
  - extension_ib: Points / IB_range
  - extension_or: Points / OR_range
- **Acceptance evaluation:** 30-minute sustained threshold applies
- **Two-sided rejection:** 30-minute window applies

**Episode logging:** Episode-level summaries only (no bar-level tracking)

---

### Overnight Phase
**Purpose:** Volume profile calculation for overnight session

- **Duration:** 3:00pm (previous day) - 6:29am (current day)
- **Included in:** Overnight volume profile (VAH/VAL/POC calculation)
- **Excluded from:** RTH episode analysis, state tracking
- **Rationale:** Different liquidity and participant mix; tracked separately

**Use case:** Compare overnight value establishment vs prior RTH value to understand gap behavior and opening auction context.

---

## Inputs

### Required Inputs
- **1-minute OHLCV data** (raw input)
- Date range: **2023-2025 only**
- File: `MNQ_1min_2023_2025.csv` (or similar)

### Configuration Parameters
- **Timeframe**: 5m | 15m | 30m (default: 5m)
  - Determines resampling frequency and acceptance threshold conversion
- **ATR window**: Integer (default: 5)
  - Number of bars for ATR calculation on resampled data
- **Opening window**: Integer minutes (default: 15)
  - Duration of Opening Range period starting at 6:30am
- **Acceptance threshold**: Integer minutes (default: 30)
  - Minimum sustained duration for Acceptance outcome
- **Two-sided threshold**: Integer minutes (default: 30)
  - Window for detecting opposite boundary touch (triggers rejection)

### Derived Inputs (computed by scripts)
- **Prior RTH Session** (6:30am - 1:00pm previous day):
  - Prior RTH VAH / VAL / POC
  - Prior RTH High / Low
- **Overnight Session** (3:00pm previous day - 6:29am current day):
  - Overnight VAH / VAL / POC
  - Overnight High / Low
- **Opening Range** (first N minutes of RTH):
  - ORH / ORL
  - OR_range = ORH - ORL
  - OR position classifications
- **Initial Balance** (6:30am - 7:25am):
  - IBH / IBL / IB_range
- **ATR(window)** on resampled bars
- **Anchored VWAP** (anchored to 6:30am RTH open)

---

## Script Registry (Phase 1)

| Script | Status | Purpose | Depends On |
|--------|--------|---------|------------|
| `load_data.py` | ✅ Required | Load 1-min data, resample to target timeframe, filter to 2023-2025 | Raw data file |
| `session_profile.py` | ✅ Required | Compute Prior RTH and Overnight VAH/VAL/POC, IB | `load_data.py` |
| `opening_range.py` | ✅ Required | Compute OR, classify position relative to value areas | `load_data.py`, `session_profile.py` |
| `vwap_engine.py` | ✅ Required | Anchored VWAP (6:30am anchor) | `load_data.py` |
| `volatility.py` | ✅ Required | ATR(window) on resampled bars | `load_data.py` |
| `state_engine.py` | ✅ Required | Encode Balance/Discovery states, episode tracking | All above |
| `episode_logger.py` | ✅ Required | Persist episode-level summaries | `state_engine.py` |
| `session_logger.py` | ✅ Required | Persist session-level outcomes | `state_engine.py` |
| `transition_counter.py` | ✅ Required | Count context→outcome transitions (raw frequencies) | `episode_logger.py` |
| `tests_pipeline.py` | ✅ Required | Verify scripts executed correctly | All scripts |
| `readme_generator.py` | ✅ Required | Generate README.md from Agent2.md | Agent2.md |
| `run_pipeline.py` | ✅ Required | Execute full pipeline end-to-end | All scripts |

### Phase 2 Scripts (Deferred)

| Script | Status | Purpose | Phase |
|--------|--------|---------|-------|
| `markov_matrix.py` | ⏸️ Phase 2a | Convert transition counts to probabilities | Phase 2a |
| `session_predictor.py` | ⏸️ Phase 2b | Aggregate episode predictions to session level | Phase 2b |
| `model_validator.py` | ⏸️ Phase 2c | Backtest and calibration checks | Phase 2c |
| `regime_detector.py` | ⏸️ Phase 2d | Detect structural breaks, sliding window | Phase 2d |
| `alternative_models.py` | ⏸️ Phase 2e | RF/LR comparison models | Phase 2e |
| `exploration.ipynb` | ⏸️ Phase 2 | Visualizations and EDA | Phase 2 |

---

## Testing Responsibilities (Phase 1 Only)

The Agent **must not produce final outputs unless all Phase 1 tests pass**.

### Pipeline Tests (Must Pass)

- ✅ Data loads without gaps or ordering errors
- ✅ Data resampled correctly to target timeframe
- ✅ Data filtered to 2023-2025 range only
- ✅ Sessions detected correctly (~934 sessions expected @ 5m)
- ✅ Prior RTH VAH/VAL/POC exist for all sessions
- ✅ Overnight VAH/VAL/POC exist for all sessions
- ✅ Opening Range computed for all sessions
- ✅ IB values exist for sessions with pre-IB bars
- ✅ ATR(window) series is non-null and positive
- ✅ VWAP series is non-null
- ✅ Episode counts non-zero for both phases
- ✅ Episode-level summaries contain terminal outcomes (A, A*, R, B)
- ✅ Session-level outcomes logged for all sessions
- ✅ Transition count tables are non-empty
- ✅ failure_count values are {0, 1, 2, 3, 4} only
- ✅ Two-sided rejection logic applied (episodes with duration <30 min and outcome=R)
- ✅ Acceptance threshold applied correctly (duration ≥30 min for outcome=A)
- ✅ VWAP reset logic implemented (failure_count=0 after cross)
- ✅ Output files written successfully

**Failure indicates a bug or misconfiguration**, not a market insight.

**Phase 2 tests (model validation) are intentionally excluded from Phase 1.**

---

## Outputs (Phase 1)

### Generated Artifacts
```
output/
├── cleaned_data.csv                      # Validated, resampled bar data (2023-2025)
├── session_profiles_prior_rth.csv        # Prior RTH VAH/VAL/POC (6:30am-1pm)
├── session_profiles_overnight.csv        # Overnight VAH/VAL/POC (3pm-6:29am)
├── opening_range_analysis.csv            # OR metrics and classifications (separate context)
├── vwap_data.csv                         # Bar-level VWAP
├── volatility_data.csv                   # Bar-level ATR(window)
├── episodes_pre_ib.csv                   # Pre-IB episode summaries
├── episodes_post_ib.csv                  # Post-IB episode summaries
├── session_outcomes.csv                  # Session-level classifications
├── transition_counts_pre_ib.csv          # Context→outcome frequency table (pre-IB)
├── transition_counts_post_ib.csv         # Context→outcome frequency table (post-IB)
├── extension_distributions.csv           # Extension stats by outcome type
├── pipeline_test_report.txt              # Test results
└── README.md                             # Generated user documentation
```

---

### Episode Summary Schema

**Files:** `episodes_pre_ib.csv`, `episodes_post_ib.csv`

Each row represents one complete episode:

```csv
session_date,episode_id,state_type,direction,failure_count_at_start,
start_time,end_time,duration_minutes,
max_extension_points,max_extension_atr,max_extension_ib,max_extension_or,
max_retracement_points,
acceptance_achieved,time_in_acceptance_minutes,
terminal_outcome,phase,
avg_atr_during_episode,ib_range,or_range,reference_boundary
```

**Key Fields:**

**failure_count_at_start:** {0, 1, 2, 3, 4} - failure count when episode began

**max_extension_points:** Raw distance from boundary (points)

**max_extension_atr:** max_extension / avg_atr_during_episode

**max_extension_ib:** max_extension / IB_range (NULL for pre-IB episodes)

**max_extension_or:** max_extension / OR_range (all episodes, OR_range constant per session)

**acceptance_achieved:** Boolean - TRUE if sustained ≥30 min at any point

**time_in_acceptance_minutes:** Duration in acceptance state before rejection/session end (0 if not achieved)

**terminal_outcome:**
- **A**: Acceptance (sustained ≥30 min, ended outside)
- **A***: Incomplete (session ended in discovery, <30 min)
- **R**: Rejection (returned inside OR opposite touched within 30 min)
- **B**: Balance (only for balance episodes)

**avg_atr_during_episode:** Average ATR across episode bars (for extension normalization)

**or_range:** Session's Opening Range (ORH - ORL), computed once at opening window end

---

### Session Outcome Schema

**File:** `session_outcomes.csv`

Each row represents one trading session:

```csv
session_date,session_classification,
discovery_up_attempts,discovery_down_attempts,
discovery_up_acceptances,discovery_down_acceptances,
discovery_up_rejections,discovery_down_rejections,
max_extension_up_points,max_extension_down_points,
time_in_balance_minutes,time_in_discovery_minutes,
final_failure_count_up,final_failure_count_down,
closing_price,prior_rth_vah,prior_rth_val,
overnight_vah,overnight_val,ibh,ibl,
close_vs_prior_rth,close_vs_overnight,close_vs_ib
```

**Session Classifications:**

**Directional:**
- `net_discovery_up`: Accepted upside, ended above value
- `net_discovery_down`: Accepted downside, ended below value
- `two_sided_discovery`: Accepted both directions

**Failed Directional:**
- `failed_up_balance`: Any upside rejection, remained in value
- `failed_down_balance`: Any downside rejection, remained in value
- `failed_two_sided_balance`: Rejected both directions, remained in value

**Rotation:**
- `failed_discovery_up`: Rejected downside, rotated to upside acceptance
- `failed_discovery_down`: Rejected upside, rotated to downside acceptance

**Pure Balance:**
- `balanced`: No acceptances, stayed in value, minimal rejections

---

### Opening Range Analysis Schema

**File:** `opening_range_analysis.csv` (separate context file)

Each row represents one session's opening range metrics:

```csv
session_date,opening_window_minutes,
orh,orl,or_range,
prior_rth_vah,prior_rth_val,prior_rth_poc,
overnight_vah,overnight_val,overnight_poc,
or_vs_prior_rth,or_vs_overnight,
retest_prior_rth_during_preib,
price_at_ib_vs_prior_rth,
ibh,ibl,ib_range
```

**OR Position Classifications:**
- `above_vah`: OR formed above value area high
- `within_value`: OR within VAH-VAL range
- `below_val`: OR formed below value area low
- `at_vah`: OR straddling VAH
- `at_val`: OR straddling VAL

**Purpose:** Context for understanding opening auction dynamics, separate from episode tracking.

---

### Transition Count Schema

**Files:** `transition_counts_pre_ib.csv`, `transition_counts_post_ib.csv`

Each row represents one discrete context state and its observed outcomes:

```csv
direction,failure_count,extension_bin,phase,
count_acceptance,count_acceptance_incomplete,count_rejection,
total_observations,avg_duration_minutes
```

**extension_bin values:**
- `0.0-0.5` (ATR units)
- `0.5-1.0`
- `1.0-1.5`
- `1.5-2.0`
- `2.0+`

**Example row:**
```
up,1,1.0-1.5,post_ib,12,3,48,63,42.5
```

**Interpretation:**
- Context: Discovery Up, failure_count=1, extension 1.0-1.5 ATR, post-IB phase
- Observed 63 episodes total with this context
- 12 ended in Acceptance (A)
- 3 ended in Incomplete Acceptance (A*)
- 48 ended in Rejection (R)
- Average episode duration: 42.5 minutes

**Phase 2 usage:** Normalize by total_observations to get P(A), P(A*), P(R).

---

## Data Quality Requirements

### Time Frame
- **Input**: 1-minute bars
- **Resampled to**: 5m | 15m | 30m (configurable)
- **Default**: 5-minute bars
- **2023-2025 data only** (filtered during load)
- Expected (5m): ~934 sessions, ~212,000 bars

### OHLCV Aggregation Rules
When resampling from 1-minute to target timeframe:
- **O** (Open): First 1-min bar's open
- **H** (High): Maximum of all 1-min bar highs
- **L** (Low): Minimum of all 1-min bar lows
- **C** (Close): Last 1-min bar's close
- **V** (Volume): Sum of all 1-min bar volumes

### Session Definition

**Prior RTH Session: 6:30am - 1:00pm (previous trading day)**
- Duration: ~6.5 hours
- Purpose: Establish "prior session value" (VAH/VAL/POC)
- Volume profile: RTH-only bars

**Overnight Session: 3:00pm (previous day) - 6:29am (current day)**
- Duration: ~15.5 hours
- Purpose: Establish "overnight value" (VAH/VAL/POC)
- Volume profile: Overnight-only bars
- **Critical:** Ends at 6:29am to exclude 6:30am RTH open bar

**Current Trading Day: 6:30am onwards**
- Pre-IB: 6:30am - 7:25am (IB formation)
- Post-IB: 7:30am - 12:55pm (main session)

### Opening Range Definition
**Opening Window**: First N minutes of RTH (default: 15 min, configurable)
- **Start**: 6:30am
- **End**: 6:30am + N minutes (e.g., 6:45am for 15-min OR)
- **ORH**: Highest high during window
- **ORL**: Lowest low during window
- **OR_range**: ORH - ORL (constant for entire session, used in extension_or normalization)

### Phase Assignment
Each bar assigned to exactly one session period:
- **prior_rth**: 6:30am - 1:00pm (previous day)
- **overnight**: 3:00pm (previous day) - 6:29am (current day)
- **pre_ib**: 6:30am - 7:25am (current day, IB formation)
- **post_ib**: 7:30am - 12:55pm (current day, main session)

---

## State Machine Logic

### Balance State
- **Entry**: Price closes within reference boundaries (VAH/VAL or IBH/IBL)
- **Tracking**: Start time, duration
- **VWAP Cross Tracking**: Set flags when VWAP crossed (for failure_count reset)
- **Exit**: Price closes outside boundary → start Discovery
- **Terminal**: Episode closes with outcome "B" when discovery starts or phase ends

### Discovery State

**Episode Start:**
- Bar closes outside reference boundary
- Bar's high (for Discovery Up) or low (for Discovery Down) touched the boundary
- **Check failure_count reset:** If VWAP crossed since last episode in this direction, reset failure_count to 0
- **Start timers:** 30-min acceptance timer, 30-min two-sided rejection timer
- **Initialize tracking:** max_extension=0, acceptance_achieved=FALSE, time_in_acceptance=0

**During Episode (each bar):**
- Update max_extension if current extension exceeds prior maximum
- Update max_retracement if pullback from peak
- **Check acceptance:** If duration ≥30 min and still outside boundary:
  - Set acceptance_achieved = TRUE
  - Start counting time_in_acceptance
- **Check two-sided rejection:** If opposite boundary touched (high/low) and duration <30 min since episode start:
  - Close episode with outcome R (two-sided rejection)
  - Log max_extension, max_retracement for the episode window
  - Increment failure_count
  - Immediately start opposite direction discovery (if close triggers it)

**Termination Conditions:**

**Rejection (R):**
- Price closes back inside reference boundary before 30 minutes elapsed
- **OR** opposite boundary touched within 30 minutes of episode start (two-sided rejection)
- Log episode with outcome R
- Increment failure_count for this direction
- Return to Balance state

**Acceptance (A):**
- Duration ≥30 minutes, price sustained outside boundary
- Session or phase ended while still outside boundary
- Log episode with outcome A
- **Do NOT increment failure_count** (acceptance does not increase failures)

**Acceptance-Incomplete (A*):**
- Session ended while in discovery state
- Duration <30 minutes (had not yet reached acceptance threshold)
- Still outside boundary at session end
- Log episode with outcome A*
- **Do NOT increment failure_count**

**Phase Transition (7:30am):**
- If still outside new reference (IB): Close pre-IB episode (A or A*), start post-IB episode
- If now inside IB: Close episode as R, increment failure_count, enter Balance

### Failure Count Tracking

**Increment:**
- Only on Rejection (R) outcomes
- Separate counters for up and down directions
- Values: 0 → 1 → 2 → 3 → 4+ (caps at 4)

**Reset (VWAP Cross - Option B):**
- **Only at new episode start**, not mid-episode
- When starting Discovery episode in direction X:
  - Check: Has VWAP been crossed since last episode in direction X ended?
  - If YES: failure_count_X = 0
  - If NO: use existing failure_count_X
- **Implementation:** Requires tracking boolean flags:
  - `vwap_crossed_since_last_up_episode`
  - `vwap_crossed_since_last_down_episode`
  - Flags set when VWAP crossed while in Balance
  - Flags consumed when new episode starts, then reset to FALSE

**Persistence:**
- failure_count carries over from pre-IB to post-IB (does NOT reset at phase transition)
- failure_count persists across balance episodes within same phase
- Only VWAP cross can reset failure_count to 0

**Example:**
```
Session start: failure_count_up=0, failure_count_down=0, flags=FALSE

10:00 - Discovery Up → R (rejection) → failure_count_up=1, enter Balance
10:15 - In Balance, price crosses VWAP → vwap_crossed_since_last_up_episode=TRUE
10:30 - Discovery Up starts:
        Check flag: TRUE
        Reset failure_count_up=0
        Set flag=FALSE
10:45 - Discovery Up → A (acceptance, benefited from VWAP reset)
```

---

## Design Principles

- **Explicit time frame**: 1-minute input, resampled to 5m/15m/30m, 2023-2025 only
- **Parameterized**: Timeframe, ATR window, opening window, acceptance threshold all configurable
- **Session separation**: Prior RTH vs Overnight tracked separately
- **Episode-level granularity**: No bar-level logging, summaries only (lighter weight)
- **Deterministic state logic**: No randomness, fully reproducible
- **Empirical foundation**: Phase 1 counts observations, Phase 2 computes probabilities
- **Strong separation**: Data collection (Phase 1) vs modeling (Phase 2)
- **Unlimited attempts**: No artificial cap, failure_count can reach 4+
- **30-minute thresholds**: Acceptance and two-sided rejection use same duration (consistent)
- **VWAP reset**: Only at episode start (Option B), prevents mid-episode confusion
- **Acceptance independence**: failure_count is context, not barrier to acceptance
- **Tests before trust**: Pipeline tests must pass before results considered valid
- **No trading decisions**: Research/feature engineering only

---

## Phase 1 Definition of "Done"

Phase 1 is complete when:

✅ Full pipeline runs end-to-end  
✅ Data loaded from 1-minute bars and resampled to target timeframe  
✅ Data filtered to 2023-2025 only  
✅ All pipeline tests pass (18+ tests)  
✅ Prior RTH and Overnight volume profiles computed separately  
✅ Opening Range analysis computed and saved to separate CSV  
✅ Episode summaries logged for both phases (pre-IB, post-IB)  
✅ Session-level outcomes logged for all sessions  
✅ Transition count tables generated (context→outcome frequencies)  
✅ failure_count capped at 4+ (never exceeds)  
✅ Two-sided rejection logic implemented (30-min window)  
✅ Acceptance threshold (30 min) applied correctly  
✅ VWAP reset logic implemented (Option B - at episode start only)  
✅ Extension metrics include: points, ATR, IB, OR  
✅ acceptance_achieved and time_in_acceptance tracked  
✅ ATR(window) configurable and tested  
✅ README.md generated from Agent2.md  
✅ Outputs reproducible on re-run  

**Phase 2 (Markov matrix, prediction, validation) is explicitly out of scope for Phase 1 completion.**

---

## Configuration Reference

```python
# Default configuration (user-adjustable)
CONFIG = {
    'timeframe': '5m',                    # Options: '5m', '15m', '30m'
    'atr_window': 5,                      # ATR lookback period (bars)
    'opening_window_minutes': 15,         # Opening Range duration
    'acceptance_threshold_minutes': 30,   # Minimum sustained duration for acceptance
    'two_sided_threshold_minutes': 30,    # Window for opposite boundary touch rejection
    'data_start_year': 2023,              # Start of analysis window
    'data_end_year': 2025,                # End of analysis window
}
```

**Acceptance threshold conversion:**
- 5-minute timeframe: 30 min = 6 bars
- 15-minute timeframe: 30 min = 2 bars
- 30-minute timeframe: 30 min = 1 bar

**Two-sided threshold conversion:**
- Same as acceptance threshold (30 min default)

**ATR window guidance:**
- Smaller window (3-5): More responsive to recent volatility
- Larger window (10-14): Smoother, less reactive
- Default 5: Balances responsiveness with stability for intraday

**Opening window guidance:**
- Shorter window (5-10 min): Captures initial auction quickly
- Standard window (15 min): Industry typical
- Longer window (30 min): More stable OR, but may miss early breakouts

---

## Running the Pipeline

```bash
# Install dependencies
pip install pandas numpy

# Configure at top of run_pipeline.py:
TIMEFRAME = "5m"                          # Resample to 5-minute bars
ATR_WINDOW = 5                            # ATR lookback period
OPENING_WINDOW_MINUTES = 15               # Opening Range duration
ACCEPTANCE_THRESHOLD_MINUTES = 30         # Sustained discovery threshold
TWO_SIDED_THRESHOLD_MINUTES = 30          # Opposite boundary touch window

# Run Phase 1 pipeline
python run_pipeline.py

# Expected execution time: ~5-8 minutes (varies by timeframe)
# Expected output: All pipeline tests pass ✅
```

---

**Last Updated:** 2026-02-13  
**Version:** 2.0 (Final - Locked In)
**Status:** Ready for Implementation
