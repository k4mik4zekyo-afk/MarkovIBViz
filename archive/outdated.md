# Agent.md  
**Agent Name:** Value Discovery State Agent

## Purpose

This Agent analyzes auction behavior using **5-minute market data** to empirically estimate how markets transition between balance and discovery.

It encodes discovery as a **state-based process**, logs outcomes across multi-year data, computes transition probabilities, and validates both:
1. **Market structure assumptions**, and  
2. **Pipeline correctness**, before results are considered valid.

The Agent is the authoritative specification for the entire pipeline and is responsible for generating a user-facing `README.md`.

---

## Operating Assumptions

- Input data is **5-minute OHLCV**
- Data spans multiple years
- Sessions are well-defined and continuous
- No lower-timeframe data is used or referenced
- No predictive trading decisions are made

---

## Core Concepts

### Discovery States

Each discovery attempt is classified into one of three terminal outcomes:

- **A — Acceptance**  
  Price extends and sustains outside prior value.

- **R — Rejection**  
  Price returns into prior value and invalidates discovery.

- **C — Continuation**  
  Price remains in discovery without resolution.

Discovery persistence is tracked as `D₀, D₁, D₂, …` based on the number of failed attempts.

---

### Episode Definition

An **episode** is a contiguous discovery sequence consisting of:
- Direction (up / down)
- Failure count
- Maximum extension
- Maximum retracement
- Terminal outcome (A / R / C)

Episodes reset when price re-enters balance or resolves.

---

## Inputs

### Required Inputs
- 5-minute OHLCV data

### Derived Inputs (computed by scripts)
- Prior Day VAH / VAL / POC
- Prior Day High / Low
- Prior Day normalized volume
- Anchored VWAP and standard deviation bands
- ATR(14) computed on 5-minute bars

No external indicators are assumed to be precomputed.

---

## Script Registry

| Script | Status | Purpose | Depends On |
|------|-------|--------|-----------|
| `load_data.py` | Required | Load and validate 5-minute data | Raw data file |
| `session_profile.py` | Required | Compute prior-day VAH/VAL/POC | `load_data.py` |
| `vwap_engine.py` | Required | Anchored VWAP + bands | `load_data.py` |
| `volatility.py` | Required | ATR(14) on 5-minute bars | `load_data.py` |
| `state_engine.py` | Required | Encode discovery states and episodes | All above |
| `episode_logger.py` | Required | Persist episode-level data | `state_engine.py` |
| `probability_engine.py` | Required | Compute empirical probabilities | `episode_logger.py` |
| `tests_pipeline.py` | Required | Verify scripts executed correctly | All scripts |
| `tests_structural.py` | Required | Validate market structure assumptions | `probability_engine.py` |
| `readme_generator.py` | Required | Generate README.md from Agent.md | Agent.md |

---

## Testing Responsibilities (Critical)

The Agent **must not produce final outputs unless all tests pass**.

### 1. Pipeline Tests (Must Pass)

These tests confirm the system is functioning correctly:

- Data loads without gaps or ordering errors
- Sessions are detected correctly
- VAH/VAL/POC exist for all sessions
- VWAP and ATR series are non-null
- Episodes are logged (non-zero count)
- Probability tables are non-empty
- Output files are written successfully

Failure here indicates a **bug or misconfiguration**, not a market insight.

---

### 2. Structural Tests (Research Validity)

These tests validate the logical consistency of results:

- Acceptance probability vs failure count behaves monotonically
- Rejection probability increases after repeated failures
- Extension distributions widen with persistence
- Upside and downside statistics are approximately symmetric
- Sample sizes per state exceed minimum thresholds

Failure here indicates **invalid assumptions or insufficient data**, not code errors.

---

## Outputs

### Generated Artifacts
- Episode log (per discovery sequence)
- Transition probability tables
- Structural test summaries
- Pipeline test report
- Auto-generated `README.md`

---

## README.md Generation

The Agent **must generate** a `README.md` that includes:

- Project overview
- Data requirements
- How to run the pipeline
- Description of outputs
- Explanation of testing philosophy
- Clear statement that results are empirical and will be used for prediction

`README.md` is treated as a **derived artifact**, not manually maintained.

---

## Notebook Usage

- Should be able to visualize end outcomes from the structural tests

---

## Design Principles

- Explicit time frame (5-minute only)
- Deterministic state logic
- Empirical probabilities over predictions
- Strong separation between computation and interpretation
- Tests before trust

---

## Definition of “Done”

The Agent is considered complete when:
- The full pipeline runs end-to-end
- All pipeline tests pass
- Structural tests are reported
- README.md is generated
- Outputs are reproducible on re-run
