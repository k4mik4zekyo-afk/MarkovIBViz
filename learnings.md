# Visualization Learnings & Design Decisions

Design notes from building the Plotly candlestick + Markov episode overlay tools.

## Scripts

| Script | Purpose | View |
|--------|---------|------|
| `candle_episode_viz.py` | Post-IB discovery episodes | Single day full RTH (pre-IB + post-IB candles) |
| `pre_ib_episode_viz.py` | Pre-IB discovery episodes | Prior day RTH + overnight + current day RTH |

---

## Plotly-Specific Learnings

### Candlestick range slider conflict
Plotly's `go.Candlestick` automatically adds a range slider on its x-axis. When adding a second range slider on the volume subplot's x-axis (`xaxis2_rangeslider`), both appear. Fix: explicitly set `xaxis_rangeslider_visible=False` on the candlestick axis.

### Horizontal lines and toggleability
`fig.add_hline()` creates layout shapes that cannot be toggled via `updatemenus` visibility arrays. To make IBH/IBL lines toggleable (prior vs current day), use `go.Scatter` traces with `mode="lines+text"` instead. These traces can be toggled via visibility arrays in `updatemenus` buttons.

### Hover text customization
Setting `hoverinfo="text"` with a custom `text` array replaces all default hover content, including the timestamp. Always include the bar's datetime explicitly in the custom hover string. Format with `strftime("%Y-%m-%d %H:%M")` for clarity.

### Subplot axis references
With `shared_xaxes=True` and 2-row subplots, the top row uses `xaxis`/`yaxis` and the bottom row uses `xaxis2`/`yaxis2`. Annotations that need to reference the shared x-axis from outside both subplots use `xref="x2"` and `yref="paper"`.

### X-axis time formatting
For hourly tick marks on datetime axes, set `dtick=3600000` (milliseconds) and `tickformat="%H:%M"`. Apply to both `row=1` and `row=2` for consistency.

### Y-axis interactivity
By default, Plotly axes are interactive (drag to zoom/pan). Setting an explicit `range=[...]` locks the initial view but users can still zoom. Setting `fixedrange=False` explicitly ensures the axis remains interactive.

---

## Layout Decisions

### Chart dimensions
- **Width**: 1800px works well for full-day 5-minute candles (~78 bars for single RTH, ~200+ for two-day view).
- **Height**: 1200px is the sweet spot. 750px was too small to see episode detail; 1600px was too tall for most screens.

### Row heights
`row_heights=[0.78, 0.22]` gives enough room for volume while keeping the price chart dominant.

### Episode annotations: floating labels vs HTML table
**Floating labels** (positioned below volume using `yref="paper"`) work for 1-3 episodes but cause overlap and clutter on busy days (e.g., May 2nd with D0-D3 escalation). **HTML table beneath the chart** scales to any number of episodes and provides a scannable format with sortable columns. The table is appended as raw HTML below the Plotly div in the output file.

### Summary panel position
Top-right corner (`xref="paper", x=1.0, y=1.0`) with monospace font and semi-transparent background keeps the summary visible without obscuring candle data.

### Weekend/holiday prior session lookup
`_get_prior_session_date` (simple index-1 lookup) returns the wrong session on Mondays: it finds Sunday, which only has overnight bars and no RTH. Fix: walk backward through session dates to find the most recent one with `pre_ib` or `post_ib` bars. This correctly returns Friday for Monday sessions and handles holidays.

### Overnight bars across weekends
On weekends, overnight bars span multiple `session_date` values (e.g., Friday's overnight on Friday's session, Sunday's overnight on Sunday's session). Collecting overnight from only the prior session_date misses the gap. Fix: gather overnight bars from all session dates between the prior RTH date and the current date (`all_dates[prior_idx:curr_idx]`).

### ATR in hover text
The `cleaned_data.csv` does not include ATR — it lives in `volatility_data.csv` (columns: `datetime`, `atr_5`, `atr_14`). Merge by datetime in `load_data()` to make ATR available for hover text. Use `CONFIG['atr_window']` to pick the correct column name (`atr_5` for default config).

### Opening window Fibonacci levels
Fib levels (0%, 25%, 50%, 75%, 100%) drawn from the high/low of the first `opening_window_minutes` bars in the current session. Extended as dotted horizontal lines from the last opening window bar to the chart's right edge using `go.Scatter` traces (per the toggleability learning — these are traces, not layout shapes). Lines use `hoverinfo="skip"` to avoid cluttering the unified hover tooltip.

---

## Data & Domain Learnings

### Episode structure
- **Balance**: No probe outside reference boundary. Skipped in annotations/table.
- **Discovery**: Price probes outside IBH/IBL (post-IB) or prior VAH/VAL (pre-IB).
- **D-state**: D0 = first attempt, D1 = one prior rejection in same direction, D2+ = escalation.
- **Terminal outcomes**: A (acceptance, 30+ min outside), A* (incomplete acceptance), R (rejection, close back inside), B (balance), C (continuation at session end).

### Rejection semantics
A single bar closing back inside the reference boundary triggers rejection (R). This means a 235-minute episode with `acceptance_achieved=True` can still end as R if price eventually closes back inside. The rejection trigger is immediate on the closing bar.

### Overnight candle importance
Overnight bars (between RTH sessions) are critical for validating overnight high/low levels and determining if overnight activity is "repairing" the prior day's range. Always include in multi-day views.

### Pre-IB reference boundaries
Pre-IB episodes use **prior RTH VAH/VAL** as reference boundaries, not IBH/IBL. The current day's IB hasn't formed yet during the pre-IB phase (6:30-7:25am PT).

---

## File Output Strategy

### Custom HTML template
Instead of `fig.write_html()`, use `fig.to_html(full_html=False, include_plotlyjs="cdn")` to get just the Plotly div, then wrap it in a custom HTML template that includes the episode table below. This approach:
- Keeps the Plotly chart interactive
- Adds the episode data table as standard HTML beneath the chart
- Uses CDN for Plotly.js (smaller file size)
- Allows custom CSS for table styling (hover highlights, alternating row colors)

### Interactive mode
For `fig.show()` equivalent with the HTML table, write to a temp file and open with `webbrowser.open()`. This preserves the combined chart+table experience.
