"""
pre_ib_episode_viz.py — Plotly candlestick visualization with pre-IB Markov episode overlays.

Displays a two-day view: prior day's full RTH session + overnight + current day's
pre-IB window (6:30–7:25am) with pre-IB episode overlays including:
- Prior RTH VAH / VAL reference boundaries (pre-IB discovery boundaries)
- Prior RTH POC
- Overnight high / low
- IBH / IBL lines (formed at end of pre-IB window)
- Phase separators (prior RTH end, overnight, RTH open)
- Episode shading with color-coded backgrounds
- D-state + failure count annotations
- Transition arrows on D-state escalation
- Terminal outcome markers (A, A*, R, B)

Usage:
    python pre_ib_episode_viz.py                        # auto-select interesting date
    python pre_ib_episode_viz.py 2025-05-09             # specific date
    python pre_ib_episode_viz.py 2025-05-09 --save      # save to HTML file
    python pre_ib_episode_viz.py --list                  # list dates with D1+ activity
"""

import argparse
import sys
import os
from datetime import time, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CONFIG, SESSION_TIMES

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Colour palette (shared with candle_episode_viz.py) ──────────────────────
OUTCOME_COLORS = {
    "A":  "rgba(16, 185, 129, 0.18)",
    "A*": "rgba(56, 189, 248, 0.18)",
    "R":  "rgba(239, 68, 68, 0.12)",
    "B":  "rgba(148, 163, 184, 0.10)",
    "C":  "rgba(251, 191, 36, 0.18)",
}
OUTCOME_BORDER = {
    "A":  "rgba(16, 185, 129, 0.55)",
    "A*": "rgba(56, 189, 248, 0.55)",
    "R":  "rgba(239, 68, 68, 0.40)",
    "B":  "rgba(148, 163, 184, 0.25)",
    "C":  "rgba(251, 191, 36, 0.55)",
}
OUTCOME_TEXT = {
    "A":  "#059669",
    "A*": "#0284c7",
    "R":  "#dc2626",
    "B":  "#64748b",
    "C":  "#d97706",
}
DIRECTION_MARKER = {
    "up":   "triangle-up",
    "down": "triangle-down",
}

# Colours for Opening Range Fibonacci levels (from low to high)
FIB_COLORS = {
    0.0:  "#e11d48",   # rose-600  (OR low)
    0.25: "#f97316",   # orange-500
    0.50: "#eab308",   # yellow-500
    0.75: "#22c55e",   # green-500
    1.0:  "#06b6d4",   # cyan-500  (OR high)
}
FIB_LABELS = {0.0: "0%", 0.25: "25%", 0.50: "50%", 0.75: "75%", 1.0: "100%"}


def load_data():
    """Load cleaned candle data, pre-IB episode log, session profiles, and ATR."""
    candles = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), parse_dates=["datetime"])
    episodes = pd.read_csv(os.path.join(OUTPUT_DIR, "episode_log_pre_ib.csv"), parse_dates=["start_time", "end_time"])
    profiles = pd.read_csv(os.path.join(OUTPUT_DIR, "session_profiles.csv"))

    # Merge ATR data onto candles
    vol_path = os.path.join(OUTPUT_DIR, "volatility_data.csv")
    if os.path.exists(vol_path):
        vol = pd.read_csv(vol_path, parse_dates=["datetime"])
        atr_col = [c for c in vol.columns if c.startswith("atr_") and c != "atr_14"]
        atr_col = atr_col[0] if atr_col else "atr_5"
        candles = candles.merge(vol[["datetime", atr_col]], on="datetime", how="left")
        if atr_col != "atr":
            candles.rename(columns={atr_col: "atr"}, inplace=True)
    else:
        candles["atr"] = np.nan

    return candles, episodes, profiles


def available_dates(episodes):
    """Return sorted list of dates that have pre-IB episodes."""
    return sorted(episodes["session_date"].unique())


def _get_prior_session_date(session_date, profiles, candles=None):
    """Get the prior session date that has RTH bars (pre_ib or post_ib).

    On Mondays, the immediately preceding session_date is Sunday which has no
    RTH bars (futures open Sunday evening with overnight only).  Walk backward
    through the sorted date list until a date with RTH data is found.

    When *candles* is provided the check is authoritative (look for actual
    pre_ib/post_ib rows).  Otherwise fall back to the profile-level
    prior_rth_vah column as a proxy.
    """
    all_dates = sorted(profiles["session_date"].unique())
    try:
        idx = all_dates.index(session_date)
    except ValueError:
        return None
    if idx == 0:
        return None

    # Walk backward to find a date with actual RTH bars
    for i in range(idx - 1, -1, -1):
        candidate = all_dates[i]
        if candles is not None:
            rth_bars = candles[
                (candles["session_date"] == candidate) &
                (candles["phase"].isin(["pre_ib", "post_ib"]))
            ]
            if not rth_bars.empty:
                return candidate
        else:
            # Fallback: check profile has prior RTH data computed
            prof_row = profiles[profiles["session_date"] == candidate]
            if not prof_row.empty and pd.notna(prof_row.iloc[0].get("ibh")):
                return candidate
    return None


def build_figure(session_date, candles, episodes, profiles):
    """Build the Plotly figure for a two-day view centred on the pre-IB window."""

    prior_date = _get_prior_session_date(session_date, profiles, candles)
    day_profile = profiles[profiles["session_date"] == session_date]
    if day_profile.empty:
        print(f"No session profile for {session_date}.")
        sys.exit(1)

    prof = day_profile.iloc[0]
    prior_vah = prof["prior_rth_vah"] if pd.notna(prof.get("prior_rth_vah")) else np.nan
    prior_val = prof["prior_rth_val"] if pd.notna(prof.get("prior_rth_val")) else np.nan
    prior_poc = prof["prior_rth_poc"] if pd.notna(prof.get("prior_rth_poc")) else np.nan
    ibh = prof["ibh"] if pd.notna(prof.get("ibh")) else np.nan
    ibl = prof["ibl"] if pd.notna(prof.get("ibl")) else np.nan
    on_high = prof["overnight_high"] if pd.notna(prof.get("overnight_high")) else np.nan
    on_low = prof["overnight_low"] if pd.notna(prof.get("overnight_low")) else np.nan

    # ── Gather candles: prior day RTH + overnight + current pre-IB ──────────
    frames = []

    # For the overnight bars we need the *immediate* predecessor session_date
    # (e.g. Sunday for Monday) which may differ from prior_date when that was
    # pushed back to find actual RTH bars (e.g. Friday).
    all_dates = sorted(profiles["session_date"].unique())
    try:
        cur_idx = all_dates.index(session_date)
    except ValueError:
        cur_idx = 0
    immediate_prior = all_dates[cur_idx - 1] if cur_idx > 0 else None

    # Prior day: pre_ib + post_ib (the RTH bars) — uses the date with actual RTH
    if prior_date:
        prior_rth = candles[
            (candles["session_date"] == prior_date) &
            (candles["phase"].isin(["pre_ib", "post_ib"]))
        ].copy()
        prior_rth["section"] = "prior_rth"
        frames.append(prior_rth)

    # Overnight bars — collect from prior_date's overnight (post-RTH on that day)
    # PLUS any intermediate sessions' overnight (e.g. Sunday for weekend gaps)
    overnight_dates = set()
    if prior_date:
        overnight_dates.add(prior_date)
    if immediate_prior and immediate_prior != prior_date:
        overnight_dates.add(immediate_prior)
    if overnight_dates:
        overnight = candles[
            (candles["session_date"].isin(overnight_dates)) &
            (candles["phase"] == "overnight")
        ].copy()
        overnight["section"] = "overnight"
        frames.append(overnight)

    # Current day pre-IB
    current_pre = candles[
        (candles["session_date"] == session_date) &
        (candles["phase"] == "pre_ib")
    ].copy()
    current_pre["section"] = "pre_ib"
    frames.append(current_pre)

    if not frames:
        print(f"No candle data for {session_date} (or prior day).")
        sys.exit(1)

    all_candles = pd.concat(frames).sort_values("datetime").reset_index(drop=True)

    if all_candles.empty:
        print(f"No candle data assembled for {session_date}.")
        sys.exit(1)

    # ── Episodes ────────────────────────────────────────────────────────────
    day_episodes = episodes[episodes["session_date"] == session_date].copy().reset_index(drop=True)

    price_min = all_candles["low"].min()
    price_max = all_candles["high"].max()
    price_pad = (price_max - price_min) * 0.05

    # ── Create figure ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    # ── Episode background shading ──────────────────────────────────────────
    for _, ep in day_episodes.iterrows():
        outcome = ep["terminal_outcome"]
        fill = OUTCOME_COLORS.get(outcome, "rgba(200,200,200,0.1)")
        border = OUTCOME_BORDER.get(outcome, "rgba(200,200,200,0.3)")
        fig.add_vrect(
            x0=ep["start_time"], x1=ep["end_time"],
            fillcolor=fill,
            line=dict(color=border, width=1, dash="dot"),
            layer="below",
            row=1, col=1,
        )

    # ── Candlestick trace ───────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=all_candles["datetime"],
            open=all_candles["open"],
            high=all_candles["high"],
            low=all_candles["low"],
            close=all_candles["close"],
            name="MNQ 5m",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#ef4444",
            whiskerwidth=0.4,
        ),
        row=1, col=1,
    )

    # ── ATR hover trace (invisible line that shows ATR in unified tooltip) ──
    atr_vals = all_candles["atr"] if "atr" in all_candles.columns else pd.Series(np.nan, index=all_candles.index)
    fig.add_trace(
        go.Scatter(
            x=all_candles["datetime"],
            y=all_candles["close"],
            mode="markers",
            marker=dict(size=0, opacity=0),
            name="ATR",
            customdata=atr_vals,
            hovertemplate="ATR: %{customdata:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Volume bars ─────────────────────────────────────────────────────────
    colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(all_candles["close"], all_candles["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=all_candles["datetime"],
            y=all_candles["volume"],
            marker_color=colors,
            opacity=0.5,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Reference lines ─────────────────────────────────────────────────────
    # Prior VAH / VAL (the pre-IB discovery boundaries)
    if pd.notna(prior_vah):
        fig.add_hline(
            y=prior_vah, line=dict(color="#8b5cf6", width=1.5, dash="dash"),
            annotation_text=f"Prior VAH {prior_vah:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#8b5cf6", size=10),
            row=1, col=1,
        )
    if pd.notna(prior_val):
        fig.add_hline(
            y=prior_val, line=dict(color="#8b5cf6", width=1.5, dash="dash"),
            annotation_text=f"Prior VAL {prior_val:.2f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#8b5cf6", size=10),
            row=1, col=1,
        )
    if pd.notna(prior_poc):
        fig.add_hline(
            y=prior_poc, line=dict(color="#a78bfa", width=1, dash="dot"),
            annotation_text=f"Prior POC {prior_poc:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#a78bfa", size=9),
            row=1, col=1,
        )

    # IBH / IBL (formed at end of pre-IB)
    if pd.notna(ibh):
        fig.add_hline(
            y=ibh, line=dict(color="#3b82f6", width=1.5, dash="dash"),
            annotation_text=f"IBH {ibh:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#3b82f6", size=10),
            row=1, col=1,
        )
    if pd.notna(ibl):
        fig.add_hline(
            y=ibl, line=dict(color="#f59e0b", width=1.5, dash="dash"),
            annotation_text=f"IBL {ibl:.2f}",
            annotation_position="bottom right",
            annotation_font=dict(color="#f59e0b", size=10),
            row=1, col=1,
        )

    # Overnight high / low
    if pd.notna(on_high):
        fig.add_hline(
            y=on_high, line=dict(color="#6b7280", width=1, dash="dot"),
            annotation_text=f"ON High {on_high:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#6b7280", size=9),
            row=1, col=1,
        )
    if pd.notna(on_low):
        fig.add_hline(
            y=on_low, line=dict(color="#6b7280", width=1, dash="dot"),
            annotation_text=f"ON Low {on_low:.2f}",
            annotation_position="bottom right",
            annotation_font=dict(color="#6b7280", size=9),
            row=1, col=1,
        )

    # ── Opening Range Fibonacci levels ───────────────────────────────────────
    # Compute OR high/low from the first N minutes of the current session
    or_fib_minutes = CONFIG.get('opening_range_fib_minutes', 10)
    fib_levels = CONFIG.get('opening_range_fib_levels', [0.0, 0.25, 0.50, 0.75, 1.0])
    rth_start_time = time(*SESSION_TIMES['rth_start'])
    or_fib_end_time = (
        pd.Timestamp.combine(pd.Timestamp(session_date), rth_start_time)
        + timedelta(minutes=or_fib_minutes)
    ).time()

    or_fib_bars = current_pre[
        (current_pre["datetime"].dt.time >= rth_start_time) &
        (current_pre["datetime"].dt.time < or_fib_end_time)
    ]
    if not or_fib_bars.empty:
        or_high = or_fib_bars["high"].max()
        or_low = or_fib_bars["low"].min()
        or_range = or_high - or_low

        for level in fib_levels:
            fib_price = or_low + or_range * level
            color = FIB_COLORS.get(level, "#9ca3af")
            label = FIB_LABELS.get(level, f"{level:.0%}")
            fig.add_hline(
                y=fib_price,
                line=dict(color=color, width=1.5, dash="solid"),
                annotation_text=f"OR {label} {fib_price:.2f}",
                annotation_position="bottom left" if level < 0.5 else "top left",
                annotation_font=dict(color=color, size=9),
                row=1, col=1,
            )

    # ── Phase separators ────────────────────────────────────────────────────
    # Prior RTH end → overnight start
    if prior_date:
        prior_post = candles[
            (candles["session_date"] == prior_date) &
            (candles["phase"] == "post_ib")
        ]
        if not prior_post.empty:
            rth_end_time = prior_post["datetime"].max() + pd.Timedelta(minutes=5)
            fig.add_vline(
                x=rth_end_time,
                line=dict(color="#94a3b8", width=1, dash="dashdot"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=rth_end_time, y=price_max + price_pad * 0.9,
                text="Prior RTH End",
                showarrow=False,
                font=dict(size=8, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                row=1, col=1,
            )

    # RTH open (current day 6:30am)
    if not current_pre.empty:
        rth_open_time = current_pre["datetime"].min()
        fig.add_vline(
            x=rth_open_time,
            line=dict(color="#94a3b8", width=1.5, dash="dashdot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=rth_open_time, y=price_max + price_pad * 0.9,
            text="RTH Open",
            showarrow=False,
            font=dict(size=9, color="#64748b"),
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1,
        )

    # ── Episode annotations ─────────────────────────────────────────────────
    prev_d_state = {}

    for _, ep in day_episodes.iterrows():
        outcome = ep["terminal_outcome"]
        state_type = ep["state_type"]
        direction = ep["direction"] if pd.notna(ep["direction"]) else None
        d_state = ep["discovery_state"]
        mid_time = ep["start_time"] + (ep["end_time"] - ep["start_time"]) / 2

        if state_type == "balance":
            fig.add_annotation(
                x=mid_time, y=price_max + price_pad * 0.3,
                text="B",
                showarrow=False,
                font=dict(size=9, color=OUTCOME_TEXT["B"]),
                opacity=0.6,
                row=1, col=1,
            )
            continue

        # ── Discovery episode annotations ───────────────────────────────────
        fc = int(ep["failure_count_at_start"])
        ext_pts = ep["max_extension_points"]
        ext_atr = ep["max_extension_atr"] if pd.notna(ep["max_extension_atr"]) else 0
        dur_min = int(ep["duration_minutes"])

        # Reference boundary for markers: prior VAH for up, prior VAL for down
        if direction == "up":
            ann_y = price_max + price_pad * 0.7
            marker_y = prior_vah if pd.notna(prior_vah) else ibh
        else:
            ann_y = price_min - price_pad * 0.7
            marker_y = prior_val if pd.notna(prior_val) else ibl

        outcome_label = outcome
        if outcome == "A" and ep.get("acceptance_achieved"):
            acc_min = int(ep["time_in_acceptance_min"]) if pd.notna(ep["time_in_acceptance_min"]) else 0
            outcome_label = f"A ({acc_min}m)"

        text_lines = [
            f"<b>{d_state}</b>  →  <b>{outcome_label}</b>",
            f"{ext_pts:.0f}pt / {ext_atr:.1f}σ  ·  {dur_min}m",
        ]
        if fc > 0:
            text_lines[0] = f"<b>{d_state}</b> (×{fc} fail)  →  <b>{outcome_label}</b>"

        text_color = OUTCOME_TEXT.get(outcome, "#374151")

        fig.add_annotation(
            x=mid_time, y=ann_y,
            text="<br>".join(text_lines),
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor=text_color,
            ax=0, ay=30 if direction == "down" else -30,
            font=dict(size=10, color=text_color),
            align="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=text_color,
            borderwidth=1,
            borderpad=3,
            row=1, col=1,
        )

        # ── Transition arrow ────────────────────────────────────────────────
        if direction and direction in prev_d_state:
            prev = prev_d_state[direction]
            if d_state != prev and d_state != "D0":
                fig.add_annotation(
                    x=ep["start_time"],
                    y=ann_y + (price_pad * 0.2 if direction == "up" else -price_pad * 0.2),
                    ax=-40, ay=0,
                    text=f"{prev}→{d_state}",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.2,
                    arrowwidth=1.5,
                    arrowcolor="#7c3aed",
                    font=dict(size=9, color="#7c3aed", family="monospace"),
                    bgcolor="rgba(237,233,254,0.9)",
                    bordercolor="#7c3aed",
                    borderwidth=1,
                    borderpad=2,
                    row=1, col=1,
                )

        if direction:
            prev_d_state[direction] = d_state

        # ── Start/end markers ───────────────────────────────────────────────
        marker_symbol = DIRECTION_MARKER.get(direction, "circle")
        fig.add_trace(
            go.Scatter(
                x=[ep["start_time"]], y=[marker_y],
                mode="markers",
                marker=dict(symbol=marker_symbol, size=10, color=text_color,
                            line=dict(width=1, color="white")),
                showlegend=False,
                hovertext=f"Episode start: {d_state} {direction}",
                hoverinfo="text",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[ep["end_time"]], y=[marker_y],
                mode="markers+text",
                marker=dict(
                    symbol="x" if outcome == "R" else "star" if outcome in ("A", "A*") else "square",
                    size=10 if outcome == "R" else 13,
                    color=text_color,
                    line=dict(width=1, color="white"),
                ),
                text=[outcome],
                textposition="top center" if direction == "up" else "bottom center",
                textfont=dict(size=9, color=text_color),
                showlegend=False,
                hovertext=f"Outcome: {outcome} ({d_state} {direction})",
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # ── Summary panel ───────────────────────────────────────────────────────
    disc_eps = day_episodes[day_episodes["state_type"] == "discovery"]
    total_disc = len(disc_eps)
    total_rej = (disc_eps["terminal_outcome"] == "R").sum()
    total_acc = disc_eps["terminal_outcome"].isin(["A", "A*"]).sum()
    max_d = disc_eps["discovery_state"].max() if not disc_eps.empty else "D0"

    summary_lines = [
        f"<b>Pre-IB: {session_date}</b>",
    ]
    if pd.notna(prior_vah) and pd.notna(prior_val):
        summary_lines.append(f"Ref: Prior VAH {prior_vah:.0f} / VAL {prior_val:.0f}")
    if pd.notna(ibh) and pd.notna(ibl):
        summary_lines.append(f"IB formed: {ibh:.0f} / {ibl:.0f} ({ibh - ibl:.0f}pt)")
    summary_lines += [
        f"Episodes: {total_disc} discovery / {len(day_episodes)} total",
        f"Rejections: {total_rej}  ·  Acceptances: {total_acc}",
        f"Max D-state: {max_d}",
    ]
    for d in ["up", "down"]:
        d_eps = disc_eps[disc_eps["direction"] == d]
        if not d_eps.empty:
            d_rej = (d_eps["terminal_outcome"] == "R").sum()
            d_states = d_eps["discovery_state"].tolist()
            summary_lines.append(f"  {d}: {len(d_eps)} probes, {d_rej}R, states={d_states}")

    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.0,
        xanchor="right", yanchor="top",
        text="<br>".join(summary_lines),
        showarrow=False,
        font=dict(size=10, family="monospace"),
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#94a3b8",
        borderwidth=1,
        borderpad=6,
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    for outcome, color in OUTCOME_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(color=OUTCOME_BORDER[outcome], width=2)),
                name=f"{outcome} episode",
                showlegend=True,
            )
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"MNQ Pre-IB Discovery — {session_date} (prior day + overnight + pre-IB)",
            font=dict(size=16),
        ),
        xaxis_rangeslider_visible=False,
        xaxis2_title="Time (PT)",
        yaxis_title="Price",
        yaxis2_title="Volume",
        template="plotly_white",
        height=750,
        margin=dict(l=60, r=200, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top", y=0.65,
            xanchor="left", x=1.01,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(range=[price_min - price_pad, price_max + price_pad], row=1, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plotly candle + pre-IB episode overlay (two-day view).")
    parser.add_argument("date", nargs="?", default=None, help="Session date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save to HTML instead of opening browser")
    parser.add_argument("--list", action="store_true", help="List available session dates with D1+ activity")
    args = parser.parse_args()

    candles, episodes, profiles = load_data()
    dates = available_dates(episodes)

    if args.list:
        print(f"Available dates: {len(dates)} sessions with pre-IB episodes")
        print("\nDates with D1+ failure escalation:")
        for d in dates:
            day_eps = episodes[episodes["session_date"] == d]
            disc = day_eps[day_eps["state_type"] == "discovery"]
            states = disc["discovery_state"].tolist()
            if any(s in ["D1", "D2", "D3", "D4+"] for s in states):
                outcomes = disc["terminal_outcome"].tolist()
                directions = disc["direction"].tolist()
                print(f"  {d}  states={states}  outcomes={outcomes}  dirs={directions}")
        return

    if args.date is None:
        # Auto-select: highest D-state, then most episodes
        best_date = None
        best_score = -1
        d_order = {"D0": 0, "D1": 1, "D2": 2, "D3": 3, "D4+": 4}
        for d in dates:
            day_eps = episodes[episodes["session_date"] == d]
            disc = day_eps[day_eps["state_type"] == "discovery"]
            if disc.empty:
                continue
            max_d = max(d_order.get(s, 0) for s in disc["discovery_state"])
            score = max_d * 100 + len(disc)
            if score > best_score:
                best_score = score
                best_date = d
        session_date = best_date or dates[0]
        print(f"Auto-selected: {session_date}")
    else:
        session_date = args.date
        if session_date not in dates:
            print(f"Date {session_date} not found. Use --list to see available dates.")
            sys.exit(1)

    fig = build_figure(session_date, candles, episodes, profiles)

    if args.save:
        out_path = os.path.join(OUTPUT_DIR, f"pre_ib_episodes_{session_date}.html")
        fig.write_html(out_path)
        print(f"Saved: {out_path}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
