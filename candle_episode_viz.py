"""
candle_episode_viz.py — Plotly candlestick visualization with Markov episode overlays.

Displays a single trading day's full RTH session (pre-IB + post-IB) 5-minute candles
with post-IB episode overlays including:
- IBH / IBL reference boundaries
- Prior RTH VAH / VAL reference lines (pre-IB discovery boundaries)
- Phase separator at IB formation (7:30am)
- Episode shading (discovery vs balance) with color-coded backgrounds
- Failure count (D-state) annotations on each discovery episode
- Terminal outcome labels (A, A*, R, B) at episode end
- Transition arrows when D-state escalates (D0 → D1 → D2 → ...)

Usage:
    python candle_episode_viz.py                        # interactive date picker
    python candle_episode_viz.py 2025-05-02             # specific date
    python candle_episode_viz.py 2025-05-02 --save      # save to HTML file
    python candle_episode_viz.py --list                  # list available dates
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Colour palette ──────────────────────────────────────────────────────────
OUTCOME_COLORS = {
    "A":  "rgba(16, 185, 129, 0.18)",   # green  – acceptance
    "A*": "rgba(56, 189, 248, 0.18)",   # cyan   – acceptance incomplete
    "R":  "rgba(239, 68, 68, 0.12)",    # red    – rejection
    "B":  "rgba(148, 163, 184, 0.10)",  # slate  – balance
    "C":  "rgba(251, 191, 36, 0.18)",   # amber  – continuation
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


def load_data():
    """Load cleaned candle data, episode log, and session profiles."""
    candles = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), parse_dates=["datetime"])
    episodes = pd.read_csv(os.path.join(OUTPUT_DIR, "episode_log_post_ib.csv"), parse_dates=["start_time", "end_time"])
    profiles = pd.read_csv(os.path.join(OUTPUT_DIR, "session_profiles.csv"))
    return candles, episodes, profiles


def available_dates(episodes):
    """Return sorted list of dates that have post-IB episodes."""
    return sorted(episodes["session_date"].unique())


def build_figure(session_date, candles, episodes, profiles):
    """Build the Plotly figure for a single trading day (full RTH: pre-IB + post-IB)."""

    # ── Filter data ─────────────────────────────────────────────────────────
    day_candles = candles[
        (candles["session_date"] == session_date) &
        (candles["phase"].isin(["pre_ib", "post_ib"]))
    ].copy().sort_values("datetime").reset_index(drop=True)

    day_episodes = episodes[episodes["session_date"] == session_date].copy().reset_index(drop=True)

    day_profile = profiles[profiles["session_date"] == session_date]

    if day_candles.empty:
        print(f"No candle data for {session_date}.")
        sys.exit(1)
    if day_episodes.empty:
        print(f"No episodes for {session_date}.")
        sys.exit(1)

    ibh = day_profile["ibh"].iloc[0]
    ibl = day_profile["ibl"].iloc[0]
    prior_vah = day_profile["prior_rth_vah"].iloc[0] if "prior_rth_vah" in day_profile.columns else np.nan
    prior_val = day_profile["prior_rth_val"].iloc[0] if "prior_rth_val" in day_profile.columns else np.nan

    price_min = day_candles["low"].min()
    price_max = day_candles["high"].max()
    price_pad = (price_max - price_min) * 0.06

    # ── Create figure with volume subplot ───────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(None, None),
    )

    # ── Episode background shading (vrects) ─────────────────────────────────
    for idx, ep in day_episodes.iterrows():
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
            x=day_candles["datetime"],
            open=day_candles["open"],
            high=day_candles["high"],
            low=day_candles["low"],
            close=day_candles["close"],
            name="MNQ 5m",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#ef4444",
            whiskerwidth=0.4,
        ),
        row=1, col=1,
    )

    # ── Volume bars ─────────────────────────────────────────────────────────
    colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(day_candles["close"], day_candles["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=day_candles["datetime"],
            y=day_candles["volume"],
            marker_color=colors,
            opacity=0.5,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── IBH / IBL reference lines ───────────────────────────────────────────
    fig.add_hline(
        y=ibh, line=dict(color="#3b82f6", width=1.5, dash="dash"),
        annotation_text=f"IBH {ibh:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#3b82f6", size=10),
        row=1, col=1,
    )
    fig.add_hline(
        y=ibl, line=dict(color="#f59e0b", width=1.5, dash="dash"),
        annotation_text=f"IBL {ibl:.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color="#f59e0b", size=10),
        row=1, col=1,
    )

    # ── Prior RTH VAH / VAL reference lines ─────────────────────────────────
    if pd.notna(prior_vah):
        fig.add_hline(
            y=prior_vah, line=dict(color="#8b5cf6", width=1, dash="dot"),
            annotation_text=f"Prior VAH {prior_vah:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#8b5cf6", size=9),
            row=1, col=1,
        )
    if pd.notna(prior_val):
        fig.add_hline(
            y=prior_val, line=dict(color="#8b5cf6", width=1, dash="dot"),
            annotation_text=f"Prior VAL {prior_val:.2f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#8b5cf6", size=9),
            row=1, col=1,
        )

    # ── IB formation separator (vertical line at 7:30am) ────────────────────
    pre_ib_candles = day_candles[day_candles["phase"] == "pre_ib"]
    if not pre_ib_candles.empty:
        ib_boundary_time = pre_ib_candles["datetime"].max() + pd.Timedelta(minutes=5)
        fig.add_vline(
            x=ib_boundary_time,
            line=dict(color="#94a3b8", width=1.5, dash="dashdot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=ib_boundary_time, y=price_max + price_pad * 0.95,
            text="IB Formed",
            showarrow=False,
            font=dict(size=9, color="#64748b"),
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1,
        )

    # ── Episode annotations (below volume panel to avoid overlap) ───────────
    prev_d_state = {}  # direction -> previous D-state string
    disc_counter = 0   # for alternating y-levels

    for idx, ep in day_episodes.iterrows():
        outcome = ep["terminal_outcome"]
        state_type = ep["state_type"]
        direction = ep["direction"] if pd.notna(ep["direction"]) else None
        d_state = ep["discovery_state"]
        mid_time = ep["start_time"] + (ep["end_time"] - ep["start_time"]) / 2

        if state_type == "balance":
            continue

        # ── Discovery episode annotations ───────────────────────────────────
        fc = int(ep["failure_count_at_start"])
        d_label = d_state
        ext_pts = ep["max_extension_points"]
        ext_atr = ep["max_extension_atr"] if pd.notna(ep["max_extension_atr"]) else 0
        dur_min = int(ep["duration_minutes"])

        marker_y = ibh if direction == "up" else ibl

        # Build annotation text block
        outcome_label = outcome
        if outcome == "A" and ep.get("acceptance_achieved"):
            acc_min = int(ep["time_in_acceptance_min"]) if pd.notna(ep["time_in_acceptance_min"]) else 0
            outcome_label = f"A ({acc_min}m)"

        # Compact single-line format
        fail_tag = f" (x{fc} fail)" if fc > 0 else ""
        dir_arrow = "\u25B2" if direction == "up" else "\u25BC"
        ann_text = f"{dir_arrow} <b>{d_label}</b>{fail_tag} → <b>{outcome_label}</b>  {ext_pts:.0f}pt/{ext_atr:.1f}σ  {dur_min}m"

        text_color = OUTCOME_TEXT.get(outcome, "#374151")

        # Alternate between two y-levels below volume
        y_level = -0.07 if disc_counter % 2 == 0 else -0.16
        disc_counter += 1

        fig.add_annotation(
            x=mid_time, y=y_level,
            xref="x2", yref="paper",
            text=ann_text,
            showarrow=False,
            font=dict(size=9, color=text_color),
            align="center",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=text_color,
            borderwidth=1,
            borderpad=3,
        )

        # ── Transition arrow (D-state escalation — in candle area) ──────────
        if direction and direction in prev_d_state:
            prev = prev_d_state[direction]
            if d_state != prev and d_state != "D0":
                trans_y = price_max + price_pad * 0.3 if direction == "up" else price_min - price_pad * 0.3
                fig.add_annotation(
                    x=ep["start_time"], y=trans_y,
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

        # Track D-state for transition detection
        if direction:
            prev_d_state[direction] = d_state

        # ── Start/end markers on boundary line ──────────────────────────────
        marker_symbol = DIRECTION_MARKER.get(direction, "circle")
        # Start marker
        fig.add_trace(
            go.Scatter(
                x=[ep["start_time"]],
                y=[marker_y],
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=10,
                    color=text_color,
                    line=dict(width=1, color="white"),
                ),
                showlegend=False,
                hovertext=f"Episode start: {d_state} {direction}",
                hoverinfo="text",
            ),
            row=1, col=1,
        )
        # End marker (outcome)
        fig.add_trace(
            go.Scatter(
                x=[ep["end_time"]],
                y=[marker_y],
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

    # ── Failure count summary panel (top-right annotation) ──────────────────
    disc_eps = day_episodes[day_episodes["state_type"] == "discovery"]
    total_disc = len(disc_eps)
    total_rej = (disc_eps["terminal_outcome"] == "R").sum()
    total_acc = disc_eps["terminal_outcome"].isin(["A", "A*"]).sum()
    max_d = disc_eps["discovery_state"].max() if not disc_eps.empty else "D0"
    ib_range = ibh - ibl

    summary_lines = [
        f"<b>{session_date}</b>",
        f"IB Range: {ib_range:.1f}pt  (IBH {ibh:.0f} / IBL {ibl:.0f})",
    ]
    if pd.notna(prior_vah) and pd.notna(prior_val):
        summary_lines.append(f"Prior VAH {prior_vah:.0f} / VAL {prior_val:.0f}")
    summary_lines += [
        f"Post-IB: {total_disc} discovery / {len(day_episodes)} total",
        f"Rejections: {total_rej}  ·  Acceptances: {total_acc}",
        f"Max D-state: {max_d}",
    ]

    # Build per-direction breakdown
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

    # ── Legend shapes (custom) ──────────────────────────────────────────────
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
            text=f"MNQ Full RTH Session — {session_date}",
            font=dict(size=16),
        ),
        xaxis_rangeslider_visible=False,
        xaxis2_title="Time (PT)",
        yaxis_title="Price",
        yaxis2_title="Volume",
        template="plotly_white",
        height=800,
        margin=dict(l=60, r=200, t=60, b=140),
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
    parser = argparse.ArgumentParser(description="Plotly candle + episode overlay for a single trading day.")
    parser.add_argument("date", nargs="?", default=None, help="Session date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save to HTML instead of opening browser")
    parser.add_argument("--list", action="store_true", help="List available session dates with D1+ activity")
    args = parser.parse_args()

    candles, episodes, profiles = load_data()
    dates = available_dates(episodes)

    if args.list:
        print(f"Available dates: {len(dates)} sessions")
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
        # Default to the most interesting day (highest max D-state)
        best_date = None
        best_max = -1
        d_order = {"D0": 0, "D1": 1, "D2": 2, "D3": 3, "D4+": 4}
        for d in dates:
            day_eps = episodes[episodes["session_date"] == d]
            disc = day_eps[day_eps["state_type"] == "discovery"]
            if disc.empty:
                continue
            max_d = max(d_order.get(s, 0) for s in disc["discovery_state"])
            n_disc = len(disc)
            score = max_d * 100 + n_disc
            if score > best_max:
                best_max = score
                best_date = d
        session_date = best_date or dates[0]
        print(f"Auto-selected: {session_date} (highest D-state escalation)")
    else:
        session_date = args.date
        if session_date not in dates:
            print(f"Date {session_date} not found. Use --list to see available dates.")
            sys.exit(1)

    fig = build_figure(session_date, candles, episodes, profiles)

    if args.save:
        out_path = os.path.join(OUTPUT_DIR, f"candle_episodes_{session_date}.html")
        fig.write_html(out_path)
        print(f"Saved: {out_path}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
