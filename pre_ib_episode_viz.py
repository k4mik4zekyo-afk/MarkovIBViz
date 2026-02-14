"""
pre_ib_episode_viz.py — Plotly candlestick visualization with pre-IB Markov episode overlays.

Displays prior day RTH + overnight + current day RTH with pre-IB episode
overlays on the current day's pre-IB window. Includes a range slider to
zoom into specific regions. Features:
- Prior RTH VAH / VAL reference boundaries (pre-IB discovery boundaries)
- Prior RTH POC
- Toggle button to switch between prior day IBH/IBL vs current day IBH/IBL
- Phase separators (Prior RTH End, RTH Open, IB Formed)
- Episode shading with color-coded backgrounds
- D-state + failure count annotations (below volume to avoid overlap)
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

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Colour palette ──────────────────────────────────────────────────────────
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


def load_data():
    """Load cleaned candle data, pre-IB episode log, and session profiles."""
    candles = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), parse_dates=["datetime"])
    episodes = pd.read_csv(os.path.join(OUTPUT_DIR, "episode_log_pre_ib.csv"), parse_dates=["start_time", "end_time"])
    profiles = pd.read_csv(os.path.join(OUTPUT_DIR, "session_profiles.csv"))
    return candles, episodes, profiles


def available_dates(episodes):
    """Return sorted list of dates that have pre-IB episodes."""
    return sorted(episodes["session_date"].unique())


def _get_prior_session_date(session_date, profiles):
    """Get the session date immediately before the given date."""
    all_dates = sorted(profiles["session_date"].unique())
    try:
        idx = all_dates.index(session_date)
    except ValueError:
        return None
    if idx == 0:
        return None
    return all_dates[idx - 1]


def build_figure(session_date, candles, episodes, profiles):
    """Build the Plotly figure showing two full RTH sessions with pre-IB episode overlays."""

    prior_date = _get_prior_session_date(session_date, profiles)
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

    # Prior day IBH/IBL for toggle
    prior_ibh = np.nan
    prior_ibl = np.nan
    if prior_date:
        prior_prof = profiles[profiles["session_date"] == prior_date]
        if not prior_prof.empty:
            pp = prior_prof.iloc[0]
            prior_ibh = pp["ibh"] if pd.notna(pp.get("ibh")) else np.nan
            prior_ibl = pp["ibl"] if pd.notna(pp.get("ibl")) else np.nan

    # ── Gather candles: prior RTH + overnight + current RTH ───────────────
    frames = []

    # Prior day full RTH (pre_ib + post_ib)
    if prior_date:
        prior_rth = candles[
            (candles["session_date"] == prior_date) &
            (candles["phase"].isin(["pre_ib", "post_ib"]))
        ].copy()
        prior_rth["section"] = "prior_rth"
        frames.append(prior_rth)

    # Overnight bars (live under the prior session_date)
    if prior_date:
        overnight = candles[
            (candles["session_date"] == prior_date) &
            (candles["phase"] == "overnight")
        ].copy()
        overnight["section"] = "overnight"
        frames.append(overnight)

    # Current day full RTH (pre_ib + post_ib)
    current_rth = candles[
        (candles["session_date"] == session_date) &
        (candles["phase"].isin(["pre_ib", "post_ib"]))
    ].copy()
    current_rth["section"] = "current_rth"
    frames.append(current_rth)

    if not frames or all(f.empty for f in frames):
        print(f"No candle data for {session_date}.")
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
    hover_texts = []
    for _, r in all_candles.iterrows():
        hover_texts.append(
            f"O: {r['open']:.2f}  H: {r['high']:.2f}<br>"
            f"L: {r['low']:.2f}  C: {r['close']:.2f}<br>"
            f"Vol: {int(r['volume']):,}"
        )
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
            text=hover_texts,
            hoverinfo="text",
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
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── Reference lines (hlines for VAH/VAL/POC — always visible) ──────────
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

    # ── IBH/IBL as Scatter traces (toggleable) ─────────────────────────────
    # We need trace indices for the toggle buttons.
    # Count existing traces: candlestick (1) + volume (1) = 2 traces so far.
    x_range = [all_candles["datetime"].min(), all_candles["datetime"].max()]

    # Current day IBH/IBL traces (visible by default)
    curr_ibh_trace = None
    curr_ibl_trace = None
    if pd.notna(ibh):
        curr_ibh_trace = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[ibh, ibh],
                mode="lines+text",
                line=dict(color="#3b82f6", width=1.5, dash="dash"),
                text=[f"Current IBH {ibh:.2f}", ""],
                textposition="top left",
                textfont=dict(color="#3b82f6", size=10),
                name=f"Current IBH {ibh:.0f}",
                showlegend=False,
                visible=True,
            ),
            row=1, col=1,
        )
    if pd.notna(ibl):
        curr_ibl_trace = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[ibl, ibl],
                mode="lines+text",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                text=[f"Current IBL {ibl:.2f}", ""],
                textposition="bottom left",
                textfont=dict(color="#f59e0b", size=10),
                name=f"Current IBL {ibl:.0f}",
                showlegend=False,
                visible=True,
            ),
            row=1, col=1,
        )

    # Prior day IBH/IBL traces (hidden by default)
    prior_ibh_trace = None
    prior_ibl_trace = None
    if pd.notna(prior_ibh):
        prior_ibh_trace = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[prior_ibh, prior_ibh],
                mode="lines+text",
                line=dict(color="#3b82f6", width=1.5, dash="longdashdot"),
                text=[f"Prior IBH {prior_ibh:.2f}", ""],
                textposition="top left",
                textfont=dict(color="#3b82f6", size=10),
                name=f"Prior IBH {prior_ibh:.0f}",
                showlegend=False,
                visible=False,
            ),
            row=1, col=1,
        )
    if pd.notna(prior_ibl):
        prior_ibl_trace = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[prior_ibl, prior_ibl],
                mode="lines+text",
                line=dict(color="#f59e0b", width=1.5, dash="longdashdot"),
                text=[f"Prior IBL {prior_ibl:.2f}", ""],
                textposition="bottom left",
                textfont=dict(color="#f59e0b", size=10),
                name=f"Prior IBL {prior_ibl:.0f}",
                showlegend=False,
                visible=False,
            ),
            row=1, col=1,
        )

    # Track index where toggle traces end (before episode markers start)
    n_traces_before_episodes = len(fig.data)

    # ── Phase separators ────────────────────────────────────────────────────
    # Prior day RTH end
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
                text=f"Prior RTH End ({prior_date})",
                showarrow=False,
                font=dict(size=8, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                row=1, col=1,
            )

    # Overnight start
    if prior_date:
        overnight_bars = candles[
            (candles["session_date"] == prior_date) &
            (candles["phase"] == "overnight")
        ]
        if not overnight_bars.empty:
            on_start_time = overnight_bars["datetime"].min()
            fig.add_vline(
                x=on_start_time,
                line=dict(color="#6b7280", width=1, dash="dot"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=on_start_time, y=price_max + price_pad * 0.6,
                text="Overnight Start",
                showarrow=False,
                font=dict(size=8, color="#6b7280"),
                bgcolor="rgba(255,255,255,0.8)",
                row=1, col=1,
            )

    # Current day RTH open (6:30am)
    current_pre = current_rth[current_rth["phase"] == "pre_ib"]
    if not current_pre.empty:
        rth_open_time = current_pre["datetime"].min()
        fig.add_vline(
            x=rth_open_time,
            line=dict(color="#94a3b8", width=1.5, dash="dashdot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=rth_open_time, y=price_max + price_pad * 0.9,
            text=f"RTH Open ({session_date})",
            showarrow=False,
            font=dict(size=9, color="#64748b"),
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1,
        )

    # IB formed (7:30am current day)
    if not current_pre.empty:
        ib_boundary_time = current_pre["datetime"].max() + pd.Timedelta(minutes=5)
        fig.add_vline(
            x=ib_boundary_time,
            line=dict(color="#94a3b8", width=1.5, dash="dashdot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=ib_boundary_time, y=price_max + price_pad * 0.6,
            text="IB Formed",
            showarrow=False,
            font=dict(size=9, color="#64748b"),
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1,
        )

    # ── Episode annotations (below volume panel) ───────────────────────────
    prev_d_state = {}
    disc_counter = 0  # for alternating y-levels

    for _, ep in day_episodes.iterrows():
        outcome = ep["terminal_outcome"]
        state_type = ep["state_type"]
        direction = ep["direction"] if pd.notna(ep["direction"]) else None
        d_state = ep["discovery_state"]
        mid_time = ep["start_time"] + (ep["end_time"] - ep["start_time"]) / 2

        if state_type == "balance":
            continue

        # ── Discovery episode annotations ───────────────────────────────────
        fc = int(ep["failure_count_at_start"])
        ext_pts = ep["max_extension_points"]
        ext_atr = ep["max_extension_atr"] if pd.notna(ep["max_extension_atr"]) else 0
        dur_min = int(ep["duration_minutes"])

        # Reference boundary for markers
        if direction == "up":
            marker_y = prior_vah if pd.notna(prior_vah) else ibh
        else:
            marker_y = prior_val if pd.notna(prior_val) else ibl

        outcome_label = outcome
        if outcome == "A" and ep.get("acceptance_achieved"):
            acc_min = int(ep["time_in_acceptance_min"]) if pd.notna(ep["time_in_acceptance_min"]) else 0
            outcome_label = f"A ({acc_min}m)"

        # Compact single-line format
        fail_tag = f" (x{fc} fail)" if fc > 0 else ""
        ann_text = f"<b>{d_state}</b>{fail_tag} → <b>{outcome_label}</b>  {ext_pts:.0f}pt/{ext_atr:.1f}σ  {dur_min}m"

        dir_arrow = "\u25B2" if direction == "up" else "\u25BC"
        ann_text = f"{dir_arrow} {ann_text}"

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

        # ── Transition arrow (in candle area) ───────────────────────────────
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

        if direction:
            prev_d_state[direction] = d_state

        # ── Start/end markers on boundary line ──────────────────────────────
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

    # ── IBH/IBL toggle button ───────────────────────────────────────────────
    has_curr = curr_ibh_trace is not None or curr_ibl_trace is not None
    has_prior = prior_ibh_trace is not None or prior_ibl_trace is not None

    if has_curr and has_prior:
        n_total = len(fig.data)
        # Build visibility arrays: True for all non-IB traces, then set IB pairs
        base_vis = [True] * n_total

        def _make_vis(show_curr, show_prior):
            vis = list(base_vis)
            if curr_ibh_trace is not None:
                vis[curr_ibh_trace] = show_curr
            if curr_ibl_trace is not None:
                vis[curr_ibl_trace] = show_curr
            if prior_ibh_trace is not None:
                vis[prior_ibh_trace] = show_prior
            if prior_ibl_trace is not None:
                vis[prior_ibl_trace] = show_prior
            return vis

        toggle_buttons = [
            dict(
                label=f"Current IBH/IBL ({session_date})",
                method="update",
                args=[{"visible": _make_vis(True, False)}],
            ),
            dict(
                label=f"Prior IBH/IBL ({prior_date})",
                method="update",
                args=[{"visible": _make_vis(False, True)}],
            ),
            dict(
                label="Both IBH/IBL",
                method="update",
                args=[{"visible": _make_vis(True, True)}],
            ),
        ]

        updatemenus = [dict(
            type="buttons",
            direction="right",
            x=0.0,
            xanchor="left",
            y=1.12,
            yanchor="top",
            buttons=toggle_buttons,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        )]
    else:
        updatemenus = []

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
        summary_lines.append(f"Current IB: {ibh:.0f} / {ibl:.0f} ({ibh - ibl:.0f}pt)")
    if pd.notna(prior_ibh) and pd.notna(prior_ibl):
        summary_lines.append(f"Prior IB: {prior_ibh:.0f} / {prior_ibl:.0f} ({prior_ibh - prior_ibl:.0f}pt)")
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
            text=f"MNQ Pre-IB Discovery — {session_date} (prior RTH + overnight + current RTH)",
            font=dict(size=16),
        ),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider=dict(visible=True, thickness=0.04),
        xaxis2_title="Time (PT)",
        yaxis_title="Price",
        yaxis2_title="Volume",
        template="plotly_white",
        width=1800,
        height=1600,
        margin=dict(l=60, r=220, t=80, b=160),
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
        updatemenus=updatemenus,
    )

    fig.update_yaxes(range=[price_min - price_pad, price_max + price_pad], row=1, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plotly candle + pre-IB episode overlay (two-day RTH view).")
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
