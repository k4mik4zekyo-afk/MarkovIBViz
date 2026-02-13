"""
state_engine.py — Two-phase discovery state machine with full Agent.md spec.

Key features per Agent.md:
- 30-minute acceptance threshold (configurable)
- Two-sided rejection within 30 minutes
- VWAP reset of failure_count at episode start only (Option B)
- Terminal outcomes: A (Acceptance), A* (Incomplete), R (Rejection), B (Balance)
- failure_count capped at 4+
- Episode starts on bar CLOSE outside boundary with touch

Phase 1 (Pre-IB): 6:30am - 7:25am
  Reference boundaries: Prior session VAH / VAL

Transition at 7:30am (IB established):
  Re-evaluate against IBH/IBL

Phase 2 (Post-IB): 7:30am - 12:55pm
  Reference boundaries: IBH / IBL
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

from config import CONFIG, FAILURE_COUNT_CAP, get_acceptance_bars, get_two_sided_bars

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _make_episode(
    session_date, state_type, direction, failure_count_at_start,
    start_time, end_time, duration_bars, duration_minutes,
    max_extension_points, max_retracement_points,
    acceptance_achieved, time_in_acceptance_min,
    terminal_outcome, phase, avg_atr, ib_range, or_range, reference_boundary
):
    """Create an episode dict with all fields per Agent.md schema."""
    ext_atr = max_extension_points / avg_atr if avg_atr and avg_atr > 0 else np.nan
    ext_ib = max_extension_points / ib_range if ib_range and ib_range > 0 and phase == "post_ib" else np.nan
    ext_or = max_extension_points / or_range if or_range and or_range > 0 else np.nan
    
    return {
        "session_date": session_date,
        "state_type": state_type,
        "direction": direction,
        "failure_count_at_start": min(failure_count_at_start, FAILURE_COUNT_CAP) if failure_count_at_start else 0,
        "start_time": start_time,
        "end_time": end_time,
        "duration_bars": duration_bars,
        "duration_minutes": duration_minutes,
        "max_extension_points": max_extension_points,
        "max_extension_atr": ext_atr,
        "max_extension_ib": ext_ib,
        "max_extension_or": ext_or,
        "max_retracement_points": max_retracement_points,
        "acceptance_achieved": acceptance_achieved,
        "time_in_acceptance_min": time_in_acceptance_min,
        "terminal_outcome": terminal_outcome,
        "phase": phase,
        "avg_atr_during_episode": avg_atr,
        "ib_range": ib_range,
        "or_range": or_range,
        "reference_boundary": reference_boundary,
        # Legacy compatibility
        "max_extension": max_extension_points,
        "max_retracement": max_retracement_points,
        "atr_at_start": avg_atr,
        "extension_atr": ext_atr,
        "extension_ib": ext_ib,
        "failure_count": min(failure_count_at_start, FAILURE_COUNT_CAP) if failure_count_at_start else 0,
    }


class DiscoveryStateMachine:
    """State machine for tracking discovery episodes with full Agent.md logic."""
    
    def __init__(self, session_date, ib_range, or_range, timeframe='5m'):
        self.session_date = session_date
        self.ib_range = ib_range
        self.or_range = or_range
        self.timeframe = timeframe
        
        # State
        self.state = "BALANCE"
        self.direction = None
        
        # Episode tracking
        self.episode_start_time = None
        self.episode_start_idx = 0
        self.extension_peak = 0.0
        self.max_extension = 0.0
        self.max_retracement = 0.0
        self.atrs_in_episode = []
        
        # Acceptance tracking (30-min threshold)
        self.bars_outside = 0
        self.acceptance_threshold_bars = get_acceptance_bars(timeframe)
        self.acceptance_achieved = False
        self.bars_in_acceptance = 0
        
        # Two-sided rejection tracking
        self.two_sided_threshold_bars = get_two_sided_bars(timeframe)
        self.bars_since_episode_start = 0
        
        # Failure counts (separate for up/down)
        self.failure_counts = {"up": 0, "down": 0}
        
        # VWAP reset flags (Option B - at episode start only)
        self.vwap_crossed_since_last_up = False
        self.vwap_crossed_since_last_down = False
        self.last_vwap = None
        
        # Balance tracking
        self.balance_start_time = None
        self.balance_start_idx = 0
        self.balance_atrs = []
        
        # Reference boundaries (set per phase)
        self.ref_up = None
        self.ref_down = None
        self.phase = None
        
    def set_references(self, ref_up, ref_down, phase):
        """Set reference boundaries for current phase."""
        self.ref_up = ref_up
        self.ref_down = ref_down
        self.phase = phase
    
    def check_vwap_cross(self, close, vwap):
        """Check if VWAP was crossed (for failure_count reset tracking)."""
        if self.last_vwap is not None and vwap is not None:
            # Cross from below to above
            if self.last_vwap < vwap and close > vwap:
                self.vwap_crossed_since_last_up = True
                self.vwap_crossed_since_last_down = True
            # Cross from above to below
            elif self.last_vwap > vwap and close < vwap:
                self.vwap_crossed_since_last_up = True
                self.vwap_crossed_since_last_down = True
        self.last_vwap = vwap
    
    def _reset_failure_count_if_vwap_crossed(self, direction):
        """Reset failure count if VWAP was crossed since last episode (Option B)."""
        if direction == "up" and self.vwap_crossed_since_last_up:
            self.failure_counts["up"] = 0
            self.vwap_crossed_since_last_up = False
        elif direction == "down" and self.vwap_crossed_since_last_down:
            self.failure_counts["down"] = 0
            self.vwap_crossed_since_last_down = False
    
    def _start_discovery(self, direction, dt, idx, high, low, atr):
        """Start a new discovery episode."""
        # Check VWAP reset (Option B - at episode start only)
        self._reset_failure_count_if_vwap_crossed(direction)
        
        self.state = f"DISCOVERY_{direction.upper()}"
        self.direction = direction
        self.episode_start_time = dt
        self.episode_start_idx = idx
        
        if direction == "up":
            self.extension_peak = high
            self.max_extension = high - self.ref_up
        else:
            self.extension_peak = low
            self.max_extension = self.ref_down - low
        
        self.max_retracement = 0.0
        self.atrs_in_episode = [atr] if not np.isnan(atr) else []
        
        # Reset acceptance tracking
        self.bars_outside = 1
        self.acceptance_achieved = False
        self.bars_in_acceptance = 0
        self.bars_since_episode_start = 1
    
    def _start_balance(self, dt, idx, atr):
        """Start a new balance episode."""
        self.state = "BALANCE"
        self.direction = None
        self.balance_start_time = dt
        self.balance_start_idx = idx
        self.balance_atrs = [atr] if not np.isnan(atr) else []
    
    def _close_episode(self, outcome, dt, idx, ref_boundary):
        """Close current discovery episode and return episode dict."""
        duration_bars = idx - self.episode_start_idx + 1
        
        # Calculate duration in minutes based on timeframe
        if self.timeframe == '5m':
            duration_minutes = duration_bars * 5
        elif self.timeframe == '15m':
            duration_minutes = duration_bars * 15
        elif self.timeframe == '30m':
            duration_minutes = duration_bars * 30
        else:
            duration_minutes = duration_bars * 5
        
        avg_atr = np.mean(self.atrs_in_episode) if self.atrs_in_episode else np.nan
        
        # Calculate time in acceptance
        if self.timeframe == '5m':
            time_in_acceptance = self.bars_in_acceptance * 5
        elif self.timeframe == '15m':
            time_in_acceptance = self.bars_in_acceptance * 15
        elif self.timeframe == '30m':
            time_in_acceptance = self.bars_in_acceptance * 30
        else:
            time_in_acceptance = self.bars_in_acceptance * 5
        
        episode = _make_episode(
            session_date=self.session_date,
            state_type="discovery",
            direction=self.direction,
            failure_count_at_start=self.failure_counts.get(self.direction, 0),
            start_time=self.episode_start_time,
            end_time=dt,
            duration_bars=duration_bars,
            duration_minutes=duration_minutes,
            max_extension_points=self.max_extension,
            max_retracement_points=self.max_retracement,
            acceptance_achieved=self.acceptance_achieved,
            time_in_acceptance_min=time_in_acceptance,
            terminal_outcome=outcome,
            phase=self.phase,
            avg_atr=avg_atr,
            ib_range=self.ib_range,
            or_range=self.or_range,
            reference_boundary=ref_boundary,
        )
        
        # Increment failure count only on Rejection
        if outcome == "R":
            self.failure_counts[self.direction] = min(
                self.failure_counts[self.direction] + 1, 
                FAILURE_COUNT_CAP
            )
        
        return episode
    
    def _close_balance(self, dt, idx):
        """Close current balance episode and return episode dict."""
        if self.balance_start_time is None:
            return None
        
        duration_bars = idx - self.balance_start_idx + 1
        
        if self.timeframe == '5m':
            duration_minutes = duration_bars * 5
        elif self.timeframe == '15m':
            duration_minutes = duration_bars * 15
        else:
            duration_minutes = duration_bars * 5
        
        avg_atr = np.mean(self.balance_atrs) if self.balance_atrs else np.nan
        
        episode = _make_episode(
            session_date=self.session_date,
            state_type="balance",
            direction=None,
            failure_count_at_start=0,
            start_time=self.balance_start_time,
            end_time=dt,
            duration_bars=duration_bars,
            duration_minutes=duration_minutes,
            max_extension_points=0.0,
            max_retracement_points=0.0,
            acceptance_achieved=False,
            time_in_acceptance_min=0,
            terminal_outcome="B",
            phase=self.phase,
            avg_atr=avg_atr,
            ib_range=self.ib_range,
            or_range=self.or_range,
            reference_boundary=None,
        )
        
        self.balance_start_time = None
        return episode
    
    def process_bar(self, idx, high, low, close, dt, atr, vwap, is_last_bar=False):
        """Process a single bar and return any completed episodes."""
        episodes = []
        
        # Track VWAP crosses while in Balance
        if self.state == "BALANCE":
            self.check_vwap_cross(close, vwap)
        
        # --- BALANCE STATE ---
        if self.state == "BALANCE":
            if self.balance_start_time is None:
                self._start_balance(dt, idx, atr)
            else:
                if not np.isnan(atr):
                    self.balance_atrs.append(atr)
            
            # Check for breakout (close outside + touch)
            # Discovery Up: high touched ref_up AND close > ref_up
            if high >= self.ref_up and close > self.ref_up:
                # Close balance episode
                bal_ep = self._close_balance(dt, idx)
                if bal_ep:
                    episodes.append(bal_ep)
                
                # Start Discovery Up
                self._start_discovery("up", dt, idx, high, low, atr)
            
            # Discovery Down: low touched ref_down AND close < ref_down
            elif low <= self.ref_down and close < self.ref_down:
                # Close balance episode
                bal_ep = self._close_balance(dt, idx)
                if bal_ep:
                    episodes.append(bal_ep)
                
                # Start Discovery Down
                self._start_discovery("down", dt, idx, high, low, atr)
            
            # End of phase/session - close balance
            elif is_last_bar and self.balance_start_time is not None:
                bal_ep = self._close_balance(dt, idx)
                if bal_ep:
                    episodes.append(bal_ep)
            
            return episodes
        
        # --- DISCOVERY UP STATE ---
        if self.state == "DISCOVERY_UP":
            if not np.isnan(atr):
                self.atrs_in_episode.append(atr)
            
            self.bars_since_episode_start += 1
            
            # Update extension
            if high > self.extension_peak:
                self.extension_peak = high
                self.max_extension = self.extension_peak - self.ref_up
            
            # Update retracement
            retracement = self.extension_peak - low
            if retracement > self.max_retracement:
                self.max_retracement = retracement
            
            # Check if still outside
            if close > self.ref_up:
                self.bars_outside += 1
                
                # Check acceptance threshold (30 minutes)
                if self.bars_outside >= self.acceptance_threshold_bars:
                    if not self.acceptance_achieved:
                        self.acceptance_achieved = True
                    self.bars_in_acceptance += 1
            
            # Two-sided rejection check: opposite boundary touched within threshold
            if low <= self.ref_down and self.bars_since_episode_start <= self.two_sided_threshold_bars:
                # Two-sided rejection
                ep = self._close_episode("R", dt, idx, self.ref_up)
                episodes.append(ep)
                
                # Start balance, then check for opposite direction
                self._start_balance(dt, idx, atr)
                
                if close < self.ref_down:
                    bal_ep = self._close_balance(dt, idx)
                    if bal_ep:
                        episodes.append(bal_ep)
                    self._start_discovery("down", dt, idx, high, low, atr)
                
                return episodes
            
            # Rejection: close back inside
            if close <= self.ref_up:
                ep = self._close_episode("R", dt, idx, self.ref_up)
                episodes.append(ep)
                
                self._start_balance(dt, idx, atr)
                
                # Check immediate opposite breakout
                if low <= self.ref_down and close < self.ref_down:
                    bal_ep = self._close_balance(dt, idx)
                    if bal_ep:
                        episodes.append(bal_ep)
                    self._start_discovery("down", dt, idx, high, low, atr)
                
                return episodes
            
            # End of phase/session
            if is_last_bar:
                if self.acceptance_achieved:
                    outcome = "A"
                else:
                    outcome = "A*"  # Incomplete
                ep = self._close_episode(outcome, dt, idx, self.ref_up)
                episodes.append(ep)
            
            return episodes
        
        # --- DISCOVERY DOWN STATE ---
        if self.state == "DISCOVERY_DOWN":
            if not np.isnan(atr):
                self.atrs_in_episode.append(atr)
            
            self.bars_since_episode_start += 1
            
            # Update extension
            if low < self.extension_peak:
                self.extension_peak = low
                self.max_extension = self.ref_down - self.extension_peak
            
            # Update retracement
            retracement = high - self.extension_peak
            if retracement > self.max_retracement:
                self.max_retracement = retracement
            
            # Check if still outside
            if close < self.ref_down:
                self.bars_outside += 1
                
                if self.bars_outside >= self.acceptance_threshold_bars:
                    if not self.acceptance_achieved:
                        self.acceptance_achieved = True
                    self.bars_in_acceptance += 1
            
            # Two-sided rejection check
            if high >= self.ref_up and self.bars_since_episode_start <= self.two_sided_threshold_bars:
                ep = self._close_episode("R", dt, idx, self.ref_down)
                episodes.append(ep)
                
                self._start_balance(dt, idx, atr)
                
                if close > self.ref_up:
                    bal_ep = self._close_balance(dt, idx)
                    if bal_ep:
                        episodes.append(bal_ep)
                    self._start_discovery("up", dt, idx, high, low, atr)
                
                return episodes
            
            # Rejection: close back inside
            if close >= self.ref_down:
                ep = self._close_episode("R", dt, idx, self.ref_down)
                episodes.append(ep)
                
                self._start_balance(dt, idx, atr)
                
                if high >= self.ref_up and close > self.ref_up:
                    bal_ep = self._close_balance(dt, idx)
                    if bal_ep:
                        episodes.append(bal_ep)
                    self._start_discovery("up", dt, idx, high, low, atr)
                
                return episodes
            
            # End of phase/session
            if is_last_bar:
                if self.acceptance_achieved:
                    outcome = "A"
                else:
                    outcome = "A*"
                ep = self._close_episode(outcome, dt, idx, self.ref_down)
                episodes.append(ep)
            
            return episodes
        
        return episodes


def process_session(session_bars, prior_vah, prior_val, ibh, ibl, ib_range, or_range, session_date, timeframe='5m'):
    """Process a single session with two-phase discovery analysis."""
    pre_ib_bars = session_bars[session_bars["phase"] == "pre_ib"].copy()
    post_ib_bars = session_bars[session_bars["phase"] == "post_ib"].copy()
    
    pre_ib_episodes = []
    post_ib_episodes = []
    
    # Create state machine
    sm = DiscoveryStateMachine(session_date, ib_range, or_range, timeframe)
    
    # Get ATR column name
    atr_col = f"atr_{CONFIG['atr_window']}"
    if atr_col not in session_bars.columns:
        atr_col = "atr_14"  # Fallback
    
    # --- Phase 1: Pre-IB (6:30-7:25am) against prior VAH/VAL ---
    if not pre_ib_bars.empty and pd.notna(prior_vah) and pd.notna(prior_val):
        sm.set_references(prior_vah, prior_val, "pre_ib")
        
        for i, (idx, row) in enumerate(pre_ib_bars.iterrows()):
            is_last = (i == len(pre_ib_bars) - 1)
            atr = row.get(atr_col, np.nan)
            vwap = row.get("vwap", np.nan)
            
            eps = sm.process_bar(
                idx=i,
                high=row["high"],
                low=row["low"],
                close=row["close"],
                dt=row["datetime"],
                atr=atr,
                vwap=vwap,
                is_last_bar=is_last
            )
            pre_ib_episodes.extend(eps)
    
    # --- Transition at 7:30am ---
    if pd.notna(ibh) and pd.notna(ibl) and not post_ib_bars.empty:
        # Get last pre-IB close
        last_close = pre_ib_bars["close"].iloc[-1] if not pre_ib_bars.empty else None
        
        # Re-evaluate current state against IB
        sm.set_references(ibh, ibl, "post_ib")
        
        # If in discovery, check if still valid against IB
        if sm.state == "DISCOVERY_UP" and last_close is not None:
            if last_close <= ibh:
                # Rejected by IB boundary
                if sm.episode_start_time is not None:
                    ep = sm._close_episode("R", pre_ib_bars["datetime"].iloc[-1], 
                                          len(pre_ib_bars)-1, prior_vah)
                    pre_ib_episodes.append(ep)
                sm._start_balance(post_ib_bars["datetime"].iloc[0] if not post_ib_bars.empty else None, 0, np.nan)
        
        elif sm.state == "DISCOVERY_DOWN" and last_close is not None:
            if last_close >= ibl:
                if sm.episode_start_time is not None:
                    ep = sm._close_episode("R", pre_ib_bars["datetime"].iloc[-1],
                                          len(pre_ib_bars)-1, prior_val)
                    pre_ib_episodes.append(ep)
                sm._start_balance(post_ib_bars["datetime"].iloc[0] if not post_ib_bars.empty else None, 0, np.nan)
        
        # --- Phase 2: Post-IB (7:30am - 12:55pm) against IBH/IBL ---
        for i, (idx, row) in enumerate(post_ib_bars.iterrows()):
            is_last = (i == len(post_ib_bars) - 1)
            atr = row.get(atr_col, np.nan)
            vwap = row.get("vwap", np.nan)
            
            eps = sm.process_bar(
                idx=i,
                high=row["high"],
                low=row["low"],
                close=row["close"],
                dt=row["datetime"],
                atr=atr,
                vwap=vwap,
                is_last_bar=is_last
            )
            post_ib_episodes.extend(eps)
    
    return pre_ib_episodes, post_ib_episodes


def encode_discovery_states(df, timeframe=None):
    """Process all sessions and extract two-phase discovery episodes."""
    if timeframe is None:
        timeframe = CONFIG['timeframe']
    
    all_pre_ib = []
    all_post_ib = []
    session_dates = sorted(df["session_date"].unique())
    
    for sd in session_dates:
        session = df[df["session_date"] == sd]
        if session.empty:
            continue
        
        # Get reference values
        prior_vah = session["prior_vah"].iloc[0] if "prior_vah" in session.columns else np.nan
        prior_val = session["prior_val"].iloc[0] if "prior_val" in session.columns else np.nan
        ibh = session["ibh"].iloc[0] if "ibh" in session.columns else np.nan
        ibl = session["ibl"].iloc[0] if "ibl" in session.columns else np.nan
        ib_range = session["ib_range"].iloc[0] if "ib_range" in session.columns else np.nan
        or_range = session["or_range"].iloc[0] if "or_range" in session.columns else np.nan
        
        # Skip sessions without prior-session profile
        if pd.isna(prior_vah) or pd.isna(prior_val):
            continue
        
        pre_eps, post_eps = process_session(
            session, prior_vah, prior_val, ibh, ibl, ib_range, or_range, sd, timeframe
        )
        all_pre_ib.extend(pre_eps)
        all_post_ib.extend(post_eps)
    
    print(f"Pre-IB episodes: {len(all_pre_ib)}, Post-IB episodes: {len(all_post_ib)}")
    return all_pre_ib, all_post_ib


def save_episodes(episodes, output_path):
    """Save episodes to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ep_df = pd.DataFrame(episodes) if episodes else pd.DataFrame()
    ep_df.to_csv(output_path, index=False)
    print(f"Saved {len(ep_df)} episodes to {output_path}")
    return ep_df


if __name__ == "__main__":
    from load_data import load_and_validate
    from session_profile import compute_session_profiles, merge_profiles_to_bars
    from opening_range import compute_opening_range, merge_or_to_bars
    from vwap_engine import compute_session_vwap
    from volatility import compute_atr
    
    df = load_and_validate()
    profiles = compute_session_profiles(df)
    df = merge_profiles_to_bars(df, profiles)
    
    or_df = compute_opening_range(df, profiles)
    df = merge_or_to_bars(df, or_df)
    
    df = compute_session_vwap(df)
    df = compute_atr(df)
    
    pre_ib_episodes, post_ib_episodes = encode_discovery_states(df)
    
    save_episodes(pre_ib_episodes, os.path.join(OUTPUT_DIR, "episodes_pre_ib.csv"))
    save_episodes(post_ib_episodes, os.path.join(OUTPUT_DIR, "episodes_post_ib.csv"))
    
    print(f"\n--- Episode Summary ---")
    print(f"Pre-IB: {len(pre_ib_episodes)}")
    print(f"Post-IB: {len(post_ib_episodes)}")
    
    all_eps = pre_ib_episodes + post_ib_episodes
    if all_eps:
        ep_df = pd.DataFrame(all_eps)
        print(f"Total: {len(ep_df)}")
        print(f"By outcome: {ep_df['terminal_outcome'].value_counts().to_dict()}")
