"""
config.py — Centralized configuration for the Markov Discovery Model pipeline.

All configurable parameters are defined here. Modify values as needed.
"""

# =============================================
# Default configuration (user-adjustable)
# =============================================
CONFIG = {
    # Timeframe for resampling (input is 1-min or 5-min)
    # Options: '5m', '15m', '30m'
    'timeframe': '5m',
    
    # ATR lookback period (bars on resampled data)
    'atr_window': 5,
    
    # Opening Range duration in minutes from 6:30am
    'opening_window_minutes': 15,
    
    # Minimum sustained duration for Acceptance outcome (minutes)
    'acceptance_threshold_minutes': 30,
    
    # Window for opposite boundary touch rejection (minutes)
    'two_sided_threshold_minutes': 30,
    
    # Analysis window years
    'data_start_year': 2025,
    'data_end_year': 2025,
}

# =============================================
# Session timing constants (Pacific Time)
# =============================================
SESSION_TIMES = {
    # RTH session start
    'rth_start': (6, 30),  # 6:30am PT
    
    # RTH session end
    'rth_end': (13, 0),  # 1:00pm PT
    
    # Pre-IB phase (IB formation)
    'pre_ib_start': (6, 30),  # 6:30am PT
    'pre_ib_end': (7, 25),    # 7:25am PT (last bar of IB)
    
    # Post-IB phase (main session)
    'post_ib_start': (7, 30),  # 7:30am PT
    'post_ib_end': (12, 55),   # 12:55pm PT (last bar)
    
    # Overnight session (for volume profile)
    'overnight_start': (15, 0),  # 3:00pm PT (previous day)
    'overnight_end': (6, 29),    # 6:29am PT (current day)
    
    # Prior RTH session (previous day)
    'prior_rth_start': (6, 30),  # 6:30am PT
    'prior_rth_end': (13, 0),    # 1:00pm PT
}

# =============================================
# Derived constants
# =============================================
def get_acceptance_bars(timeframe):
    """Get number of bars needed for acceptance threshold."""
    minutes = CONFIG['acceptance_threshold_minutes']
    if timeframe == '5m':
        return minutes // 5  # 6 bars for 30 min
    elif timeframe == '15m':
        return minutes // 15  # 2 bars
    elif timeframe == '30m':
        return minutes // 30  # 1 bar
    return 6

def get_two_sided_bars(timeframe):
    """Get number of bars for two-sided rejection window."""
    minutes = CONFIG['two_sided_threshold_minutes']
    if timeframe == '5m':
        return minutes // 5
    elif timeframe == '15m':
        return minutes // 15
    elif timeframe == '30m':
        return minutes // 30
    return 6

# Tick size for MNQ (Micro E-mini Nasdaq)
TICK_SIZE = 0.25

# Value Area percentage
VALUE_AREA_PCT = 0.70

# Failure count cap
FAILURE_COUNT_CAP = 4

# Extension bins for transition counting (ATR units)
EXTENSION_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, float('inf')]
EXTENSION_BIN_LABELS = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
