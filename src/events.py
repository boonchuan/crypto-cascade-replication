"""
events.py — Appendix A, Steps 2–4: drawdown scan, greedy de-overlapping,
crash-window relocation.

All three event-selection parameters are exposed so the same code produces
both the baseline 62-event sample (threshold=3.0, window=30, sep=6) and
every cell of the Appendix B sensitivity grid.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSpec:
    threshold_pct: float = 3.0   # minimum drawdown magnitude, percent
    window_min: int = 30         # rolling scan window length, minutes
    sep_hours: float = 6.0       # greedy de-overlap separation, hours

    @property
    def label(self) -> str:
        return (f"dd{self.threshold_pct:g}pct_"
                f"w{self.window_min}min_sep{self.sep_hours:g}h")

    def analysis_bounds(self) -> tuple[int, int]:
        """Minutes before/after the trough for the relocated analysis
        window [t*-before, t*+after], INCLUSIVE of both endpoints, sized
        so the window contains exactly `window_min` one-minute
        observations: before + after + 1 = window_min. Baseline (30 min)
        gives [t*-10, t*+19] = 30 observations. (v1.1 fix: v1.0 used
        [t*-10, t*+20] = 31 observations under pandas inclusive .loc.)"""
        before = self.window_min // 3
        after = self.window_min - before - 1
        return before, after


@dataclass
class Event:
    trough_ts: pd.Timestamp      # BTC spot trough minute (analysis anchor)
    scan_end_ts: pd.Timestamp    # candidate minute t from the scan
    drawdown_pct: float          # scan-window drawdown that triggered selection


def rolling_drawdown(close: pd.Series, window_min: int) -> pd.Series:
    """Directional trailing drawdown, in percent:

        DD(t) = C(t) / max(C(t-w..t)) - 1

    i.e. the current price relative to the highest price over the
    trailing window. This is ~0 during rallies and most negative at a
    trough that follows a peak, enforcing peak-before-trough ordering.
    (v1.1 fix: the v1.0 scan used min(window)/max(window)-1, which is
    direction-agnostic and can flag pure rallies as drawdowns.)"""
    roll_max = close.rolling(window_min, min_periods=window_min).max()
    return (close / roll_max - 1.0) * 100.0


def select_events(btc_spot: pd.DataFrame, spec: EventSpec) -> list[Event]:
    """Appendix A Steps 2–4 for one parameterization."""
    df = btc_spot.set_index("ts")
    dd = rolling_drawdown(df["close"], spec.window_min)

    candidates = dd[dd <= -spec.threshold_pct].sort_values()  # most negative first
    sep = pd.Timedelta(hours=spec.sep_hours)

    selected: list[Event] = []
    selected_ts: list[pd.Timestamp] = []

    for t, d in candidates.items():
        if any(abs(t - s) <= sep for s in selected_ts):
            continue
        # Step 4: locate the BTC spot trough within the scan window
        win = df["close"].loc[t - pd.Timedelta(minutes=spec.window_min): t]
        trough_ts = win.idxmin()
        selected.append(Event(trough_ts=trough_ts, scan_end_ts=t,
                              drawdown_pct=float(d)))
        selected_ts.append(t)

    selected.sort(key=lambda e: e.trough_ts)
    return selected


def find_event_containing(events: list[Event], target: pd.Timestamp,
                          tol_hours: float = 12.0) -> int | None:
    """Return the index of the event whose trough is nearest to `target`
    within `tol_hours`, else None. Used to locate October 10 in each
    grid cell (its trough minute can shift slightly under alternative
    window lengths)."""
    best, best_gap = None, pd.Timedelta(hours=tol_hours)
    for i, ev in enumerate(events):
        gap = abs(ev.trough_ts - target)
        if gap < best_gap:
            best, best_gap = i, gap
    return best
