"""
metrics.py — Appendix A, Steps 5–6: per-event microstructure metrics,
z-scoring, and composite anomaly measures.

The 11 metrics of Table 6, computed within the relocated analysis window
[t* - before, t* + after] with a 24-hour pre-window volume baseline.
"""

import numpy as np
import pandas as pd

from .events import Event, EventSpec

METRICS = [
    "btc_spot_dd", "btc_fut_dd", "basis_swing", "mark_minus_spot",
    "max_intramin_spread", "vol_surge_ratio", "peak_vol_ratio",
    "vol_lead_min", "sol_spot_dd", "sol_fut_dd", "sol_gap_pp",
]

# Direction of "more extreme" per metric: -1 = more negative is extreme,
# +1 = larger is extreme. Used for ranking and for sign-flipping in the
# sum-of-|z| composite (Appendix A Step 6).
EXTREME_SIGN = {
    "btc_spot_dd": -1, "btc_fut_dd": -1, "basis_swing": +1,
    "mark_minus_spot": -1, "max_intramin_spread": +1,
    "vol_surge_ratio": +1, "peak_vol_ratio": +1, "vol_lead_min": +1,
    "sol_spot_dd": -1, "sol_fut_dd": -1, "sol_gap_pp": +1,
}


def _dd_pct(close: pd.Series) -> float:
    """Peak-to-trough drawdown within a window, percent (negative)."""
    peak = close.cummax()
    return float(((close / peak) - 1.0).min() * 100.0)


def compute_event_metrics(ev: Event, spec: EventSpec,
                          data: dict[str, pd.DataFrame]) -> dict:
    """Appendix A Step 5 for a single event.

    `data` maps series name -> DataFrame indexed by ts:
      'btc_spot', 'btc_perp', 'btc_mark', 'sol_spot', 'sol_perp'
    """
    before, after = spec.analysis_bounds()
    w0 = ev.trough_ts - pd.Timedelta(minutes=before)
    w1 = ev.trough_ts + pd.Timedelta(minutes=after)
    b0 = w0 - pd.Timedelta(hours=24)

    bs = data["btc_spot"].loc[w0:w1]
    bf = data["btc_perp"].loc[w0:w1]
    bm = data["btc_mark"].loc[w0:w1]
    ss = data["sol_spot"].loc[w0:w1]
    sf = data["sol_perp"].loc[w0:w1]
    baseline_vol = data["btc_spot"].loc[b0:w0 - pd.Timedelta(minutes=1),
                                        "volume"]

    out = {"trough_ts": ev.trough_ts}

    # (i)-(ii) BTC drawdowns
    out["btc_spot_dd"] = _dd_pct(bs["close"])
    out["btc_fut_dd"] = _dd_pct(bf["close"])

    # (iii) basis swing = max(F-S) - min(F-S), closes
    basis = bf["close"] - bs["close"]
    out["basis_swing"] = float(basis.max() - basis.min())

    # (iv) mark-to-spot undershoot, SIMULTANEOUS definition (v1.1):
    # the most negative same-minute close-to-close gap over the window.
    # (v1.0 compared the mark candle LOW to the spot candle CLOSE, which
    # mixes timestamps within the minute.) The candle-low comparison is
    # retained as an auxiliary diagnostic, clearly labelled.
    mark_spot_gap = bm["close"] - bs["close"]
    out["mark_minus_spot"] = float(mark_spot_gap.min())
    out["mark_minus_spot_minute"] = str(mark_spot_gap.idxmin())
    mark_low_ts = bm["low"].idxmin()
    out["aux_marklow_minus_spotlow"] = float(bm.loc[mark_low_ts, "low"]
                                             - bs.loc[mark_low_ts, "low"])

    # (v) max intra-minute price RANGE on BTC spot, percent
    # (v1.1 rename: this is a candle high-low range, not a bid-ask
    # spread; column name kept for schema continuity, display label
    # changed in make_tables.)
    out["max_intramin_spread"] = float(
        ((bs["high"] - bs["low"]) / bs["low"] * 100.0).max())

    # (vi)-(vii) volume ratios vs 24h baseline; authoritative volume
    # diagnostics recorded alongside (v1.1, Issue 4)
    base_mean = float(baseline_vol.mean())
    out["vol_surge_ratio"] = float(bs["volume"].mean() / base_mean)
    out["peak_vol_ratio"] = float(bs["volume"].max() / base_mean)
    out["diag_baseline_mean_vol"] = base_mean
    out["diag_baseline_median_vol"] = float(baseline_vol.median())
    out["diag_window_mean_vol"] = float(bs["volume"].mean())
    out["diag_window_peak_vol"] = float(bs["volume"].max())
    out["diag_window_n_obs"] = int(len(bs))

    # (viii) volume lead: trough minute minus volume-peak minute
    vol_peak_ts = bs["volume"].idxmax()
    price_trough_ts = bs["close"].idxmin()
    out["vol_lead_min"] = float(
        (price_trough_ts - vol_peak_ts) / pd.Timedelta(minutes=1))

    # (ix) SOL metrics
    out["sol_spot_dd"] = _dd_pct(ss["close"])
    out["sol_fut_dd"] = _dd_pct(sf["close"])
    out["sol_gap_pp"] = abs(out["sol_fut_dd"]) - abs(out["sol_spot_dd"])

    return out


def build_metric_table(events: list[Event], spec: EventSpec,
                       data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [compute_event_metrics(ev, spec, data) for ev in events]
    return pd.DataFrame(rows).set_index("trough_ts")


def zscore_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Z-scores against the within-sample empirical mean and std."""
    z = (tbl[METRICS] - tbl[METRICS].mean()) / tbl[METRICS].std(ddof=1)
    return z


def ranks_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 = most extreme, per EXTREME_SIGN direction."""
    ranks = {}
    for m in METRICS:
        vals = tbl[m] * EXTREME_SIGN[m]  # so larger = more extreme
        ranks[m] = vals.rank(ascending=False, method="min").astype("Int64")
    return pd.DataFrame(ranks, index=tbl.index)


def composite_scores(tbl: pd.DataFrame) -> pd.DataFrame:
    """Sum of |z| and Mahalanobis distance in z-space (pseudo-inverse
    if the covariance matrix is numerically singular). Appendix A Step 6."""
    z = zscore_table(tbl)
    sum_abs_z = z.abs().sum(axis=1)

    if len(tbl) > len(METRICS) + 1 and not z.isna().any().any():
        zv = z.values
        cov = np.cov(zv, rowvar=False)
        icov = np.linalg.pinv(cov)  # pseudo-inverse per Appendix A Step 6
        d = zv - zv.mean(axis=0)
        maha = np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d))
    else:
        # Too few events for a meaningful covariance estimate
        maha = np.full(len(tbl), np.nan)

    out = pd.DataFrame({"sum_abs_z": sum_abs_z,
                        "mahalanobis": maha}, index=tbl.index)
    out["rank_sum_abs_z"] = out["sum_abs_z"].rank(
        ascending=False, method="min").astype("Int64")
    out["rank_mahalanobis"] = out["mahalanobis"].rank(
        ascending=False, method="min").astype("Int64")
    return out
