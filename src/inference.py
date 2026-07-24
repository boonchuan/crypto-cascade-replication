"""
inference.py — the remaining new analyses for the revision:

1. bootstrap_rank_cis:   bootstrap CIs on October 10's z-scores and
                         leave-one-out rank stability (R1.S5, R2.12)
2. empirical_exceedance: empirical tail probabilities for the composites
3. trivariate_var:       VAR(spot, futures, mark) returns with system
                         Granger tests and IRFs (R1.6, R2.2, R2.11)
4. rolling_correlations: metric-pair dependence, cascade vs baseline
                         (R2.2), and SOL-BTC time-varying correlation
                         (R2.4)

Usage (after run_grid has produced the baseline metrics table):
  python -m src.inference
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .events import EventSpec
from .fetch_data import cache_path
from .metrics import METRICS, EXTREME_SIGN

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
BASELINE = EventSpec(3.0, 30, 6.0)
OCT10 = pd.Timestamp("2025-10-10 21:19:00", tz="UTC")
RNG = np.random.default_rng(20251010)


# ---------------------------------------------------------------- 1 & 2

def bootstrap_rank_cis(n_boot: int = 10_000) -> pd.DataFrame:
    """Resample the event sample (holding October 10's raw values fixed)
    and recompute its z-score against each resample's mean/std.
    Reports 95% CIs on z, plus leave-one-out min/max rank."""
    tbl = pd.read_csv(OUT_DIR / f"metrics_{BASELINE.label}.csv",
                      parse_dates=["trough_ts"]).set_index("trough_ts")
    oct10_ts = min(tbl.index, key=lambda t: abs(t - OCT10))

    rows = []
    for m in METRICS:
        vals = tbl[m].values
        x = float(tbl.loc[oct10_ts, m])
        others = tbl[m].drop(oct10_ts).values

        # Bootstrap z: resample the full sample with replacement
        idx = RNG.integers(0, len(vals), size=(n_boot, len(vals)))
        samp = vals[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_boot = (x - samp.mean(axis=1)) / samp.std(axis=1, ddof=1)
        z_boot = z_boot[np.isfinite(z_boot)]
        if len(z_boot):
            lo, hi = np.percentile(z_boot, [2.5, 97.5])
        else:
            lo = hi = np.nan

        # Leave-one-out rank stability
        extreme = (vals * EXTREME_SIGN[m])
        x_e = x * EXTREME_SIGN[m]
        loo_ranks = []
        for j in range(len(vals)):
            if tbl.index[j] == oct10_ts:
                continue
            rest = np.delete(extreme, j)
            loo_ranks.append(int((rest > x_e).sum() + 1))
        # Empirical exceedance: share of other events at least as extreme
        exceed = float(((others * EXTREME_SIGN[m]) >= x_e).mean())

        rows.append({
            "metric": m, "oct10_value": round(x, 3),
            "z_ci_lo": round(float(lo), 2), "z_ci_hi": round(float(hi), 2),
            "loo_rank_min": min(loo_ranks), "loo_rank_max": max(loo_ranks),
            "empirical_exceedance": round(exceed, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "bootstrap_rank_cis.csv", index=False)
    return out


# -------------------------------------------------------------------- 3

def trivariate_var(window_hours: float = 6.0, maxlags: int = 10):
    """VAR on 1-minute log returns of BTC spot, perpetual, and mark price
    over a window bracketing the October 10 cascade. Reports lag order
    (AIC/BIC), within-system Granger causality tests, and saves IRFs.

    Requires statsmodels.
    """
    from statsmodels.tsa.api import VAR

    def _load(name, sym, kind, col):
        df = pd.read_parquet(cache_path(sym, kind)).set_index("ts")
        return df[col].rename(name)

    px = pd.concat([
        _load("spot", "BTCUSDT", "spot", "close"),
        _load("fut", "BTCUSDT", "perp", "close"),
        _load("mark", "BTCUSDT", "mark", "close"),
    ], axis=1)

    w0 = OCT10 - pd.Timedelta(hours=window_hours / 2)
    w1 = OCT10 + pd.Timedelta(hours=window_hours / 2)
    ret = np.log(px.loc[w0:w1]).diff().dropna()

    # Guard: near-perfect collinearity among the three return series makes
    # the residual covariance singular. Real spot/futures/mark data are
    # highly but not perfectly correlated; this trips only on degenerate
    # (e.g., synthetic) inputs.
    corr = ret.corr().values
    off_diag_max = np.abs(corr[np.triu_indices_from(corr, k=1)]).max()
    if off_diag_max > 0.9999:
        raise ValueError(
            f"Return series are near-perfectly collinear "
            f"(max |corr| = {off_diag_max:.6f}); VAR covariance would be "
            f"singular. Check the input data.")

    model = VAR(ret)
    sel = model.select_order(maxlags)
    p = max(int(sel.aic), 1)
    res = model.fit(p)

    lines = [f"Trivariate VAR: BTC spot/futures/mark 1m log-returns",
             f"Window: {w0} to {w1}  (n = {len(ret)})",
             f"Lag order (AIC): {sel.aic}, (BIC): {sel.bic}; fitted p = {p}",
             ""]
    for caused in ret.columns:
        causing = [c for c in ret.columns if c != caused]
        gc = res.test_causality(caused, causing, kind="f")
        lines.append(f"H0: {{{', '.join(causing)}}} do not Granger-cause "
                     f"{caused}: F = {gc.test_statistic:.2f}, "
                     f"p = {gc.pvalue:.4f}")
    # Pairwise within-system tests
    lines.append("")
    for a in ret.columns:
        for b in ret.columns:
            if a == b:
                continue
            gc = res.test_causality(b, [a], kind="f")
            lines.append(f"H0: {a} does not Granger-cause {b}: "
                         f"F = {gc.test_statistic:.2f}, p = {gc.pvalue:.4f}")

    report = "\n".join(lines)
    (OUT_DIR / "var_granger_report.txt").write_text(report + "\n")
    print(report)

    irf = res.irf(10)
    fig = irf.plot(orth=False)
    fig.savefig(OUT_DIR / "var_irf.png", dpi=200, bbox_inches="tight")
    return res


# -------------------------------------------------------------------- 4

def rolling_correlations(pair_window_min: int = 60):
    """(a) SOL-BTC rolling return correlation across the October event
    vs a baseline day; (b) saved series for the metric-dependence figure."""
    btc = pd.read_parquet(cache_path("BTCUSDT", "spot")).set_index("ts")
    sol = pd.read_parquet(cache_path("SOLUSDT", "spot")).set_index("ts")
    ret = pd.concat([
        np.log(btc["close"]).diff().rename("btc"),
        np.log(sol["close"]).diff().rename("sol"),
    ], axis=1).dropna()

    roll = ret["btc"].rolling(pair_window_min).corr(ret["sol"])

    event_day = roll.loc["2025-10-10":"2025-10-11"]
    baseline_day = roll.loc["2025-09-10":"2025-09-11"]  # same weekday count back

    out = pd.DataFrame({
        "event_window_corr": event_day.describe(),
        "baseline_window_corr": baseline_day.describe(),
    })
    out.to_csv(OUT_DIR / "sol_btc_rolling_corr_summary.csv")
    roll.loc["2025-10-09":"2025-10-12"].to_csv(
        OUT_DIR / "sol_btc_rolling_corr_event.csv")
    print(out)
    return roll


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("== Bootstrap CIs and rank stability ==")
    print(bootstrap_rank_cis().to_string(index=False))
    print("\n== Trivariate VAR ==")
    trivariate_var()
    print("\n== Rolling correlations ==")
    rolling_correlations()


if __name__ == "__main__":
    main()
