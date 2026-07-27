"""make_var_table.py - Appendix: trivariate VAR diagnostics.

Refits the trivariate VAR in BTC spot / perpetual futures / mark-price
1-minute log returns over the event window and emits the full diagnostic
table: lag-order selection, Granger-causality tests with degrees of
freedom, VAR stability roots, and residual autocorrelation.

Reported as a dependence diagnostic, NOT as causal evidence: the mark
price is mechanically constructed from index and premium inputs derived
from traded prices.

Run from the repository root:
    python src/make_var_table.py

Input:  data/BTCUSDT_spot_1m.parquet, BTCUSDT_perp_1m.parquet,
        BTCUSDT_mark_1m.parquet
Output: output/var_table.md, output/var_diagnostics.csv
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

START = "2025-10-10 18:19:00+00:00"
END = "2025-10-11 00:19:00+00:00"
MAXLAGS = 40
SERIES = {"spot": "data/BTCUSDT_spot_1m.parquet",
          "fut": "data/BTCUSDT_perp_1m.parquet",
          "mark": "data/BTCUSDT_mark_1m.parquet"}
OUT_MD = "output/var_table.md"
OUT_CSV = "output/var_diagnostics.csv"


def load(path: str) -> pd.Series:
    df = pd.read_parquet(path)
    # locate the timestamp column whatever it is called
    ts = next((c for c in df.columns
               if c.lower() in ("ts", "timestamp", "open_time", "time", "date")), None)
    if ts is not None:
        df = df.set_index(ts)
    df.index = pd.to_datetime(df.index, utc=True)
    close = next(c for c in df.columns if c.lower() == "close")
    return df[close].astype(float).sort_index()


def main() -> None:
    px = pd.DataFrame({k: load(v) for k, v in SERIES.items()})
    px = px.loc[START:END].dropna()
    r = np.log(px).diff().dropna() * 100.0
    n = len(r)
    print(f"window {START} to {END}: {len(px)} price obs, {n} return obs")

    sel = VAR(r).select_order(maxlags=MAXLAGS)
    p = int(sel.bic)
    res = VAR(r).fit(p)
    k = 1 + len(r.columns) * p            # params per equation
    df_denom = res.nobs - k

    rows = []
    names = list(r.columns)
    for target in names:
        others = [c for c in names if c != target]
        t = res.test_causality(target, others, kind="f")
        rows.append(dict(direction=f"{{{', '.join(others)}}} -> {target}",
                         stat=t.test_statistic, df_num=len(others) * p,
                         df_den=df_denom, pvalue=t.pvalue))
        for src in others:
            t = res.test_causality(target, [src], kind="f")
            rows.append(dict(direction=f"{src} -> {target}",
                             stat=t.test_statistic, df_num=p,
                             df_den=df_denom, pvalue=t.pvalue))

    tab = pd.DataFrame(rows)
    # statsmodels VARResults.roots are roots of the characteristic
    # polynomial: stability requires every modulus to EXCEED 1.
    roots = np.abs(res.roots)
    stable = bool(res.is_stable(verbose=False))
    resid = pd.DataFrame(np.asarray(res.resid), columns=names)
    lb = {c: acorr_ljungbox(resid[c], lags=[10], return_df=True).iloc[0]
          for c in names}

    tab.to_csv(OUT_CSV, index=False)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("Table D1. Trivariate VAR Diagnostics: BTC Spot, Perpetual "
                "Futures, and Mark Price\n\n")
        f.write(f"Window: {START} to {END} ({n} one-minute return "
                f"observations). Lag order selected by AIC: {int(sel.aic)}; "
                f"BIC: {int(sel.bic)}; HQIC: {int(sel.hqic)}; fitted p = {p}.\n\n")
        f.write("| Direction | F | df | p |\n|---|---|---|---|\n")
        for _, row in tab.iterrows():
            pv = "<0.0001" if row.pvalue < 1e-4 else f"{row.pvalue:.4f}"
            f.write(f"| {row.direction} | {row.stat:.2f} | "
                    f"({int(row.df_num)}, {int(row.df_den)}) | {pv} |\n")
        verdict = ("the system is stable" if stable
                   else "the stability condition is NOT satisfied")
        f.write(f"\nStability: the {len(roots)} roots of the characteristic "
                f"polynomial have moduli between {roots.min():.3f} and "
                f"{roots.max():.3f}; stability requires every modulus to "
                f"exceed unity, and {verdict}.\n\n")
        f.write("Residual autocorrelation (Ljung-Box, 10 lags):\n\n")
        f.write("| Equation | Q(10) | p |\n|---|---|---|\n")
        for eq in names:
            f.write(f"| {eq} | {lb[eq]['lb_stat']:.2f} | "
                    f"{lb[eq]['lb_pvalue']:.4f} |\n")
        f.write("\nNotes: F-tests are Granger-causality tests within the "
                "fitted VAR. Because the mark price is constructed by the "
                "exchange from an index and a premium component derived from "
                "traded prices, these tests are reported as short-horizon "
                "predictive-dependence diagnostics and do not identify "
                "directional causation.\n")

    print(f"\nwrote {OUT_MD} and {OUT_CSV}")
    print(f"lag selection: AIC={int(sel.aic)} BIC={int(sel.bic)} "
          f"HQIC={int(sel.hqic)}; fitted p={p}; nobs={res.nobs}; "
          f"df_den={df_denom}")
    print(f"root moduli: min {roots.min():.3f} max {roots.max():.3f}; is_stable={stable}")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
