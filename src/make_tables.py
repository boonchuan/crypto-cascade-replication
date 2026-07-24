"""
make_tables.py — regenerate manuscript Tables 6 and 7 from the baseline
58-event metrics CSV.

Outputs (markdown, ready to transcribe into the manuscript):
  output/table6.md   October 10's value, rank, percentile, z per metric
  output/table7.md   Top 10 events by SOL futures-spot gap

Usage (from repo root):
  python -m src.make_tables
"""

from pathlib import Path

import pandas as pd

from .events import EventSpec
from .metrics import METRICS, EXTREME_SIGN

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
BASELINE = EventSpec(3.0, 30, 6.0)
OCT10_DATE = "2025-10-10"

DISPLAY = {
    "btc_spot_dd":         ("BTC spot DD (30-min)",        "{:.2f}%"),
    "btc_fut_dd":          ("BTC futures DD (30-min)",     "{:.2f}%"),
    "basis_swing":         ("BTC basis swing",             "${:,.0f}"),
    "mark_minus_spot":     ("BTC mark - spot at trough",   "-${:,.0f}"),
    "max_intramin_spread": ("BTC max intra-minute spread", "{:.2f}%"),
    "vol_surge_ratio":     ("BTC volume surge ratio",      "{:.2f}x"),
    "peak_vol_ratio":      ("BTC peak volume ratio",       "{:.2f}x"),
    "vol_lead_min":        ("BTC volume lead (minutes)",   "{:.1f}"),
    "sol_spot_dd":         ("SOL spot DD",                 "{:.2f}%"),
    "sol_fut_dd":          ("SOL futures DD",              "{:.2f}%"),
    "sol_gap_pp":          ("SOL futures-spot gap (pp)",   "{:.2f}"),
}


def main():
    tbl = pd.read_csv(OUT_DIR / f"metrics_{BASELINE.label}.csv",
                      parse_dates=["trough_ts"]).set_index("trough_ts")
    n = len(tbl)
    z = (tbl[METRICS] - tbl[METRICS].mean()) / tbl[METRICS].std(ddof=1)

    oct10 = [t for t in tbl.index if str(t.date()) == OCT10_DATE]
    assert len(oct10) == 1, f"expected one Oct 10 event, found {len(oct10)}"
    o = oct10[0]

    lines = [f"**Table 6: Rank of October 10 Within {n}-Event "
             f"Empirical Distribution**", "",
             "| Metric | Oct 10 Value | Rank | Pctile | Z-score |",
             "|---|---|---|---|---|"]
    for m in METRICS:
        label, fmt = DISPLAY[m]
        val = tbl.loc[o, m]
        val_str = fmt.format(abs(val) if fmt.startswith("-$") else val)
        rank = int(tbl[f"{m}_rank"].loc[o]) if f"{m}_rank" in tbl.columns \
            else int((tbl[m] * EXTREME_SIGN[m]).rank(
                ascending=False, method="min").loc[o])
        pctile = (rank - 1) / n * 100.0
        zval = z.loc[o, m]
        lines.append(f"| {label} | {val_str} | {rank} / {n} "
                     f"| {pctile:.1f}% | {zval:+.2f} |")
    lines += [
        "",
        f"Notes: Sample is {n} non-overlapping 30-minute BTC drawdowns "
        ">=3% between January 2024 and April 2026, identified from "
        "Binance 1-minute klines. Z-scores are computed against the "
        "empirical mean and standard deviation of each metric within "
        f"the {n}-event sample. Rank 1 denotes the most extreme "
        "observation; percentile gives the share of events more "
        "extreme than October 10.",
    ]
    (OUT_DIR / "table6.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")

    top = tbl.nlargest(10, "sol_gap_pp")
    lines7 = [f"**Table 7: Top 10 Events in the {n}-Event Sample "
              "Ranked by SOL Futures-Spot Gap**", "",
              "| Rank | Date | BTC DD 30-min | SOL Gap (pp) "
              "| BTC Mark - Spot ($) |",
              "|---|---|---|---|---|"]
    for i, (ts, r) in enumerate(top.iterrows(), 1):
        bold = "**" if str(ts.date()) == OCT10_DATE else ""
        lines7.append(
            f"| {i} | {bold}{ts.strftime('%b %d, %Y')}{bold} "
            f"| {r['btc_spot_dd']:.2f}% | {r['sol_gap_pp']:.2f} "
            f"| {r['mark_minus_spot']:,.0f} |")
    ratio = (top["sol_gap_pp"].iloc[0] / top["sol_gap_pp"].iloc[1])
    lines7 += [
        "",
        f"Runner-up gap: {top['sol_gap_pp'].iloc[1]:.2f} pp "
        f"({top.index[1].strftime('%b %d, %Y')}); October 10 multiple: "
        f"{ratio:.1f}x  <- update Section 4.3 / 6.2 text with these.",
    ]
    (OUT_DIR / "table7.md").write_text("\n".join(lines7) + "\n",
                                       encoding="utf-8")

    med = tbl["sum_abs_z"].median()
    extras = [
        f"n_events: {n}",
        f"Oct 10 chronological event index (1-based): "
        f"{list(tbl.sort_index().index).index(o) + 1}",
        f"sum|z| median: {med:.2f}; Oct 10 multiple of median: "
        f"{tbl.loc[o, 'sum_abs_z'] / med:.1f}x",
        f"Section 4.2 z-values: vol_surge z = {z.loc[o, 'vol_surge_ratio']:+.2f}, "
        f"peak_vol z = {z.loc[o, 'peak_vol_ratio']:+.2f}",
        f"Section 4.3 z-value: sol_gap z = {z.loc[o, 'sol_gap_pp']:+.2f}",
    ]
    (OUT_DIR / "table_support_numbers.txt").write_text(
        "\n".join(extras) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print()
    print("\n".join(lines7))
    print()
    print("\n".join(extras))


if __name__ == "__main__":
    main()
