"""
run_grid.py â€” Appendix B sensitivity grid.

Reruns the full Appendix A pipeline under every combination of:
  drawdown threshold : 2, 3, 4, 5 (%)
  scan/analysis window : 15, 30, 60 (minutes)
  de-overlap separation : 3, 6, 12 (hours)

For each of the 36 specifications, reports the number of events, October
10's rank on each of the 11 metrics, and its rank on both composite
anomaly measures. Baseline spec (3%, 30min, 6h) is flagged.

Outputs:
  output/appendix_b_grid.csv        full results, one row per spec
  output/appendix_b_table.md        formatted Table B1
  output/metrics_<spec>.csv         per-event metric tables (baseline spec
                                    always saved; others with --save-all)

Usage:
  python -m src.run_grid
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

from .events import EventSpec, select_events, find_event_containing
from .fetch_data import cache_path
from .metrics import (METRICS, build_metric_table, ranks_table,
                      composite_scores, zscore_table)

OUT_DIR = Path(__file__).resolve().parents[1] / "output"

THRESHOLDS = [2.0, 3.0, 4.0, 5.0]
WINDOWS = [15, 30, 60]
SEPARATIONS = [3.0, 6.0, 12.0]
BASELINE = EventSpec(3.0, 30, 6.0)

# October 10, 2025 BTC trough (Table 1 of the paper)
OCT10 = pd.Timestamp("2025-10-10 21:19:00", tz="UTC")


def load_all() -> dict[str, pd.DataFrame]:
    series = {
        "btc_spot": ("BTCUSDT", "spot"),
        "btc_perp": ("BTCUSDT", "perp"),
        "btc_mark": ("BTCUSDT", "mark"),
        "sol_spot": ("SOLUSDT", "spot"),
        "sol_perp": ("SOLUSDT", "perp"),
    }
    data = {}
    for name, (sym, kind) in series.items():
        path = cache_path(sym, kind)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing â€” run `python -m src.fetch_data` first")
        data[name] = pd.read_parquet(path).set_index("ts").sort_index()
    return data


def run_spec(spec: EventSpec, data: dict[str, pd.DataFrame],
             save_metrics: bool = False) -> dict:
    btc_spot = data["btc_spot"].reset_index()
    events = select_events(btc_spot, spec)
    row = {
        "threshold_pct": spec.threshold_pct,
        "window_min": spec.window_min,
        "sep_hours": spec.sep_hours,
        "n_events": len(events),
        "is_baseline": spec == BASELINE,
    }

    idx = find_event_containing(events, OCT10)
    if idx is None:
        row["oct10_found"] = False
        return row
    row["oct10_found"] = True

    tbl = build_metric_table(events, spec, data)
    ranks = ranks_table(tbl)
    comps = composite_scores(tbl)
    zs = zscore_table(tbl)

    def _i(v):
        return int(v) if pd.notna(v) else None

    oct10_ts = tbl.index[idx]
    for m in METRICS:
        row[f"rank_{m}"] = _i(ranks.loc[oct10_ts, m])
        zv = zs.loc[oct10_ts, m]
        row[f"z_{m}"] = round(float(zv), 2) if pd.notna(zv) else None
    row["rank_sum_abs_z"] = _i(comps.loc[oct10_ts, "rank_sum_abs_z"])
    row["rank_mahalanobis"] = _i(comps.loc[oct10_ts, "rank_mahalanobis"])
    row["sum_abs_z"] = round(float(comps.loc[oct10_ts, "sum_abs_z"]), 1)
    row["mahalanobis"] = round(float(comps.loc[oct10_ts, "mahalanobis"]), 2)

    if save_metrics:
        full = tbl.join(ranks, rsuffix="_rank").join(comps)
        full.to_csv(OUT_DIR / f"metrics_{spec.label}.csv")

    return row


def format_table_b1(grid: pd.DataFrame) -> str:
    """Compact markdown Table B1: one row per spec, October 10's rank on
    the four cascade-diagnostic metrics plus composites."""
    diag = ["sol_gap_pp", "basis_swing", "mark_minus_spot",
            "vol_surge_ratio"]
    lines = [
        "| Threshold | Window | De-overlap | N events | "
        "SOL gap | Basis swing | Mark undershoot | Vol surge | "
        "Rank Î£\\|z\\| | Rank Mahal. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in grid.iterrows():
        def _c(v):
            return str(int(v)) if pd.notna(v) else "â€”"
        if not r.get("oct10_found", False):
            cells = ["â€”"] * 6
        else:
            cells = [_c(r[f"rank_{m}"]) for m in diag] + \
                    [_c(r["rank_sum_abs_z"]), _c(r["rank_mahalanobis"])]
        base = " *(baseline)*" if r["is_baseline"] else ""
        lines.append(
            f"| {r['threshold_pct']:g}%{base} | {int(r['window_min'])} min "
            f"| {r['sep_hours']:g} h | {int(r['n_events'])} | "
            + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-all", action="store_true",
                    help="save per-event metric tables for every spec")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    data = load_all()

    rows = []
    combos = list(itertools.product(THRESHOLDS, WINDOWS, SEPARATIONS))
    for i, (th, w, sep) in enumerate(combos, 1):
        spec = EventSpec(th, w, sep)
        print(f"[{i:2d}/{len(combos)}] {spec.label} ...", flush=True)
        rows.append(run_spec(spec, data,
                             save_metrics=args.save_all or spec == BASELINE))

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "appendix_b_grid.csv", index=False)

    md = format_table_b1(grid)
    (OUT_DIR / "appendix_b_table.md").write_text(
        "# Table B1. Sensitivity of October 10's Distributional Position\n\n"
        "October 10, 2025's rank (1 = most extreme) on the four "
        "cascade-diagnostic metrics and both composite anomaly measures, "
        "under every event-selection specification.\n\n" + md + "\n", encoding="utf-8")

    print("\n" + md)

    base = grid[grid["is_baseline"]].iloc[0]
    print(f"\nBaseline spec: {int(base['n_events'])} events "
          f"(paper reports 62 â€” investigate any discrepancy before "
          f"updating the manuscript).")


if __name__ == "__main__":
    main()

