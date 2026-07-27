"""four_events.py — recompute the four benchmark events (Tables 4 and 5)
under the v1.1 definitions, and regenerate Figures 1 and 3.

The four events are located by finding the BTC spot trough within a
search window around each announced date, then applying the standard
relocated analysis window and metric set.

Outputs:
  output/four_events.csv    all metrics for the four events
  output/figure1.png        crash timeline with corrected annotation
  output/figure3.png        three-panel cross-event comparison

Usage: python -m src.four_events
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .events import Event, EventSpec
from .fetch_data import cache_path
from .metrics import compute_event_metrics

OUT = Path(__file__).resolve().parents[1] / "output"
SPEC = EventSpec(3.0, 30, 6.0)

EVENTS = {
    "Yen Carry Trade (Aug 5, 2024)": ("2024-08-04 12:00", "2024-08-06 12:00"),
    "Trade War (Feb 3, 2025)":       ("2025-02-02 12:00", "2025-02-04 12:00"),
    "Liberation Day (Apr 7, 2025)":  ("2025-04-06 12:00", "2025-04-08 12:00"),
    "October Cascade (Oct 10, 2025)": ("2025-10-09 12:00", "2025-10-11 12:00"),
}


def load_all():
    names = {"btc_spot": ("BTCUSDT", "spot"), "btc_perp": ("BTCUSDT", "perp"),
             "btc_mark": ("BTCUSDT", "mark"), "sol_spot": ("SOLUSDT", "spot"),
             "sol_perp": ("SOLUSDT", "perp")}
    return {k: pd.read_parquet(cache_path(*v)).set_index("ts").sort_index()
            for k, v in names.items()}


def main():
    OUT.mkdir(exist_ok=True)
    data = load_all()
    btc = data["btc_spot"]

    rows = []
    for label, (a, b) in EVENTS.items():
        seg = btc.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC"),
                      "close"]
        if seg.empty:
            print(f"WARNING: no cached data for {label}; skipped")
            continue
        trough = seg.idxmin()
        ev = Event(trough_ts=trough, scan_end_ts=trough, drawdown_pct=0.0)
        m = compute_event_metrics(ev, SPEC, data)
        # 48h directional peak-to-trough around the event
        peak = seg.cummax()
        m["btc_dd_48h"] = float(((seg / peak) - 1.0).min() * 100.0)
        m["event"] = label
        rows.append(m)

    df = pd.DataFrame(rows).set_index("event")
    cols = ["trough_ts", "btc_dd_48h", "btc_spot_dd", "btc_fut_dd",
            "sol_gap_pp", "basis_swing", "mark_minus_spot",
            "mark_minus_spot_minute", "aux_marklow_minus_spotlow",
            "vol_surge_ratio", "max_intramin_spread"]
    df[cols].to_csv(OUT / "four_events.csv")
    print(df[cols].to_string())

    # ---------------- Figure 1: crash timeline ----------------
    w0 = pd.Timestamp("2025-10-10 21:08", tz="UTC")
    w1 = pd.Timestamp("2025-10-10 21:34", tz="UTC")
    spot = data["btc_spot"].loc[w0:w1, "close"]
    fut = data["btc_perp"].loc[w0:w1, "close"]
    mark = data["btc_mark"].loc[w0:w1, "close"]
    gap = (mark - spot)
    gmin_ts, gmin = gap.idxmin(), gap.min()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spot.index, spot, "o-", color="#1f4e79", label="Spot close (Binance)")
    ax.plot(fut.index, fut, "s-", color="#c00000", label="Perpetual futures close")
    ax.plot(mark.index, mark, "^-", color="#2e7d32", label="Mark price close")
    ax.axvspan(pd.Timestamp("2025-10-10 21:17", tz="UTC"),
               pd.Timestamp("2025-10-10 21:21", tz="UTC"),
               color="red", alpha=0.08)
    ax.annotate(f"Mark-spot close gap\nmin {gmin:,.0f} USD at "
                f"{gmin_ts.strftime('%H:%M')} UTC",
                xy=(gmin_ts, mark.loc[gmin_ts]),
                xytext=(0.62, 0.25), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#2e7d32"),
                fontsize=9, color="#2e7d32")
    ax.set_title("Figure 1. BTC Price Cascade: Spot, Futures, and Mark Price "
                 "(October 10, 2025, 21:08-21:34 UTC)", fontsize=11)
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (UTC), October 10, 2025")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figure1.png", dpi=300, bbox_inches="tight")
    print("saved figure1.png; min close-close mark-spot gap:",
          round(gmin, 0), "at", gmin_ts)

    # ---------------- Figure 3: cross-event panels ----------------
    order = ["October Cascade (Oct 10, 2025)", "Yen Carry Trade (Aug 5, 2024)",
             "Trade War (Feb 3, 2025)", "Liberation Day (Apr 7, 2025)"]
    short = ["Oct 10, 2025", "Aug 5, 2024", "Feb 3, 2025", "Apr 7, 2025"]
    colors = ["darkred", "0.6", "0.6", "0.6"]
    panels = [("sol_gap_pp", "(a) SOL futures-spot gap (pp)", "{:.2f}"),
              ("basis_swing", "(b) BTC basis swing (USD)", "{:,.0f}"),
              ("mark_minus_spot", "(c) Min mark-spot close gap (USD)",
               "{:,.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = [df.loc[e, col] for e in order]
        vals_plot = [abs(v) if col == "mark_minus_spot" else v for v in vals]
        bars = ax.bar(short, vals_plot, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(abs(v) if col == "mark_minus_spot" else v),
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Figure 3. Cross-Event Microstructure Comparison "
                 "(v1.1 definitions)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "figure3.png", dpi=300, bbox_inches="tight")
    print("saved figure3.png")


if __name__ == "__main__":
    main()
