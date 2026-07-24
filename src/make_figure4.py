"""make_figure4.py — regenerate Figure 4 for the 58-event sample."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("output")
t = pd.read_csv(OUT / "metrics_dd3pct_w30min_sep6h.csv",
                parse_dates=["trough_ts"]).set_index("trough_ts")
oct10 = [x for x in t.index if str(x.date()) == "2025-10-10"][0]
others = t.drop(oct10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): histogram
ax1.hist(others["sol_gap_pp"], bins=25, color="0.6",
         label="57 other BTC drawdowns >=3% (2024-2026)")
ax1.hist([t.loc[oct10, "sol_gap_pp"]], bins=[10.8, 11.3], color="darkred",
         label="October 10, 2025")
ax1.annotate("October 10, 2025\nSOL gap = 11.03pp\nz-score = +7.39\nrank 1 of 58",
             xy=(11.03, 1), xytext=(6.2, 12),
             arrowprops=dict(arrowstyle="->", color="darkred"),
             bbox=dict(boxstyle="round", fc="white", ec="darkred"),
             fontsize=9, color="darkred")
ax1.set_xlabel("SOL Futures-Spot Drawdown Gap (percentage points)")
ax1.set_ylabel("Number of events")
ax1.set_title("(a) Empirical distribution of SOL futures-spot gap across 58 events",
              fontsize=10)
ax1.legend(fontsize=8)

# Panel (b): top-10 bars
top = t.nlargest(10, "sol_gap_pp").iloc[::-1]
labels = [ts.strftime("%b %d, %Y") for ts in top.index]
colors = ["darkred" if str(ts.date()) == "2025-10-10" else "0.6"
          for ts in top.index]
bars = ax2.barh(labels, top["sol_gap_pp"], color=colors)
for b, v in zip(bars, top["sol_gap_pp"]):
    ax2.text(v + 0.15, b.get_y() + b.get_height()/2, f"{v:.2f}",
             va="center", fontsize=8)
ax2.set_xlabel("SOL Futures-Spot Gap (pp)")
ax2.set_title("(b) Top 10 events ranked by SOL gap (12.6x larger than #2)",
              fontsize=10)
ax2.set_xlim(0, 12.5)

fig.suptitle("Figure 4. Distributional Evidence: October 10 as Statistical Outlier "
             "(n = 58 BTC drawdown events, 2024-2026)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "figure4.png", dpi=300, bbox_inches="tight")
print("saved", OUT / "figure4.png")
