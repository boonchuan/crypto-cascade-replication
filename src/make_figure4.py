"""make_figure4.py - Figure 4: distributional position of October 10, 2025.

Panel (a): histogram of the SOL futures-spot drawdown gap across the event
sample, with October 10 highlighted.
Panel (b): the ten largest gaps, ranked.

Every sample-size label, z-score, rank and ratio is derived from the input
data. Nothing is hardcoded, so the figure remains correct for any event
sample the pipeline produces.

Run from the repository root:
    python src/make_figure4.py

Input:  output/metrics_dd3pct_w30min_sep6h.csv
Output: output/figure4.png  (3570 x 1338 px at 300 dpi)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "output/metrics_dd3pct_w30min_sep6h.csv"
OUT = "output/figure4.png"
OCT = "2025-10-10"
DARK = "#8B0000"
GREY = "#808080"

# bbox='tight' silently changes the saved pixel dimensions, which breaks
# exact-size replacement of the image inside the manuscript.
matplotlib.rcParams["savefig.bbox"] = None


def main() -> None:
    m = pd.read_csv(SRC)
    n = len(m)

    hits = m.index[m.trough_ts.str.startswith(OCT)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {OCT} event, found {len(hits)}")
    i = hits[0]

    gap = float(m.sol_gap_pp.loc[i])
    z = (gap - m.sol_gap_pp.mean()) / m.sol_gap_pp.std(ddof=1)
    rank = int(m.sol_gap_pp.rank(ascending=False, method="min").loc[i])

    fig, ax = plt.subplots(1, 2, figsize=(11.9, 4.46), dpi=300)

    # --- panel (a): empirical distribution ---
    bins = np.arange(-0.5, np.ceil(gap) + 1.1, 0.1)
    ax[0].hist(
        m.drop(index=i).sol_gap_pp,
        bins=bins,
        color=GREY,
        label=f"{n - 1} other BTC drawdowns >=3% (2024-2026)",
    )
    ax[0].hist([gap], bins=bins, color=DARK, label="October 10, 2025")
    ax[0].set_title(
        f"(a) Empirical distribution of SOL futures-spot gap across {n} events",
        fontsize=11,
    )
    ax[0].set_xlabel(
        "SOL Futures-Spot Drawdown Gap (percentage points)", fontsize=10
    )
    ax[0].set_ylabel("Number of events", fontsize=10)
    ax[0].legend(fontsize=8.5, loc="upper right")

    ymax = ax[0].get_ylim()[1]
    ax[0].annotate(
        f"October 10, 2025\n"
        f"SOL gap = {gap:.2f}pp\n"
        f"z-score = {z:+.2f}\n"
        f"rank {rank} of {n}",
        xy=(gap, 1.0),
        xytext=(gap * 0.39, ymax * 0.55),
        fontsize=10,
        color=DARK,
        bbox=dict(boxstyle="round", fc="white", ec=DARK),
        arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5),
    )

    # --- panel (b): top ten, ascending up the axis ---
    top = m.nlargest(10, "sol_gap_pp").iloc[::-1]
    ax[1].barh(
        range(len(top)),
        top.sol_gap_pp,
        color=[DARK if k == i else GREY for k in top.index],
    )
    ax[1].set_yticks(range(len(top)))
    ax[1].set_yticklabels(
        [pd.to_datetime(t).strftime("%b %d, %Y") for t in top.trough_ts],
        fontsize=10,
    )
    for y, v in enumerate(top.sol_gap_pp):
        ax[1].text(v + 0.15, y, f"{v:.2f}", va="center", fontsize=10)

    runner_up = float(top.sol_gap_pp.iloc[-2])
    ratio = gap / runner_up
    ax[1].set_title(
        f"(b) Top 10 events ranked by SOL gap ({ratio:.1f}x larger than #2)",
        fontsize=11,
    )
    ax[1].set_xlabel("SOL Futures-Spot Gap (pp)", fontsize=10)
    ax[1].set_xlim(0, gap * 1.13)

    fig.suptitle(
        f"Figure 4. Distributional Evidence: October 10 as Statistical Outlier "
        f"(n = {n} BTC drawdown events, 2024-2026)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=300)

    print(
        f"wrote {OUT}: n={n}, gap={gap:.2f}pp, z={z:+.2f}, "
        f"rank={rank} of {n}, ratio={ratio:.1f}x"
    )


if __name__ == "__main__":
    main()
