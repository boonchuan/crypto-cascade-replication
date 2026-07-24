# Replication: Anatomy of a Crypto Cascade

Replication package for **"Anatomy of a Crypto Cascade: Minute-Level
Evidence from the October 2025 Crash"** (Boon Chuan Lim, Independent
Researcher, Singapore).

All data derive from free, unauthenticated Binance public REST APIs.
The pipeline implements the deterministic event-construction algorithm
specified in Appendix A of the paper, the Appendix B sensitivity grid,
and the VAR / bootstrap / correlation analyses of the revised manuscript.

## Quick start

```bash
pip install -r requirements.txt

# 1. Download and cache 1-minute klines (Jan 2024 - Apr 2026, ~5 series).
#    One-time; ~1.2M minutes per series. Cached to data/*.parquet.
python -m src.fetch_data

# 2. Run the full Appendix B sensitivity grid (36 specifications).
#    Produces output/appendix_b_grid.csv and output/appendix_b_table.md,
#    plus the baseline 62-event metrics table.
python -m src.run_grid

# 3. Bootstrap CIs, trivariate VAR with Granger tests and IRFs,
#    rolling correlations.
python -m src.inference
```

## Pipeline map (paper section -> code)

| Paper | Code |
|---|---|
| Appendix A Step 1 (data acquisition) | `src/fetch_data.py` |
| Appendix A Steps 2–4 (event construction) | `src/events.py` |
| Appendix A Steps 5–6 (metrics, z-scores, composites) | `src/metrics.py` |
| Appendix B (sensitivity grid) | `src/run_grid.py` |
| Section 4.1 revised (trivariate VAR) | `src/inference.py::trivariate_var` |
| Bootstrap CIs / rank stability | `src/inference.py::bootstrap_rank_cis` |
| Rolling correlations | `src/inference.py::rolling_correlations` |

## Reproducibility notes

- The event-selection algorithm is deterministic; the only stochastic
  components (bootstrap, permutation) use a fixed seed (20251010).
- Exchange APIs occasionally backfill or correct historical candles.
  The cached parquet files pin the exact data vintage used; the archived
  dataset deposit (Zenodo/OSF) contains the vintage used in the paper.
- Baseline specification (3% drawdown, 30-minute window, 6-hour
  de-overlap) should reproduce the paper's 62-event sample. `run_grid`
  prints a warning if the event count differs, which would indicate an
  API data revision relative to the archived vintage.

## Citation

Lim, B.C. (2026) 'Anatomy of a Crypto Cascade: Minute-Level Evidence
from the October 2025 Crash', *International Journal of Blockchains and
Cryptocurrencies* [details on publication].

## License

Code: MIT. Data: sourced from Binance public APIs; see exchange terms.
