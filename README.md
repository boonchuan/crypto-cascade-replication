# Replication: Anatomy of a Crypto Cascade

**v1.1** — definitional corrections relative to v1.0 (see Changelog).

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
#    plus the baseline metrics table.
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
| Tables 6 and 7 (manuscript-ready markdown) | `src/make_tables.py` |
| Figure 4 (distributional position) | `src/make_figure4.py` |

## Reproducibility notes

- The event-selection algorithm is deterministic; the only stochastic
  components (bootstrap, permutation) use a fixed seed (20251010).
- Exchange APIs occasionally backfill or correct historical candles.
  The cached parquet files pin the exact data vintage used; the archived
  dataset attached to release v1.0 of this repository contains the
  vintage used in the paper. The original submission's
  event count (62, October 2025 vintage, v1.0-era scan) differs from
  the reconstructed archived vintage; the original vintage was not
  preserved, so the archived vintage plus MANIFEST.sha256 constitute
  the reference. See Appendix B of the paper.
- Baseline specification (3% drawdown, 30-minute window, 6-hour
  de-overlap) reproduces the paper's reported event count against the
  archived vintage. `run_grid` prints a note if the event count differs,
  which would indicate a further API data revision.

## Citation

Lim, B.C. (2026) 'Anatomy of a Crypto Cascade: Minute-Level Evidence
from the October 2025 Crash', *International Journal of Blockchains and
Cryptocurrencies* [details on publication].

## License

Code: MIT. Data: sourced from Binance public APIs; see exchange terms.


## Changelog

### v1.1 (definitional corrections)
- **Event scan** now uses the directional trailing drawdown
  DD(t) = C(t)/max(C(t-w..t)) - 1, enforcing peak-before-trough
  ordering. (v1.0 used min/max within the window, which is
  direction-agnostic and could admit rally windows.)
- **Mark-to-spot undershoot** is now the minimum same-minute
  close-to-close gap, min(mark_close - spot_close), a simultaneous
  comparison. (v1.0 compared the mark candle low to the spot candle
  close, mixing timestamps within the minute.) The candle-low
  comparison is retained as a labelled auxiliary column.
- **"Intra-minute spread" renamed to intra-minute price range** — it is
  a candle high-low range, not a bid-ask spread.
- **Analysis window** standardized to exactly 30 one-minute
  observations, [t*-10, t*+19] inclusive.
- **Bootstrap** resamples the 57 comparison events with the focal event
  held out; exceedance reported as a fraction k/57 (not a permutation
  p-value; the smallest attainable one-sided rank-test p at n=58 is
  1/58 ~= 0.017).
- Added: four-event benchmark recomputation and Figures 1/3 generation
  (`src/four_events.py`), higher-order moment output, authoritative
  volume diagnostics, SHA-256 manifest (`src/make_manifest.py`).

## Exact environment
After running the pipeline, freeze the environment and commit it:

    pip freeze > requirements.lock.txt

The requirements.txt minimums are for portability; the lockfile records
the exact versions used for the published numbers.
