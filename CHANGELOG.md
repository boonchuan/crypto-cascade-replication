# Changelog

## v1.1 — 27 July 2026

Corrects three definitional errors in the v1.0 event-construction and
metric code. All results in the IJBC-334247 revision are computed under
these definitions.

### Corrections

1. **Directional trailing drawdown.** The v1.0 scan measured drawdown
   non-directionally within the window. v1.1 uses a directional
   peak-to-trough trailing measure. Event sample: 58 -> 45 under the
   July 2026 data vintage.

2. **Simultaneous mark-to-spot gap.** v1.0 compared the mark candle low
   with the spot candle close, mixing observations from different
   instants within the minute. v1.1 uses the same-minute close-to-close
   gap. October 10: -$2,507 -> -$497 at 21:17. The candle-low comparison
   is retained as a labelled auxiliary series (-$122).

3. **Intra-minute range mislabelled as spread.** The v1.0 "maximum
   intra-minute spread" is a candle high-low range, not a bid-ask
   spread. Renamed throughout; associated liquidity claims moderated.

The analysis window is standardised to exactly 30 one-minute
observations.

### Deposited with this release

- `data/` — archived 1-minute Binance kline vintage (5 parquet files)
- `output/` — processed event dataset, sensitivity grid, figures
- `MANIFEST.sha256` — SHA-256 of every archived file
- `environment-full.txt`, `requirements.txt` — dependency pins

## v1.0

Initial deposit accompanying the original submission. Source only;
superseded by v1.1.
