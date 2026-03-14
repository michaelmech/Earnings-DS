# Earnings-DS

Purpose of the repo is house code for a data science pipeline aimed towards classifying earnings performances of tickers. The general strategy is to buy a ticker right before their earnings, whether that be after market close, or before market open of next day.

## Refactor: notebook logic split into Python modules

Code from `Copy_of_Earnings.ipynb` has been organized into a package under `earnings_ds/`:

- `earnings_ds/simulations.py` – vectorbt backtest/signal simulation helpers.
- `earnings_ds/meta_labeling.py` – meta-labeling dataset and sizing functions.
- `earnings_ds/dataset_generation.py` – feature engineering and dataset builders.
- `earnings_ds/cv.py` – purged time-series CV and related validation helpers.
- `earnings_ds/execution_alpaca.py` – Alpaca order execution/rebalance utilities.
- `earnings_ds/helpers.py` – persistence, feature importance, and API convenience helpers.
- `earnings_ds/pipeline.py` – inference pipeline entrypoint(s).

You can import everything from the package root via:

```python
from earnings_ds import *
```

## Guidance: increase trade count without degrading Sharpe

When meta-labeling is tuned only for very high average precision (AP), recall usually collapses and too many valid opportunities get filtered out. A better objective is to optimize **portfolio-level utility with explicit coverage constraints**.

Recommended approach:

1. **Constrain opportunity first, then optimize quality**
   - Add minimum participation constraints in validation, such as:
     - trades per month,
     - percent of eligible events traded,
     - minimum recall in top liquidity buckets.
   - Only compare model candidates that pass these floor constraints.

2. **Tune thresholds on an efficient frontier, not AP alone**
   - Sweep decision thresholds and build a frontier of:
     - AP / precision,
     - recall / participation,
     - out-of-sample Sharpe,
     - turnover and capacity proxies.
   - Pick a threshold near the "knee" where Sharpe is stable but trade count rises materially.

3. **Replace hard gates with soft penalties where possible**
   - If illiquidity filtering is too strict, convert binary rejection into position-size scaling (smaller risk on worse liquidity) so marginal but positive-EV trades are not fully discarded.
   - Keep hard rejects only for true infeasibility (e.g., spreads/slippage beyond expected edge).

4. **Calibrate probabilities and size by expected edge**
   - Calibrate meta-model probabilities (isotonic/Platt) so score thresholds are meaningful.
   - Use confidence-weighted sizing: high-score trades get larger allocations, medium-score trades still trade small.
   - This generally increases count while preserving Sharpe through risk budgeting.

5. **Use cost-aware training and evaluation**
   - Incorporate slippage/fees/borrow assumptions directly in labels or sample weights.
   - Evaluate precision/recall conditional on implementability (liquidity, spread regime) rather than raw classification alone.

6. **Monitor segment-level recall to avoid hidden choke points**
   - Break out recall and trade acceptance by:
     - market-cap decile,
     - ADV/liquidity bucket,
     - event type (BMO/AMC),
     - volatility regime.
   - If one gate drives most rejections, retune that gate before retuning the classifier.

Practical rule of thumb: target the highest Sharpe model within a participation band (for example, at least 60-80% of your desired monthly trade count) instead of the highest AP model globally.
