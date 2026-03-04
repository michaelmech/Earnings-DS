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
