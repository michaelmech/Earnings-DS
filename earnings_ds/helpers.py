import json
import pickle
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

def get_days_to_earnings(tickers, api_key, window_days=30):
    """
    Returns a Series showing days until (positive) or since (negative) earnings.
    """
    tz = ZoneInfo("America/New_York")
    today = datetime.now(tz).date()

    # Define a range to search (e.g., 30 days back to 30 days forward)
    start_date = (today - timedelta(days=window_days)).isoformat()
    end_date = (today + timedelta(days=window_days)).isoformat()

    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {"from": start_date, "to": end_date, "token": api_key}

    try:
        data = requests.get(url, params=params).json()
        rows = data.get("earningsCalendar", [])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.Series(dtype=int)

    # Map symbols to their dates
    # Note: If a ticker appears twice in the window, this keeps the most recent one found
    date_map = {
        r["symbol"]: datetime.strptime(r["date"], "%Y-%m-%d").date()
        for r in rows if r["symbol"] in tickers
    }

    # Calculate the difference in days
    results = {}
    for ticker in tickers:
        earnings_date = date_map.get(ticker)
        if earnings_date:
            delta = (earnings_date - today).days
            results[ticker] = delta
        else:
            results[ticker] = None # Or np.nan if you prefer

    return pd.Series(results, name="days_to_earnings")


def save_model(model, path):
    """
    Saves the VotingClassifier to the specified path.
    """
    try:
        # compress=3 is a good balance between speed and file size
        joblib.dump(model, path, compress=3)
        print(f"Model successfully saved to: {path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(path):
    """
    Loads the VotingClassifier from the specified path.
    """
    try:
        model = joblib.load(path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_feat_imps(X,y_new):

  model=LGBMClassifier(verbose=-1)
  model.fit(X,y_new)

  return pd.Series(model.feature_importances_,index=X.columns).sort_values()


def get_px(
    tickers,
    start="1998-01-01",
    end=None,
    use_adj_close=True,
    ffill_limit=3,
):
    """
    Returns a DataFrame:
      index  = trading days
      columns = tickers
      values  = prices
    """

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        ignore_tz=True,
    )

    # --- extract price panel ---
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(1).unique()
        px_field = "Adj Close" if (use_adj_close and "Adj Close" in fields) else "Close"
        px = raw.xs(px_field, axis=1, level=1)
    else:
        px_field = "Adj Close" if (use_adj_close and "Adj Close" in raw.columns) else "Close"
        px = raw[[px_field]]
        px.columns = tickers[:1]

    px = px.sort_index()
    px = px.ffill(limit=ffill_limit)

    return px


