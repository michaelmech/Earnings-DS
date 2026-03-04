import numpy as np
import pandas as pd

from .dataset_generation import build_synthetic_earnings_test_dataset

def infer_primary_plus_meta(
    primary_fit,
    meta_fit,
    tickers,
    feature_cols,
    meta_feature_cols=None,
    anchors=None,
    asof=None,
    start="1998-01-01",
    min_hist_days=200,
):
    """
    Pure inference:
      1) build synthetic test dataset
      2) compute primary proba
      3) compute meta proba (using same feature recipe)
      4) output sizing signal

    Returns:
      out_df, test_ds, X_live, X_meta_live
    """

    if meta_feature_cols is None:
        meta_feature_cols = list(feature_cols) + ["p_primary"]

    # --- 1) Build synthetic test set ---
    test_ds = build_synthetic_earnings_test_dataset(
        tickers=tickers,
        asof=asof,
        start=start,
        anchor=anchors if anchors is not None else ["SPY"],
        min_hist_days=min_hist_days,
    )

    # --- 2) Align base features exactly ---
    X_live = test_ds.reindex(columns=feature_cols).copy()

    # handle inf/nan similar to your training path
    X_live = X_live.replace([np.inf, -np.inf], np.nan).fillna(-999)

    # --- 3) Primary proba ---
    p_primary_live = pd.Series(
        primary_fit.predict_proba(X_live)[:, 1],
        index=test_ds["ticker"].values
    )

    # --- 4) Meta proba ---
    X_meta_live = X_live.copy()
    X_meta_live["p_primary"] = p_primary_live.values

    X_meta_live = X_meta_live.reindex(columns=meta_feature_cols).copy()
    X_meta_live = X_meta_live.replace([np.inf, -np.inf], np.nan).fillna(-999)

    p_meta_live = pd.Series(
        meta_fit.predict_proba(X_meta_live)[:, 1],
        index=test_ds["ticker"].values
    )

    # --- 5) Simple sizing rule ---
    signed = (p_primary_live * 2.0 - 1.0)   # [-1, 1]
    size = signed * p_meta_live

    out = (
        pd.DataFrame({"p_primary": p_primary_live, "p_meta": p_meta_live, "size": size})
        .sort_values("size", ascending=False)
    )

    return out, test_ds, X_live, X_meta_live


