import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .cv import PurgedTimeSeriesSplit, cv_predict_proba_purged
from .dataset_generation import build_synthetic_earnings_test_dataset
from .helpers import save_model
from .simulations import (
    attach_returns_to_events,
    make_event_signal_matrices,
    vectorbt_trade_returns_gapaware,
)

def size_from_probs(
    close: pd.DataFrame,          # date x ticker
    p_cal: pd.Series,             # index=ticker, values in (0,1)
    asof: pd.Timestamp | None = None,
    horizon_days: int = 5,
    vol_lookback: int = 20,
    initial_capital: float = 100_000.0,

    # risk controls
    kelly_scale: float = 0.25,    # shrink Kelly (0.1–0.5 typical)
    max_gross: float = 1.0,       # max sum of weights (long-only)
    max_weight: float = 0.10,     # per-name cap
    min_prob: float = 0.55,       # only size if p >= this
    eps: float = 1e-6
):
    """
    Returns:
      weights: fraction of equity per ticker (sum <= max_gross)
      dollars: dollar allocation per ticker
      shares: share allocation per ticker at asof close
      diagnostics: df with p, z, sigma_h, raw_kelly
    """
    if asof is None:
        asof = close.index[-1]
    else:
        asof = pd.Timestamp(asof)

    # Align tickers present in both
    tickers = [t for t in p_cal.index if t in close.columns]
    p = p_cal.loc[tickers].astype(float)

    # price at asof
    px = close.loc[:asof, tickers].iloc[-1]

    # vol estimate (daily) -> horizon vol
    ret = close[tickers].pct_change()
    sigma_d = ret.loc[:asof].rolling(vol_lookback).std().iloc[-1]
    sigma_h = sigma_d * np.sqrt(horizon_days)

    # convert prob -> z-score (edge proxy)
    p_clip = p.clip(eps, 1 - eps)
    z = pd.Series(norm.ppf(p_clip), index=tickers)

    # long-only: ignore p < 0.5; enforce min_prob gate
    z = z.where(p >= min_prob, other=0.0)

    # Kelly-like fraction (normal approx): f ~ mu / var = (sigma_h*z) / sigma_h^2 = z / sigma_h
    raw = z / (sigma_h.replace(0.0, np.nan))

    # shrink + clean
    raw = (kelly_scale * raw).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = raw.clip(lower=0.0)  # long-only

    # per-name cap
    w = raw.clip(upper=max_weight)

    # gross cap (sum weights <= max_gross)
    gross = w.sum()
    if gross > max_gross and gross > 0:
        w = w * (max_gross / gross)

    dollars = w * initial_capital
    shares = (dollars / px).fillna(0.0)

    diagnostics = pd.DataFrame({
        "p_cal": p,
        "z": z,
        "sigma_d": sigma_d,
        "sigma_h": sigma_h,
        "raw_kelly": raw,
        "weight": w,
        "dollars": dollars,
        "price": px,
        "shares": shares
    }).sort_values("weight", ascending=False)

    return w, dollars, shares, diagnostics


def build_meta_dataset(X, p_primary, trade_ret, min_abs_ret=0.0):
    """
    X: base features (aligned to events)
    p_primary: primary OOF proba (aligned)
    trade_ret: realized trade returns aligned to events (positive means profit in predicted direction)
    """
    df = X.copy()
    df['p_primary'] = p_primary

    # meta label
    meta_y = (trade_ret > 0).astype(int)

    # Optional: drop tiny outcomes (noise)
    if min_abs_ret > 0:
        keep = trade_ret.abs() >= min_abs_ret
        df = df.loc[keep]
        meta_y = meta_y.loc[keep]
        trade_ret = trade_ret.loc[keep]

    return df, meta_y


def run_primary_plus_meta(
    X, y, data_with_event_day, px_close,px_open,px_high,px_low, tickers,
    horizon=5,
    primary_model=None,
    meta_model=None,
    side_threshold=0.5,
    meta_min_abs_ret=0.0,
    gap=121,
    anchors=None,
    score_mode=False,
    sl=0.032,
    tp=0.032,
    save_meta_model_path=None,
    long_only=True,
):
    if primary_model is None:
        primary_model = LGBMClassifier(verbose=-1)
    if meta_model is None:
        meta_model = make_pipeline(SimpleImputer(fill_value=-999),LogisticRegression())

    X=X.replace({np.inf: np.nan,-np.inf: np.nan})
    X=X.copy().fillna(-999)

    px_close = px_close.copy()
    px_close.index = pd.to_datetime(px_close.index).normalize()

    if score_mode:
      print('Number of mismatched tickers:',set(ds['ticker'])-set(px_close))

    # --- Purged CV object (you already do this) ---
    dates = pd.Series(X.index.get_level_values('earnings_ts'))
    cv = PurgedTimeSeriesSplit(dates=dates, gap=gap)

    # --- 1) OOF primary proba ---
    p_oof = cv_predict_proba_purged(primary_model, X, y, cv)
    p_oof = p_oof.loc[X.index]

    # --- build event-day frame aligned to X ---
    d = data_with_event_day.copy()

    d['earnings_ts'] = pd.to_datetime(d['earnings_ts'])
    d['event_day'] = pd.to_datetime(d['event_day'])

    d = d.set_index(['ticker', 'earnings_ts']).sort_index()

    # align to X
    d = d.reindex(X.index)

    print("aligned d.index.names:", d.index.names)
    print("aligned event_day non-null:", d['event_day'].notna().sum(), "/", len(d))
    print("aligned p_oof non-null:", p_oof.notna().sum(), "/", len(p_oof))

    # optional: show a few missing rows
    missing = d['event_day'].isna()
    print("missing event_day rows:", missing.sum())
    if missing.any():
        print("sample missing keys:", d.index[missing][:10].to_list())

    idx_df = d[['event_day']].copy()

    # --- 2) vectorbt realized returns for each OOF “trade” ---
    # --- build event-day frame aligned to X ---
    d = data_with_event_day.copy()
    d['earnings_ts'] = pd.to_datetime(d['earnings_ts'])
    d['event_day'] = pd.to_datetime(d['event_day'])

    d = d.set_index(['ticker', 'earnings_ts']).sort_index()
    d = d.reindex(X.index)

    idx_df = d[['event_day']].copy()
    print("PASSING idx_df.index.names:", idx_df.index.names)  # should be ['ticker','earnings_ts']

    tmp_events, el, xl, es, xs = make_event_signal_matrices(
        index_df=idx_df,
        px_close=px_close,
        p_primary=p_oof,
        horizon=horizon,
        side_threshold=side_threshold,
        debug=False,
    )

    if long_only:
      es[:] = False
      xs[:] = False

    trades = vectorbt_trade_returns_gapaware(
              open_df=px_open,
              high_df=px_high,
              low_df=px_low,
              close_df=px_close,
              entries_long=el, exits_long=xl,
              entries_short=es, exits_short=xs,
              tp=tp, sl=sl
          )

    print(trades.shape,'trades shape')
    print(len(tmp_events),'events lengths')

    events_with_ret = attach_returns_to_events(tmp_events, trades, px_close)

    print(events_with_ret.dropna(subset=['trade_ret']).shape,'shape after dropping nans')

    # ================= CONSISTENCY ASSERTIONS =================

    # 1) Entry day must match labeler rule
    def _labeler_event_day(earnings_ts, trade_days):
        ts = earnings_ts.to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(trade_days, ts, side="right") - 1

        out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
        ok = pos >= 0
        out[ok] = trade_days[pos[ok]]
        return pd.to_datetime(out).normalize()


    trade_days = px_close.index.values.astype("datetime64[ns]")

    dcheck = tmp_events.reset_index().copy()
    dcheck["earnings_ts"] = pd.to_datetime(dcheck["earnings_ts"])

    dcheck["event_day_labeler"] = _labeler_event_day(
        dcheck["earnings_ts"],
        trade_days
    )

    dcheck["event_day_used"] = pd.to_datetime(dcheck["event_day"]).dt.normalize()

    bad_day = dcheck["event_day_labeler"] != dcheck["event_day_used"]

    assert not bad_day.any(), (
        "Event day mismatch with labeler mapping:\n"
        + dcheck.loc[bad_day,
            ["ticker", "earnings_ts", "event_day_used", "event_day_labeler"]
          ].head(10).to_string()
    )


    # 2) No short trades allowed in meta (long-only invariant)
    if long_only:
        assert (es.values == 0).all(), "Short entries present while long_only=True"
        assert (xs.values == 0).all(), "Short exits present while long_only=True"


    # 3) Meta label must agree with horizon sign when no stop triggered
    #    (soft check: allow tiny tolerance)
    merged = events_with_ret.copy()

    merged["meta_y"] = (merged["trade_ret"] > 0).astype(int)

    # Compare with your original y if available
    if "y_new" in locals():
        y_aligned = y_new.reindex(merged.index)

        agree = (merged["meta_y"] == y_aligned) | y_aligned.isna()

        disagree_rate = 1 - agree.mean()

        assert disagree_rate < 0.01, (   # 1% tolerance
            f"Meta/primary label disagreement too high: {disagree_rate:.2%}"
        )

    # ==========================================================


    trade_ret = events_with_ret['trade_ret'].dropna()
    print(trade_ret.shape,'trade ret shape')

    common_idx = X.index.intersection(trade_ret.index)

    print(len(common_idx),'len common')

    trade_ret = trade_ret.loc[common_idx]
    X_meta_base = X.loc[common_idx]
    p_oof_ok = p_oof.loc[common_idx]
    y_ok = y.loc[common_idx]

    print(y_ok.shape,'initial y')

    long_only_mask = p_oof_ok >= side_threshold
    trade_ret = trade_ret.loc[long_only_mask]
    X_meta_base = X_meta_base.loc[long_only_mask]
    p_oof_ok = p_oof_ok.loc[long_only_mask]
    y_ok = y_ok.loc[long_only_mask]

    print(y_ok.shape,'y after long filter')

    # --- 3) Meta dataset (features + p_primary) -> meta label ---
    X_meta, y_meta = build_meta_dataset(
        X=X_meta_base,
        p_primary=p_oof_ok,
        trade_ret=trade_ret,
        min_abs_ret=meta_min_abs_ret
    )

    if score_mode:
      return X_meta,y_meta

    # --- 4) Fit models on full calibration ---
    primary_fit = primary_model
    primary_fit.fit(X, y)

    meta_fit = meta_model
    meta_fit.fit(X_meta, y_meta)

    if save_meta_model_path:
        pd.concat([X_meta,y_meta],axis=1).to_csv(f'meta_dataset_{primary_model.__str__()[:16]}.csv')
        save_model(meta_fit,save_meta_model_path)
        save_model(primary_fit,save_meta_model_path.replace('meta','primary'))

    # --- 5) Build synthetic “today earnings” test set (you already have this function) ---
    test_ds = build_synthetic_earnings_test_dataset(
        tickers=tickers,
        asof=None,
        start="1998-01-01",
        anchor=anchors if anchors is not None else ["SPY"],
        min_hist_days=200,
    )

    # Align columns
    X_live = test_ds[X.columns].copy()

    # --- 6) Live primary proba ---
    p_primary_live = pd.Series(
        primary_fit.predict_proba(X_live)[:, 1],
        index=test_ds['ticker'].values
    )

    # --- 7) Live meta proba (use same feature recipe: X + p_primary) ---
    X_meta_live = X_live.copy()
    X_meta_live['p_primary'] = p_primary_live.values

    p_meta_live = pd.Series(
        meta_fit.predict_proba(X_meta_live[X_meta.columns])[:, 1],
        index=test_ds['ticker'].values
    )

    # --- 8) One simple position sizing rule ---
    # signed conviction in [-1, 1] times probability of profit:
    signed = (p_primary_live * 2.0 - 1.0)   # long if >0, short if <0
    size = signed * p_meta_live            # scale by meta prob

    out = pd.DataFrame({
        'p_primary': p_primary_live,
        'p_meta': p_meta_live,
        'size': size
    }).sort_values('size', ascending=False)

    return out, {
        'p_oof': p_oof,
        'trade_ret': trade_ret,
        'X_meta': X_meta,
        'y_meta': y_meta
    }

