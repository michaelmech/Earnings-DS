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
    simulate_event_returns_from_proba,
)


def size_from_run_primary_out(
    out: pd.DataFrame,
    *,
    weighting_scheme: str = "equal",
    score_col: str = "size",
    max_gross: float = 1.0,
) -> pd.Series:
    """Convert `run_primary_plus_meta` output into portfolio weights.

    weighting_scheme:
      - ``equal``: equal gross-weight allocation per tradable name
      - ``proba``: proportional weighting by ``score_col`` magnitude
      - ``score``: backward-compatible alias of ``proba``
    """
    if weighting_scheme == "score":
        weighting_scheme = "proba"

    if weighting_scheme not in ("equal", "proba"):
        raise ValueError("weighting_scheme must be 'equal', 'proba', or 'score'")

    if score_col not in out.columns:
        raise ValueError(f"score_col '{score_col}' not found in out")

    tradable = out.get("is_tradable", pd.Series(True, index=out.index)).astype(bool)
    raw = out.loc[tradable, score_col].astype(float)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if raw.empty:
        return pd.Series(0.0, index=out.index, name="weight")

    if weighting_scheme == "equal":
        direction = np.sign(raw).replace(0.0, 1.0)
        per_name = float(max_gross) / float(len(raw))
        sized = direction * per_name
    else:
        gross = raw.abs().sum()
        sized = raw * (float(max_gross) / gross) if gross > 0 else raw * 0.0

    weights = pd.Series(0.0, index=out.index, name="weight")
    weights.loc[sized.index] = sized
    return weights

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
    px_volume=None,
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
    use_smart_slippage=True,
    smart_slippage_kwargs=None,
    use_illiquidity_gate=False,
    illiquidity_spread_df=None,
    illiquidity_spread_kwargs=None,
    weighting_scheme="equal",
    gate_debug=False,
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
      print('Number of mismatched tickers:',set(data_with_event_day['ticker'])-set(px_close))

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

    (
        events_with_ret,
        tmp_events,
        trades,
        el,
        xl,
        es,
        xs,
        pf,
    ) = simulate_event_returns_from_proba(
        index_df=idx_df,
        p_primary=p_oof,
        px_open=px_open,
        px_high=px_high,
        px_low=px_low,
        px_close=px_close,
        px_volume=px_volume,
        horizon=horizon,
        side_threshold=side_threshold,
        tp=tp,
        sl=sl,
        long_only=long_only,
        use_smart_slippage=use_smart_slippage,
        smart_slippage_kwargs=smart_slippage_kwargs,
        use_illiquidity_gate=use_illiquidity_gate,
        illiquidity_spread_df=illiquidity_spread_df,
        illiquidity_spread_kwargs=illiquidity_spread_kwargs,
        weighting=weighting_scheme,
        debug=gate_debug,
        return_pf=True,
    )

    print(trades.shape, 'trades shape')
    print(pf, 'portfolio object')
    print(len(tmp_events), 'events lengths')

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
    test_ds = test_ds.drop_duplicates(subset=['ticker'], keep='last').copy()

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

    # --- 7b) Apply live-time event gating (same path used by simulations) ---
    idx_live = (
        test_ds[['ticker', 'earnings_ts', 'event_day']]
        .assign(
            earnings_ts=lambda df: pd.to_datetime(df['earnings_ts']),
            event_day=lambda df: pd.to_datetime(df['event_day']),
        )
        # Defensive: synthetic builders can occasionally emit duplicate event keys.
        # keep one row per (ticker, earnings_ts) so downstream MultiIndex lookups are scalar.
        .drop_duplicates(subset=['ticker', 'earnings_ts'], keep='last')
        .set_index(['ticker', 'earnings_ts'])[['event_day']]
        .sort_index()
    )

    # Map per-ticker live probabilities to event-indexed series.
    p_by_ticker = p_primary_live[~p_primary_live.index.duplicated(keep='last')]
    p_primary_live_event = pd.Series(
        p_by_ticker.reindex(idx_live.index.get_level_values('ticker')).to_numpy(),
        index=idx_live.index,
    )

    spread_df = None
    illiquidity_threshold_by_event = None
    if use_illiquidity_gate:
        spread_df = illiquidity_spread_df
        if spread_df is None:
            from .simulations import calculate_agk_spread_proxy

            spread_df = calculate_agk_spread_proxy(
                open_df=px_open,
                high_df=px_high,
                low_df=px_low,
                close_df=px_close,
                **(illiquidity_spread_kwargs or {}),
            )
        spread_df = spread_df.reindex_like(px_close)
        tp_threshold_df = (float(tp) * px_close).abs()
        illiquidity_threshold_by_event = (
            tp_threshold_df.stack()
            .rename_axis(['event_day', 'ticker'])
            .rename('spread_cap')
            .reorder_levels(['ticker', 'event_day'])
        )

    from .simulations import make_event_signal_matrices

    _, el_live, _, es_live, _ = make_event_signal_matrices(
        index_df=idx_live,
        px_close=px_close,
        p_primary=p_primary_live_event,
        horizon=horizon,
        side_threshold=side_threshold,
        illiquidity_spread_df=spread_df,
        illiquidity_threshold_by_event=illiquidity_threshold_by_event,
        debug=gate_debug,
    )

    if long_only:
        es_live[:] = False

    live_entries = (el_live | es_live)
    live_entry_keys = (
        live_entries.stack()
        .loc[lambda s: s]
        .index.set_names(['event_day', 'ticker'])
        .reorder_levels(['ticker', 'event_day'])
    )
    live_entry_key_set = set(live_entry_keys.to_list())

    tradable_mask = pd.Series(
        [
            (tkr, pd.to_datetime(ev_day).normalize()) in live_entry_key_set
            for tkr, ev_day in zip(test_ds['ticker'].values, test_ds['event_day'].values)
        ],
        index=test_ds['ticker'].values,
        dtype=bool,
    )

    # --- 8) Position sizing ---
    # Base score is signed conviction in [-1, 1] times probability of profit.
    signed = (p_primary_live * 2.0 - 1.0)   # long if >0, short if <0
    size = signed * p_meta_live            # scale by meta prob

    out = pd.DataFrame({
        'p_primary': p_primary_live,
        'p_meta': p_meta_live,
        'size': size
    }).sort_values('size', ascending=False)

    out['is_tradable'] = tradable_mask.reindex(out.index).fillna(False)
    if gate_debug:
        blocked = int((~out['is_tradable']).sum())
        allowed = int(out['is_tradable'].sum())
        print(f"[run_primary_plus_meta gate debug] allowed={allowed} blocked={blocked}")
    out = out.loc[out['is_tradable']].copy()
    out['weight'] = size_from_run_primary_out(out, weighting_scheme=weighting_scheme)

    return out, {
        'p_oof': p_oof,
        'trade_ret': trade_ret,
        'X_meta': X_meta,
        'y_meta': y_meta
    }



def derive_meta_test_predictions(
    X_train,
    y_train,
    train_data_with_event_day,
    X_test,
    test_data_with_event_day,
    px_close,
    px_open,
    px_high,
    px_low,
    px_volume=None,
    horizon=5,
    primary_model=None,
    meta_model=None,
    side_threshold=0.5,
    meta_threshold=0.5,
    meta_min_abs_ret=0.0,
    gap=121,
    sl=0.032,
    tp=0.032,
    long_only=True,
    drop_feats=('UBX', 'BTCUSD'),
    use_smart_slippage=True,
    smart_slippage_kwargs=None,
    use_illiquidity_gate=False,
    illiquidity_spread_df=None,
    illiquidity_spread_kwargs=None,
    weighting_scheme="equal",
    gate_debug=False,
):
    """Train primary+meta on train split and score a fixed test split.

    This mirrors ``run_primary_plus_meta`` but avoids synthetic-live dataset generation
    and instead returns predictions + evaluation artifacts for a provided test set.
    """
    if primary_model is None:
        primary_model = LGBMClassifier(verbose=-1)
    if meta_model is None:
        meta_model = make_pipeline(SimpleImputer(fill_value=-999), LogisticRegression())

    X_train = X_train.replace({np.inf: np.nan, -np.inf: np.nan}).copy().fillna(-999)
    X_test = X_test.replace({np.inf: np.nan, -np.inf: np.nan}).copy().fillna(-999)

    if drop_feats:
        keep_cols = [c for c in X_train.columns if not any(k in c for k in drop_feats)]
        X_train = X_train[keep_cols]
        X_test = X_test.reindex(columns=keep_cols, fill_value=-999)

    def _normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.to_datetime(out.index).normalize()
        return out.sort_index()

    px_open = _normalize_ohlc_index(px_open)
    px_high = _normalize_ohlc_index(px_high)
    px_low = _normalize_ohlc_index(px_low)
    px_close = _normalize_ohlc_index(px_close)
    if px_volume is not None:
        px_volume = _normalize_ohlc_index(px_volume)

    train_dates = pd.to_datetime(X_train.index.get_level_values('earnings_ts'))
    test_dates = pd.to_datetime(X_test.index.get_level_values('earnings_ts'))

    def _filter_ohlc(start_date, end_date, *dfs):
        buffer = pd.Timedelta(days=horizon + 10)
        out = []
        for df in dfs:
            out.append(df.truncate(before=start_date - buffer, after=end_date + buffer))
        return out

    px_open_train, px_high_train, px_low_train, px_close_train = _filter_ohlc(
        train_dates.min(), train_dates.max(), px_open, px_high, px_low, px_close
    )
    px_open_test, px_high_test, px_low_test, px_close_test = _filter_ohlc(
        test_dates.min(), test_dates.max(), px_open, px_high, px_low, px_close
    )

    px_volume_train = px_volume_test = None
    if px_volume is not None:
        (px_volume_train,) = _filter_ohlc(train_dates.min(), train_dates.max(), px_volume)
        (px_volume_test,) = _filter_ohlc(test_dates.min(), test_dates.max(), px_volume)

    dates = pd.Series(pd.to_datetime(X_train.index.get_level_values('earnings_ts')))
    cv = PurgedTimeSeriesSplit(dates=dates, gap=gap)

    p_oof_train = cv_predict_proba_purged(primary_model, X_train, y_train, cv).loc[X_train.index]

    d_train = train_data_with_event_day.copy()
    d_train['earnings_ts'] = pd.to_datetime(d_train['earnings_ts'])
    d_train['event_day'] = pd.to_datetime(d_train['event_day'])
    d_train = d_train.set_index(['ticker', 'earnings_ts']).sort_index().reindex(X_train.index)
    idx_train = d_train[['event_day']].copy()

    events_train, tmp_events_train, trades_train, el_train, xl_train, es_train, xs_train, pf_train = simulate_event_returns_from_proba(
        index_df=idx_train,
        p_primary=p_oof_train,
        px_open=px_open_train,
        px_high=px_high_train,
        px_low=px_low_train,
        px_close=px_close_train,
        px_volume=px_volume_train,
        horizon=horizon,
        side_threshold=side_threshold,
        tp=tp,
        sl=sl,
        long_only=long_only,
        use_smart_slippage=use_smart_slippage,
        smart_slippage_kwargs=smart_slippage_kwargs,
        use_illiquidity_gate=use_illiquidity_gate,
        illiquidity_spread_df=illiquidity_spread_df,
        illiquidity_spread_kwargs=illiquidity_spread_kwargs,
        weighting=weighting_scheme,
        debug=gate_debug,
        return_pf=True,
    )

    trade_ret_train = events_train['trade_ret'].dropna()
    common_idx = X_train.index.intersection(trade_ret_train.index)

    X_meta_base = X_train.loc[common_idx]
    p_oof_ok = p_oof_train.loc[common_idx]
    trade_ret_train = trade_ret_train.loc[common_idx]

    if long_only:
        long_mask = p_oof_ok >= side_threshold
        X_meta_base = X_meta_base.loc[long_mask]
        p_oof_ok = p_oof_ok.loc[long_mask]
        trade_ret_train = trade_ret_train.loc[long_mask]

    X_meta_train, y_meta_train = build_meta_dataset(
        X=X_meta_base,
        p_primary=p_oof_ok,
        trade_ret=trade_ret_train,
        min_abs_ret=meta_min_abs_ret,
    )

    primary_fit = primary_model
    primary_fit.fit(X_train, y_train)

    meta_fit = meta_model
    meta_fit.fit(X_meta_train, y_meta_train)

    p_primary_test = pd.Series(primary_fit.predict_proba(X_test)[:, 1], index=X_test.index)
    X_meta_test = X_test.copy()
    X_meta_test['p_primary'] = p_primary_test
    p_meta_test = pd.Series(
        meta_fit.predict_proba(X_meta_test[X_meta_train.columns])[:, 1],
        index=X_test.index,
    )

    d_test = test_data_with_event_day.copy()
    d_test['earnings_ts'] = pd.to_datetime(d_test['earnings_ts'])
    d_test['event_day'] = pd.to_datetime(d_test['event_day'])
    d_test = d_test.set_index(['ticker', 'earnings_ts']).sort_index().reindex(X_test.index)
    idx_test = d_test[['event_day']].copy()

    events_test, tmp_events_test, trades_test, el_test, xl_test, es_test, xs_test, pf_test = simulate_event_returns_from_proba(
        index_df=idx_test,
        p_primary=p_primary_test,
        px_open=px_open_test,
        px_high=px_high_test,
        px_low=px_low_test,
        px_close=px_close_test,
        px_volume=px_volume_test,
        horizon=horizon,
        side_threshold=side_threshold,
        tp=tp,
        sl=sl,
        long_only=long_only,
        use_smart_slippage=use_smart_slippage,
        smart_slippage_kwargs=smart_slippage_kwargs,
        use_illiquidity_gate=use_illiquidity_gate,
        illiquidity_spread_df=illiquidity_spread_df,
        illiquidity_spread_kwargs=illiquidity_spread_kwargs,
        weighting=weighting_scheme,
        debug=gate_debug,
        return_pf=True,
    )

    events_test = events_test.copy()
    events_test['p_meta'] = p_meta_test.reindex(events_test.index)
    signed = (events_test['p_primary'] * 2.0 - 1.0)
    if long_only:
        signed = signed.clip(lower=0.0)
    events_test['size'] = signed * events_test['p_meta']
    events_test['weighted_trade_ret'] = events_test['trade_ret'] * events_test['size']

    take_mask = p_meta_test >= meta_threshold
    gated_idx = idx_test.loc[take_mask.reindex(idx_test.index).fillna(False)]
    gated_p_primary = p_primary_test.loc[gated_idx.index]

    events_test_meta_gated, tmp_events_gated, trades_gated, el_gated, xl_gated, es_gated, xs_gated, pf_gated = simulate_event_returns_from_proba(
        index_df=gated_idx,
        p_primary=gated_p_primary,
        px_open=px_open_test,
        px_high=px_high_test,
        px_low=px_low_test,
        px_close=px_close_test,
        px_volume=px_volume_test,
        horizon=horizon,
        side_threshold=side_threshold,
        tp=tp,
        sl=sl,
        long_only=long_only,
        use_smart_slippage=use_smart_slippage,
        smart_slippage_kwargs=smart_slippage_kwargs,
        use_illiquidity_gate=use_illiquidity_gate,
        illiquidity_spread_df=illiquidity_spread_df,
        illiquidity_spread_kwargs=illiquidity_spread_kwargs,
        weighting=weighting_scheme,
        debug=gate_debug,
        return_pf=True,
    )
    events_test_meta_gated = events_test_meta_gated.copy()
    events_test_meta_gated['p_meta'] = p_meta_test.reindex(events_test_meta_gated.index)
    signed_gated = (events_test_meta_gated['p_primary'] * 2.0 - 1.0)
    if long_only:
        signed_gated = signed_gated.clip(lower=0.0)
    events_test_meta_gated['size'] = signed_gated * events_test_meta_gated['p_meta']
    events_test_meta_gated['weighted_trade_ret'] = (
        events_test_meta_gated['trade_ret'] * events_test_meta_gated['size']
    )

    test_scores = pd.DataFrame({'p_primary': p_primary_test, 'p_meta': p_meta_test})
    test_scores['size'] = (test_scores['p_primary'] * 2.0 - 1.0) * test_scores['p_meta']
    if long_only:
        test_scores['size'] = test_scores['size'].clip(lower=0.0)

    daily_meta_backtest = (
        events_test_meta_gated.reset_index()
        .assign(event_day=lambda d: pd.to_datetime(d['event_day']).dt.normalize())
        .groupby('event_day', as_index=True)['weighted_trade_ret']
        .sum()
        .sort_index()
        .rename('daily_weighted_ret')
        .to_frame()
    )
    daily_meta_backtest['equity_curve'] = (1.0 + daily_meta_backtest['daily_weighted_ret'].fillna(0.0)).cumprod()

    return test_scores.sort_values('size', ascending=False), {
        'p_oof_train': p_oof_train,
        'X_meta_train': X_meta_train,
        'y_meta_train': y_meta_train,
        'events_train': events_train,
        'events_test': events_test,
        'events_test_meta_gated': events_test_meta_gated,
        'daily_meta_backtest': daily_meta_backtest,
        'primary_model': primary_fit,
        'meta_model': meta_fit,
        'tmp_events_train': tmp_events_train,
        'trades_train': trades_train,
        'pf_train': pf_train,
        'signals_train': {'el': el_train, 'xl': xl_train, 'es': es_train, 'xs': xs_train},
        'tmp_events_test': tmp_events_test,
        'trades_test': trades_test,
        'pf_test': pf_test,
        'signals_test': {'el': el_test, 'xl': xl_test, 'es': es_test, 'xs': xs_test},
        'tmp_events_gated': tmp_events_gated,
        'trades_gated': trades_gated,
        'pf_gated': pf_gated,
        'signals_gated': {'el': el_gated, 'xl': xl_gated, 'es': es_gated, 'xs': xs_gated},
    }
