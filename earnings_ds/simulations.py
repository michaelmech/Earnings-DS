import numpy as np
import pandas as pd
import vectorbt as vbt


def calculate_smart_slippage(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    size_frac: pd.DataFrame,
    *,
    init_cash: float = 100_000.0,
    base_spread: float = 0.0002,
    vol_mult: float = 0.1,
    impact_mult: float = 0.1,
    min_slippage: float = 0.0005,
    max_slippage: float = 0.02,
) -> pd.DataFrame:
    """Estimate per-trade slippage with volatility + participation impact.

    The output is a matrix with the same [date x ticker] shape as `size_frac`,
    intended to be passed to vectorbt's `slippage` argument.
    """
    vol_comp = ((high_df - low_df) / open_df.replace(0.0, np.nan)) * float(vol_mult)

    trade_dollar_val = size_frac * float(init_cash)
    daily_dollar_volume = volume_df * close_df
    participation_rate = (trade_dollar_val / daily_dollar_volume.replace(0.0, np.nan)).fillna(0.0)
    impact_comp = float(impact_mult) * np.sqrt(participation_rate)

    total_slippage = float(base_spread) + vol_comp + impact_comp
    return total_slippage.clip(lower=float(min_slippage), upper=float(max_slippage)).fillna(0.0)


def calculate_agk_spread_proxy(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    *,
    min_spread: float = 0.0,
    max_spread: float = 100.0,
) -> pd.DataFrame:
    """Estimate per-bar spread in price units from OHLC for illiquidity gating.

    Notes
    -----
    This is an AGK-inspired proxy derived only from OHLC inputs and returned as
    a spread matrix in price units (same units as `close_df`).
    """
    del open_df  # kept for interface symmetry/future estimator swaps

    hl_rel = 2.0 * (high_df - low_df) / (high_df + low_df).replace(0.0, np.nan)
    spread_dollars = (hl_rel * close_df).abs()
    return spread_dollars.clip(lower=float(min_spread), upper=float(max_spread)).fillna(0.0)

def _dbg(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg)


def _df_nan_report(df: pd.DataFrame, name: str, enabled: bool, max_cols: int = 10) -> None:
    if not enabled:
        return
    total = df.size
    n_nan = int(df.isna().to_numpy().sum())
    _dbg(f"[{name}] shape={df.shape}  nan={n_nan}/{total} ({(n_nan/total if total else 0):.2%})", enabled)
    if n_nan:
        cols = df.isna().sum().sort_values(ascending=False)
        cols = cols[cols > 0].head(max_cols)
        if len(cols):
            _dbg(f"[{name}] top NaN columns:\n{cols.to_string()}", enabled)


def _bool_report(b: pd.DataFrame, name: str, enabled: bool) -> None:
    if not enabled:
        return
    n_true = int(b.to_numpy().sum())
    _dbg(f"[{name}] true={n_true}  days_with_any={int(b.any(axis=1).sum())}", enabled)


def _row_sum_report(df: pd.DataFrame, name: str, enabled: bool) -> None:
    if not enabled:
        return
    s = df.sum(axis=1)
    _dbg(
        f"[{name}] row_sum: min={float(s.min()):.6f}  max={float(s.max()):.6f}  "
        f"mean={float(s.mean()):.6f}  nonzero_days={int((s>0).sum())}",
        enabled,
    )
    bad = (s > 1.000001) | (s < -1e-9)
    if bad.any():
        _dbg(f"[{name}] WARNING: row_sum outside [0,1] on {int(bad.sum())} days (showing first 5):\n{s[bad].head()}", enabled)


def _expected_event_keys_from_signals(entries_long: pd.DataFrame, entries_short: pd.DataFrame) -> pd.MultiIndex:
    long_idx = entries_long.stack()
    long_idx = long_idx[long_idx].index

    short_idx = entries_short.stack()
    short_idx = short_idx[short_idx].index

    if len(long_idx) == 0 and len(short_idx) == 0:
        return pd.MultiIndex.from_arrays([[], []], names=["event_day", "ticker"])

    both = long_idx.append(short_idx)
    return both.unique().set_names(["event_day", "ticker"])


def _expected_event_keys_from_preds(proba_last_class: pd.Series, close: pd.DataFrame) -> pd.MultiIndex:
    idx = proba_last_class.index
    tick = idx.get_level_values(0).astype(str)
    day = pd.to_datetime(idx.get_level_values(1)).normalize()

    expected = pd.MultiIndex.from_arrays([day, tick], names=["event_day", "ticker"]).unique()
    valid_days = pd.DatetimeIndex(close.index).normalize().unique()
    valid_tickers = pd.Index(close.columns).astype(str).unique()

    mask = expected.get_level_values("event_day").isin(valid_days) & expected.get_level_values("ticker").isin(valid_tickers)
    return expected[mask]


def _trade_alignment_report(
    trades: pd.DataFrame,
    expected_keys: pd.MultiIndex,
    enabled: bool,
    max_examples: int = 5,
) -> dict[str, float | int]:
    if "Entry Timestamp" not in trades.columns or "Column" not in trades.columns:
        _dbg("[trade alignment] missing required trade columns for alignment report", enabled)
        return {
            "expected": int(len(expected_keys)),
            "actual": int(len(trades)),
            "overlap": 0,
            "overlap_ratio_expected": np.nan,
            "overlap_ratio_actual": np.nan,
        }

    t = trades.copy()
    t["event_day"] = pd.to_datetime(t["Entry Timestamp"]).dt.normalize()
    t["ticker"] = t["Column"].astype(str)
    actual_keys = pd.MultiIndex.from_frame(t[["event_day", "ticker"]]).unique()

    overlap = expected_keys.intersection(actual_keys)
    expected_only = expected_keys.difference(actual_keys)
    actual_only = actual_keys.difference(expected_keys)

    n_expected = len(expected_keys)
    n_actual = len(actual_keys)
    n_overlap = len(overlap)
    overlap_ratio_expected = (n_overlap / n_expected) if n_expected else np.nan
    overlap_ratio_actual = (n_overlap / n_actual) if n_actual else np.nan

    _dbg(
        "[trade alignment] "
        f"expected={n_expected} actual={n_actual} overlap={n_overlap} "
        f"overlap/expected={overlap_ratio_expected:.2%} overlap/actual={overlap_ratio_actual:.2%}",
        enabled,
    )

    if enabled and max_examples:
        if len(expected_only):
            ex = [f"({d.date()}, {t})" for d, t in list(expected_only)[:max_examples]]
            _dbg(f"[trade alignment] expected but missing examples: {ex}", enabled)
        if len(actual_only):
            ex = [f"({d.date()}, {t})" for d, t in list(actual_only)[:max_examples]]
            _dbg(f"[trade alignment] unexpected actual examples: {ex}", enabled)

    return {
        "expected": int(n_expected),
        "actual": int(n_actual),
        "overlap": int(n_overlap),
        "overlap_ratio_expected": float(overlap_ratio_expected) if pd.notna(overlap_ratio_expected) else np.nan,
        "overlap_ratio_actual": float(overlap_ratio_actual) if pd.notna(overlap_ratio_actual) else np.nan,
    }


def simulate_earnings_bidir_vbt(
    proba_last_class: pd.Series,
    ohlcv: dict[str, pd.DataFrame],
    *,
    init_cash: float = 100_000.0,
    stop_loss: float | None = 0.05,
    take_profit: float | None = 0.10,
    horizon: int = 5,
    min_proba_long: float | None = None,
    top_n_long: int | None = None,
    max_proba_short: float | None = None,
    top_n_short: int | None = None,
    weighting: str = "proba",  # "proba" or "equal"
    fees: float = 0.0,
    slippage: float = 0.0,
    freq: str = "1D",
    stop_exits_on_open_only: bool = True,
    # --- debug ---
    debug: bool = False,
    debug_show_examples: int = 5,
) -> vbt.Portfolio:
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlcv.keys())
    if missing:
        raise ValueError(f"ohlcv missing keys: {sorted(missing)}")

    close = ohlcv["close"].copy()
    open_ = ohlcv["open"].reindex_like(close).copy()
    high = ohlcv["high"].reindex_like(close).copy()
    low = ohlcv["low"].reindex_like(close).copy()

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if weighting not in ("equal", "proba"):
        raise ValueError("weighting must be 'equal' or 'proba'")

    if not isinstance(proba_last_class.index, pd.MultiIndex) or proba_last_class.index.nlevels != 2:
        raise TypeError("proba_last_class must be a Series with MultiIndex (ticker, earnings_ts).")

    idx = proba_last_class.index
    tick_name = idx.names[0] or "ticker"
    ts_name = idx.names[1] or "earnings_ts"

    preds_df = (
        proba_last_class.rename("proba")
        .reset_index()
        .rename(columns={tick_name: "ticker", ts_name: "earnings_ts"})
        .assign(earnings_ts=lambda d: pd.to_datetime(d["earnings_ts"]).dt.normalize())
        .pivot(index="earnings_ts", columns="ticker", values="proba")
        .sort_index()
        .reindex(index=close.index, columns=close.columns)
    )

    # --- debug: basic alignment and NaNs ---
    _dbg("==== DEBUG: inputs ====", debug)
    _dbg(f"[index] close.index: {close.index.min()} -> {close.index.max()}  n={len(close.index)}", debug)
    _dbg(f"[cols ] close.columns n={len(close.columns)}", debug)
    _df_nan_report(close, "close", debug)
    _df_nan_report(open_, "open", debug)
    _df_nan_report(high, "high", debug)
    _df_nan_report(low, "low", debug)
    _df_nan_report(preds_df, "preds_df(reindexed)", debug)

    # If close is super sparse, that alone can cause NaN stats
    if debug:
        usable_price_days = close.notna().any(axis=1).sum()
        _dbg(f"[close] days with any price data: {int(usable_price_days)}/{len(close)}", debug)

    def keep_topn_each_row(mat: pd.DataFrame, n: int) -> pd.DataFrame:
        def _row_topn(row: pd.Series) -> pd.Series:
            nn = row.notna().sum()
            if nn == 0 or nn <= n:
                return row
            keep_cols = row.nlargest(n).index
            return row.where(row.index.isin(keep_cols))
        return mat.apply(_row_topn, axis=1)

    def keep_bottomn_each_row(mat: pd.DataFrame, n: int) -> pd.DataFrame:
        def _row_bottomn(row: pd.Series) -> pd.Series:
            nn = row.notna().sum()
            if nn == 0 or nn <= n:
                return row
            keep_cols = row.nsmallest(n).index
            return row.where(row.index.isin(keep_cols))
        return mat.apply(_row_bottomn, axis=1)

    # --- LONG candidates ---
    long_cand = preds_df.copy()
    if min_proba_long is not None:
        long_cand = long_cand.where(long_cand >= float(min_proba_long))
    if top_n_long is not None:
        long_cand = keep_topn_each_row(long_cand, int(top_n_long))
    long_entries = long_cand.notna()

    # --- SHORT candidates ---
    short_cand = preds_df.copy()
    if max_proba_short is not None:
        short_cand = short_cand.where(short_cand <= float(max_proba_short))
    if top_n_short is not None:
        short_cand = keep_bottomn_each_row(short_cand, int(top_n_short))
    short_entries = short_cand.notna()

    # drop both sides if conflict
    both = long_entries & short_entries
    if both.to_numpy().any():
        long_entries = long_entries & ~both
        short_entries = short_entries & ~both
        long_cand = long_cand.where(long_entries)
        short_cand = short_cand.where(short_entries)

    _dbg("==== DEBUG: selection ====", debug)
    _bool_report(long_entries, "long_entries", debug)
    _bool_report(short_entries, "short_entries", debug)
    if debug and (both.to_numpy().any()):
        _dbg(f"[both] conflicts dropped: {int(both.to_numpy().sum())}", debug)

    # If entries exist on days where prices are NaN, vectorbt may drop/NaN things downstream.
    if debug:
        price_ok = close.notna()
        bad_long = long_entries & ~price_ok
        bad_short = short_entries & ~price_ok
        n_bad = int(bad_long.to_numpy().sum() + bad_short.to_numpy().sum())
        _dbg(f"[price check] entry signals on NaN close: {n_bad}", debug)
        if n_bad and debug_show_examples:
            ex = np.argwhere((bad_long | bad_short).to_numpy())[:debug_show_examples]
            for r, c in ex:
                side = "L" if bad_long.iat[r, c] else "S"
                _dbg(f"  example: {side} {close.columns[c]} on {close.index[r]} close=NaN", debug)

    # --- sizing (gross across both sides per day) ---
    if weighting == "equal":
        w_long = long_entries.astype(float)
        w_short = short_entries.astype(float)
    else:
        w_long = long_cand.where(long_entries)
        w_short = (1.0 - short_cand).where(short_entries)

    w_raw = (w_long.fillna(0.0) + w_short.fillna(0.0)).where((long_entries | short_entries), 0.0)
    w_sum = w_raw.sum(axis=1).replace(0.0, np.nan)
    size_frac = w_raw.div(w_sum, axis=0).fillna(0.0)
    size_frac = size_frac.where((long_entries | short_entries), 0.0)

    _dbg("==== DEBUG: sizing ====", debug)
    _row_sum_report(size_frac, "size_frac", debug)
    if debug:
        n_nonzero = int((size_frac.to_numpy() != 0).sum())
        _dbg(f"[size_frac] nonzero cells={n_nonzero}", debug)

    # --- time exits ---
    long_exits_time = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_exits_time = pd.DataFrame(False, index=close.index, columns=close.columns)

    for r, c in np.argwhere(long_entries.to_numpy()):
        exit_r = r + horizon
        if exit_r < len(close.index):
            long_exits_time.iat[exit_r, c] = True

    for r, c in np.argwhere(short_entries.to_numpy()):
        exit_r = r + horizon
        if exit_r < len(close.index):
            short_exits_time.iat[exit_r, c] = True

    _dbg("==== DEBUG: exits ====", debug)
    _bool_report(long_exits_time, "long_exits_time", debug)
    _bool_report(short_exits_time, "short_exits_time", debug)
    _dbg(f"[stops] stop_exits_on_open_only={stop_exits_on_open_only}", debug)

    # --- simulate ---
    # With daily bars, setting high/low to open enforces SL/TP checks at next-session open only.
    stop_high = open_ if stop_exits_on_open_only else high
    stop_low = open_ if stop_exits_on_open_only else low

    pf = vbt.Portfolio.from_signals(
        close=close,
        open=open_,
        high=stop_high,
        low=stop_low,
        entries=long_entries,
        exits=long_exits_time,
        short_entries=short_entries,
        short_exits=short_exits_time,
        size=size_frac,
        size_type="percent",
        init_cash=init_cash,
        cash_sharing=True,
        fees=fees,
        slippage=slippage,
        freq=freq,
        sl_stop=stop_loss,
        tp_stop=take_profit,
        stop_entry_price="close",
    )

    # --- debug: portfolio diagnostics ---
    _dbg("==== DEBUG: portfolio ====", debug)
    if debug:
        # trades
        n_trades = int(pf.trades.count())
        n_pos = int(pf.positions.count())
        _dbg(f"[trades] count={n_trades}  [positions] count={n_pos}", debug)

        # value / returns
        value = pf.value()
        ret = pf.returns()

        _dbg(f"[value] nan={int(np.isnan(value.to_numpy()).sum())}/{value.size}", debug)
        _dbg(f"[returns] nan={int(np.isnan(ret.to_numpy()).sum())}/{ret.size}", debug)
        if value.size:
            _dbg(f"[value] first/last: {float(value.iloc[0]):.2f} -> {float(value.iloc[-1]):.2f}", debug)

        # if stats NaN, usually one of these:
        if n_trades == 0:
            _dbg("NOTE: No trades executed -> many stats will be NaN.", debug)
        elif np.all(np.isnan(ret.to_numpy())):
            _dbg("NOTE: Returns are all-NaN -> stats will be NaN (check price NaNs around trade windows).", debug)

        # show a tiny sample of trade records if any
        if n_trades and debug_show_examples:
            rec = pf.trades.records_readable
            _dbg(f"[trades sample]\n{rec.head(debug_show_examples).to_string(index=False)}", debug)

        expected_keys = _expected_event_keys_from_signals(long_entries, short_entries)
        _trade_alignment_report(
            pf.trades.records_readable,
            expected_keys=expected_keys,
            enabled=debug,
            max_examples=debug_show_examples,
        )

    return pf


def simulate_earnings_long_vbt(
    proba_last_class: pd.Series,
    ohlcv: dict[str, pd.DataFrame],
    *,
    init_cash: float = 100_000.0,
    stop_loss: float | None = 0.05,  # 5% SL from entry
    take_profit: float | None = 0.10,  # 10% TP from entry
    horizon: int = 5,  # exit at CLOSE after N trading days
    min_proba: float | None = None,
    top_n: int | None = None,
    weighting: str = "proba",  # "proba" or "equal"
    fees: float = 0.0,
    slippage: float = 0.0,
    freq: str = "1D",
    stop_exits_on_open_only: bool = True,
    # --- debug ---
    debug: bool = False,
    debug_show_examples: int = 5,
) -> vbt.Portfolio:
    """Simulate long entries on earnings dates using daily OHLCV.

    Assumptions / mechanics:
    - Entry is executed at the CLOSE on the (aligned) earnings date.
    - SL/TP are evaluated starting from the next bar (t+1). By default they are
      evaluated using next-session OPEN only (`stop_exits_on_open_only=True`).
    - Time exit is executed at the CLOSE after `horizon` trading days.
    - Position sizing is computed daily across that day's entries using `size_type='percent'`
      and `cash_sharing=True`.

    Inputs:
    - proba_last_class: Series with MultiIndex (ticker, earnings_ts) and values = probability.
    - ohlcv: dict with keys {'open','high','low','close'} each a DataFrame [dates x tickers].

    Notes on alignment:
    - Your OHLCV index is daily (midnight) tz-naive. Earnings timestamps often contain
      intraday times (e.g., 16:00). We normalize earnings_ts to midnight so they match
      the trading calendar exactly.
    """

    # --- validate OHLCV ---
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlcv.keys())
    if missing:
        raise ValueError(f"ohlcv missing keys: {sorted(missing)}")

    close = ohlcv["close"].copy()
    open_ = ohlcv["open"].reindex_like(close).copy()
    high = ohlcv["high"].reindex_like(close).copy()
    low = ohlcv["low"].reindex_like(close).copy()

    # --- validate preds index ---
    if not isinstance(proba_last_class.index, pd.MultiIndex) or proba_last_class.index.nlevels != 2:
        raise TypeError("proba_last_class must be a Series with MultiIndex (ticker, earnings_ts).")

    idx = proba_last_class.index
    tick_name = idx.names[0] or "ticker"
    ts_name = idx.names[1] or "earnings_ts"

    # --- reshape predictions to [dates x tickers] ---
    # Normalize earnings_ts to midnight so it matches close.index exactly.
    preds_df = (
        proba_last_class.rename("proba")
        .reset_index()
        .rename(columns={tick_name: "ticker", ts_name: "earnings_ts"})
        .assign(earnings_ts=lambda d: pd.to_datetime(d["earnings_ts"]).dt.normalize())
        .pivot(index="earnings_ts", columns="ticker", values="proba")
        .sort_index()
    )

    # align to trading calendar/universe (exact match on dates and tickers)
    preds_df = preds_df.reindex(index=close.index, columns=close.columns)

    _dbg("==== DEBUG: inputs ====", debug)
    _dbg(f"[index] close.index: {close.index.min()} -> {close.index.max()}  n={len(close.index)}", debug)
    _dbg(f"[cols ] close.columns n={len(close.columns)}", debug)
    _df_nan_report(close, "close", debug)
    _df_nan_report(open_, "open", debug)
    _df_nan_report(high, "high", debug)
    _df_nan_report(low, "low", debug)
    _df_nan_report(preds_df, "preds_df(reindexed)", debug)

    if debug:
        expected_from_preds = _expected_event_keys_from_preds(proba_last_class, close)
        _dbg(
            "[preds index alignment] "
            f"raw={len(proba_last_class.index)} unique_raw={len(proba_last_class.index.unique())} "
            f"in_universe={len(expected_from_preds)}",
            debug,
        )

    # --- selection ---
    candidates = preds_df
    if min_proba is not None:
        candidates = candidates.where(candidates >= float(min_proba))

    if top_n is not None:

        def keep_topn(row: pd.Series) -> pd.Series:
            nn = row.notna().sum()
            if nn == 0 or nn <= top_n:
                return row
            keep_cols = row.nlargest(top_n).index
            return row.where(row.index.isin(keep_cols))

        candidates = candidates.apply(keep_topn, axis=1)

    entries = candidates.notna()

    _dbg("==== DEBUG: selection ====", debug)
    _bool_report(entries, "entries", debug)

    if debug:
        price_ok = close.notna()
        bad_entries = entries & ~price_ok
        n_bad = int(bad_entries.to_numpy().sum())
        _dbg(f"[price check] entry signals on NaN close: {n_bad}", debug)
        if n_bad and debug_show_examples:
            ex = np.argwhere(bad_entries.to_numpy())[:debug_show_examples]
            for r, c in ex:
                _dbg(f"  example: {close.columns[c]} on {close.index[r]} close=NaN", debug)

    # --- sizing: per-day allocation fractions (0..1), only applied on entry bars ---
    if weighting not in ("equal", "proba"):
        raise ValueError("weighting must be 'equal' or 'proba'")

    if weighting == "equal":
        w_raw = entries.astype(float)
    else:
        w_raw = candidates.where(entries)

    w_sum = w_raw.sum(axis=1).replace(0.0, np.nan)
    size_frac = w_raw.div(w_sum, axis=0).fillna(0.0)
    size_frac = size_frac.where(entries, 0.0)

    _dbg("==== DEBUG: sizing ====", debug)
    _row_sum_report(size_frac, "size_frac", debug)
    if debug:
        n_nonzero = int((size_frac.to_numpy() != 0).sum())
        _dbg(f"[size_frac] nonzero cells={n_nonzero}", debug)

    # --- time exits at CLOSE on bar + horizon ---
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    exits_time = pd.DataFrame(False, index=close.index, columns=close.columns)
    entry_rc = np.argwhere(entries.to_numpy())

    for r, c in entry_rc:
        exit_r = r + horizon
        if exit_r < exits_time.shape[0]:
            exits_time.iat[exit_r, c] = True

    _dbg("==== DEBUG: exits ====", debug)
    _bool_report(exits_time, "exits_time", debug)
    _dbg(f"[stops] stop_exits_on_open_only={stop_exits_on_open_only}", debug)

    # --- simulate ---
    # With daily bars, setting high/low to open enforces SL/TP checks at next-session open only.
    stop_high = open_ if stop_exits_on_open_only else high
    stop_low = open_ if stop_exits_on_open_only else low

    pf = vbt.Portfolio.from_signals(
        close=close,
        open=open_,
        high=stop_high,
        low=stop_low,
        entries=entries,
        exits=exits_time,
        size=size_frac,
        size_type="percent",
        init_cash=init_cash,
        cash_sharing=True,
        fees=fees,
        slippage=slippage,
        freq=freq,
        sl_stop=stop_loss,
        tp_stop=take_profit,
        stop_entry_price="close",  # buy at close of earnings day
    )

    _dbg("==== DEBUG: portfolio ====", debug)
    if debug:
        n_trades = int(pf.trades.count())
        n_pos = int(pf.positions.count())
        _dbg(f"[trades] count={n_trades}  [positions] count={n_pos}", debug)

        value = pf.value()
        ret = pf.returns()
        _dbg(f"[value] nan={int(np.isnan(value.to_numpy()).sum())}/{value.size}", debug)
        _dbg(f"[returns] nan={int(np.isnan(ret.to_numpy()).sum())}/{ret.size}", debug)
        if value.size:
            _dbg(f"[value] first/last: {float(value.iloc[0]):.2f} -> {float(value.iloc[-1]):.2f}", debug)

        if n_trades == 0:
            _dbg("NOTE: No trades executed -> many stats will be NaN.", debug)
        elif np.all(np.isnan(ret.to_numpy())):
            _dbg("NOTE: Returns are all-NaN -> stats will be NaN (check price NaNs around trade windows).", debug)

        if n_trades and debug_show_examples:
            rec = pf.trades.records_readable
            _dbg(f"[trades sample]\n{rec.head(debug_show_examples).to_string(index=False)}", debug)

        _dbg("[trade alignment] baseline=preds_in_universe", debug)
        expected_from_preds = _expected_event_keys_from_preds(proba_last_class, close)
        _trade_alignment_report(
            pf.trades.records_readable,
            expected_keys=expected_from_preds,
            enabled=debug,
            max_examples=debug_show_examples,
        )

        _dbg("[trade alignment] baseline=selected_entries", debug)
        empty_short = pd.DataFrame(False, index=entries.index, columns=entries.columns)
        expected_from_entries = _expected_event_keys_from_signals(entries, empty_short)
        _trade_alignment_report(
            pf.trades.records_readable,
            expected_keys=expected_from_entries,
            enabled=debug,
            max_examples=debug_show_examples,
        )

    return pf


def trade_pnl_like_preds(pf, proba_last_class: pd.Series) -> pd.Series:
    """Return per-trade PnL indexed like `proba_last_class`.

    Output:
    - Series indexed by MultiIndex (ticker, earnings_ts) matching the exact ordering of
      `proba_last_class.index`.
    - If a trade did not occur for a given (ticker, earnings_ts), value will be NaN.

    Important alignment detail:
    - Your OHLCV index is daily tz-naive at midnight. To avoid mismatches between
      trade entry timestamps (from vectorbt) and prediction timestamps (often intraday),
      we normalize BOTH to midnight before reindexing.
    """

    rec = pf.trades.records_readable.copy()

    # --- ticker + entry timestamp extraction (version-tolerant) ---
    ticker = rec["Column"] if "Column" in rec.columns else pf.wrapper.columns[rec["col"].to_numpy()]

    if "Entry Timestamp" in rec.columns:
        entry_ts = pd.to_datetime(rec["Entry Timestamp"])
    else:
        entry_ts = pd.to_datetime(pf.wrapper.index[rec["entry_idx"].to_numpy()])

    # Normalize to midnight to match daily close.index convention
    entry_ts = pd.to_datetime(entry_ts).dt.normalize()

    # --- pnl extraction (version-tolerant) ---
    if "PnL" in rec.columns:
        pnl = rec["PnL"].astype(float)
    elif "pnl" in rec.columns:
        pnl = rec["pnl"].astype(float)
    else:
        pnl = pd.Series(pf.trades.pnl, index=rec.index).astype(float)

    out = (
        pd.Series(
            pnl.to_numpy(),
            index=pd.MultiIndex.from_arrays([ticker, entry_ts], names=proba_last_class.index.names),
            name="trade_pnl",
        )
        .groupby(level=[0, 1])
        .sum()
    )

    # --- normalize preds index the SAME way, then map back to original ordering ---
    idx = proba_last_class.index
    lvl0 = idx.get_level_values(0)
    lvl1 = pd.to_datetime(idx.get_level_values(1)).normalize()
    idx_norm = pd.MultiIndex.from_arrays([lvl0, lvl1], names=idx.names)

    # Reindex to normalized index (for matching), then restore original index labels/order
    return out.reindex(idx_norm).set_axis(idx)


def make_event_signal_matrices(
    index_df,
    px_close,
    p_primary,
    horizon=5,
    side_threshold=0.5,
    debug=False,
    map_to_next_trading_day=False,
    illiquidity_spread_df: pd.DataFrame | None = None,
    illiquidity_threshold_by_event: pd.Series | None = None,
    sample_n=10,
):
    """
    index_df: DataFrame aligned to p_primary.index with column 'event_day'
             Index should be (ticker, earnings_ts)
    px_close: wide DataFrame (dates x tickers)
    p_primary: Series indexed by (ticker, earnings_ts)
    """

    # ---- debug header ----
    if debug:
        print("\n=== make_event_signal_matrices DEBUG ===")
        print("index_df.index.names:", index_df.index.names)
        print("p_primary.index.names:", p_primary.index.names)
        print("px_close.index.name(s):", getattr(px_close.index, "names", None))
        print("px_close.shape:", px_close.shape)
        print("px_close.index dtype:", px_close.index.dtype)
        print("px_close.index min/max:", px_close.index.min(), px_close.index.max())
        print("px_close.columns (n):", len(px_close.columns))

    # Ensure px_close has DatetimeIndex
    px_close = px_close.copy()
    px_close.index = pd.to_datetime(px_close.index)

    # Ensure event_day exists and align

    idx = p_primary.index
    tmp = index_df.reindex(idx)[['event_day']].copy()

    tmp['p_primary'] = p_primary.reindex(tmp.index).values

    tmp = tmp.dropna(subset=['event_day', 'p_primary'])

    if debug:
        print("\n-- after reindex --")
        print("tmp.shape:", tmp.shape)
        dedup_col = "event_day_use" if "event_day_use" in tmp.columns else ("event_day_norm" if "event_day_norm" in tmp.columns else "event_day")
        dedup = tmp.reset_index().drop_duplicates(["ticker", dedup_col]).shape[0]
        print(f"distinct (ticker, {dedup_col}):", dedup)
        print("tmp.event_day null count:", tmp['event_day'].isna().sum())
        print("tmp head:")
        print(tmp.head(3))

    # Convert event_day to datetime
    tmp['event_day_raw'] = tmp['event_day']  # keep original for debugging
    tmp['event_day'] = pd.to_datetime(tmp['event_day'], errors='coerce')

    if debug:
        bad_parse = tmp['event_day'].isna().sum()
        print("\n-- datetime parse --")
        print("bad event_day parse count:", bad_parse)
        if bad_parse > 0:
            print("examples of unparseable event_day:")
            print(tmp.loc[tmp['event_day'].isna(), 'event_day_raw'].head(sample_n))

    # Normalize to date (midnight)
    tmp['event_day_norm'] = tmp['event_day'].dt.normalize()

    # Also normalize px index (common cause: px index has time component or is tz-aware)
    px_norm = pd.to_datetime(px_close.index).normalize()
    px_close.index = px_norm

    if debug:
        print("\n-- normalization --")
        print("tmp event_day_norm min/max:",
              tmp['event_day_norm'].min(), tmp['event_day_norm'].max())
        print("px_close.index (normalized) min/max:",
              px_close.index.min(), px_close.index.max())

        # Check if px index has duplicates after normalize
        dup = px_close.index.duplicated().sum()
        if dup:
            print("WARNING: px_close index has duplicates after normalize:", dup)

    # Drop rows with missing event_day
    tmp = tmp.dropna(subset=['event_day_norm'])
    if debug:
        print("\n-- after dropping NaN event_day_norm --")
        print("tmp.shape:", tmp.shape)
        dedup_col = "event_day_use" if "event_day_use" in tmp.columns else ("event_day_norm" if "event_day_norm" in tmp.columns else "event_day")
        dedup = tmp.reset_index().drop_duplicates(["ticker", dedup_col]).shape[0]
        print(f"distinct (ticker, {dedup_col}):", dedup)

    # Check overlap BEFORE filtering
    in_px = tmp['event_day_norm'].isin(px_close.index)
    if debug:
        print("\n-- overlap check --")
        print("events on px dates:", int(in_px.sum()), " / ", len(in_px))

        if len(in_px) > 0:
            # show a few that miss
            miss = tmp.loc[~in_px, 'event_day_norm'].dropna()
            if len(miss) > 0:
                print("sample missing event_day_norm values:")
                print(miss.head(sample_n).to_list())

            # common reason: off by one day due to after-hours; check next/prev trading day existence
            miss_days = tmp.loc[~in_px, 'event_day_norm'].dropna()
            if len(miss_days) > 0:
                # count weekends quickly
                weekend = miss_days.dt.weekday >= 5
                print("missing that fall on weekend:", int(weekend.sum()), "/", len(miss_days))

                # check if +1 day would match
                plus1 = (miss_days + pd.Timedelta(days=1)).isin(px_close.index)
                minus1 = (miss_days - pd.Timedelta(days=1)).isin(px_close.index)
                print("missing where +1 day matches px:", int(plus1.sum()), "/", len(miss_days))
                print("missing where -1 day matches px:", int(minus1.sum()), "/", len(miss_days))

    # Optionally map to next available trading day instead of dropping
    if map_to_next_trading_day:
        px_dates = px_close.index.values  # datetime64[ns]
        ed = tmp['event_day_norm'].values.astype('datetime64[ns]')

        pos = np.searchsorted(px_dates, ed, side='left')
        pos = np.clip(pos, 0, len(px_dates) - 1)
        mapped = px_dates[pos]

        if debug:
            mapped_in = pd.Series(mapped).isin(px_close.index).mean()
            print("\n-- mapping to next trading day --")
            print("mapped coverage ratio:", mapped_in)

        tmp['event_day_use'] = mapped
    else:
        tmp['event_day_use'] = tmp['event_day_norm']

    # Now filter to those in px index
    tmp = tmp[tmp['event_day_use'].isin(px_close.index)]

    if debug:
        print("\n-- final kept events --")
        print("tmp.shape:", tmp.shape)
        dedup_col = "event_day_use" if "event_day_use" in tmp.columns else ("event_day_norm" if "event_day_norm" in tmp.columns else "event_day")
        dedup = tmp.reset_index().drop_duplicates(["ticker", dedup_col]).shape[0]
        print(f"distinct (ticker, {dedup_col}):", dedup)
        if tmp.shape[0] == 0:
            # helpful diagnostics: range mismatch
            print("NO EVENTS KEPT.")
            print("event_day_norm unique count:", tmp['event_day_norm'].nunique() if 'event_day_norm' in tmp else None)
            # show original pre-filter ranges from earlier captured vars (recompute quickly)
            # (we still have px range above)
        else:
            print("sample kept rows:")
            print(tmp.head(3)[['event_day_raw', 'event_day', 'event_day_norm', 'event_day_use', 'p_primary']])

    if tmp.empty:
        raise ValueError("No events matched to px_close index after normalization/filtering.")

    dates = px_close.index
    tickers = px_close.columns

    entries_long  = pd.DataFrame(False, index=dates, columns=tickers)
    entries_short = pd.DataFrame(False, index=dates, columns=tickers)

    long_mask = tmp['p_primary'] >= side_threshold

    blocked_by_illiquidity = 0
    skipped_missing_ticker = 0

    for (tkr, ets), row in tmp.iterrows():
        d = row['event_day_use']
        if tkr not in tickers:
            skipped_missing_ticker += 1
            continue

        if illiquidity_spread_df is not None and illiquidity_threshold_by_event is not None:
            est_spread = illiquidity_spread_df.at[d, tkr] if (d in illiquidity_spread_df.index and tkr in illiquidity_spread_df.columns) else np.nan
            spread_cap = illiquidity_threshold_by_event.loc[(tkr, ets)] if (tkr, ets) in illiquidity_threshold_by_event.index else np.nan
            if pd.notna(est_spread) and pd.notna(spread_cap) and float(est_spread) > float(spread_cap):
                blocked_by_illiquidity += 1
                continue

        if long_mask.loc[(tkr, ets)]:
            entries_long.loc[d, tkr] = True
        else:
            entries_short.loc[d, tkr] = True

    if debug:
        total_events = len(tmp)
        total_entries = int(entries_long.values.sum() + entries_short.values.sum())
        print(f"[gate debug] total_events={total_events} entries_after_gates={total_entries}")
        if skipped_missing_ticker:
            print(f"[gate debug] skipped_missing_ticker={skipped_missing_ticker}")
        if illiquidity_spread_df is not None and illiquidity_threshold_by_event is not None:
            print(f"[illiquidity gate] blocked events: {blocked_by_illiquidity}")

    exits_long  = entries_long.vbt.signals.fshift(horizon, fill_value=False)
    exits_short = entries_short.vbt.signals.fshift(horizon, fill_value=False)

    # return tmp with chosen event day
    out_tmp = tmp.copy()
    out_tmp['event_day'] = out_tmp['event_day_use']
    out_tmp = out_tmp.drop(columns=['event_day_use'])

    return out_tmp, entries_long, exits_long, entries_short, exits_short


def attach_returns_to_events(tmp_events, trades, px_close):
    """
    Map vectorbt trades back to your event rows via (ticker, event_day).
    tmp_events: DataFrame indexed by (ticker, earnings_ts) with event_day + p_primary
    trades: vectorbt readable records
    """
    # trades has 'Column' (ticker), 'Entry Timestamp' etc in readable form
    t = trades.copy()
    t['event_day'] = pd.to_datetime(t['Entry Timestamp']).dt.normalize()
    t['ticker'] = t['Column'].astype(str)
    t['trade_ret'] = t['Return'].astype(float)

    # aggregate if multiple trades somehow occur same day per ticker (shouldn't, but safe)
    trade_map = (
        t.groupby(['ticker', 'event_day'], as_index=False)['trade_ret']
         .mean()
    )

    out = tmp_events.copy()
    out['event_day'] = pd.to_datetime(out['event_day']).dt.normalize()
    out = out.reset_index()
    out = out.merge(trade_map, on=['ticker', 'event_day'], how='left')
    out = out.set_index(['ticker', 'earnings_ts'])

    return out


def simulate_event_returns_from_proba(
    *,
    index_df: pd.DataFrame,
    p_primary: pd.Series,
    px_open: pd.DataFrame,
    px_high: pd.DataFrame,
    px_low: pd.DataFrame,
    px_close: pd.DataFrame,
    px_volume: pd.DataFrame | None = None,
    horizon: int = 5,
    side_threshold: float = 0.5,
    tp: float = 0.032,
    sl: float = 0.032,
    long_only: bool = True,
    use_smart_slippage: bool = True,
    smart_slippage_kwargs: dict | None = None,
    use_illiquidity_gate: bool = False,
    illiquidity_spread_df: pd.DataFrame | None = None,
    illiquidity_spread_fn=calculate_agk_spread_proxy,
    illiquidity_spread_kwargs: dict | None = None,
    debug: bool = False,
    return_pf: bool = False,
):
    """Run the canonical event->signals->vectorbt->event returns pipeline.

    This helper exists to keep simulation methodology identical across workflows
    (e.g., purged-CV OOF probabilities and chronological holdout probabilities).

    Returns
    -------
    events_with_ret : pd.DataFrame
        Event table indexed by (ticker, earnings_ts) with `trade_ret` attached.
    tmp_events : pd.DataFrame
        Event table used to create signal matrices.
    trades : pd.DataFrame
        `vectorbt` trade records in readable format.
    el, xl, es, xs : pd.DataFrame
        Long/short entry/exit signal matrices used in simulation.
    pf : vbt.Portfolio, optional
        Returned when ``return_pf=True`` for direct portfolio-level analysis.
    """
    if use_illiquidity_gate:
        spread_df = illiquidity_spread_df
        if spread_df is None:
            spread_df = illiquidity_spread_fn(
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
            .rename_axis(["event_day", "ticker"])
            .rename("spread_cap")
            .reorder_levels(["ticker", "event_day"])
        )
    else:
        spread_df = None
        illiquidity_threshold_by_event = None

    tmp_events, el, xl, es, xs = make_event_signal_matrices(
        index_df=index_df,
        px_close=px_close,
        p_primary=p_primary,
        horizon=horizon,
        side_threshold=side_threshold,
        illiquidity_spread_df=spread_df,
        illiquidity_threshold_by_event=illiquidity_threshold_by_event,
        debug=debug,
    )

    if long_only:
        es[:] = False
        xs[:] = False

    trades_out = vectorbt_trade_returns_gapaware(
        open_df=px_open,
        high_df=px_high,
        low_df=px_low,
        close_df=px_close,
        entries_long=el,
        exits_long=xl,
        entries_short=es,
        exits_short=xs,
        tp=tp,
        sl=sl,
        volume_df=px_volume,
        use_smart_slippage=use_smart_slippage,
        smart_slippage_kwargs=smart_slippage_kwargs,
        debug=debug,
        return_pf=return_pf,
    )

    if return_pf:
        trades, pf = trades_out
        _print_pf_key_stats(pf, prefix='[simulate_event_returns_from_proba] ')
    else:
        trades = trades_out

    events_with_ret = attach_returns_to_events(tmp_events, trades, px_close)
    if return_pf:
        return events_with_ret, tmp_events, trades, el, xl, es, xs, pf

    return events_with_ret, tmp_events, trades, el, xl, es, xs


def _print_pf_key_stats(pf, prefix: str = ""):
    stats = pf.stats()
    key_candidates = {
        'sharpe_ratio': ['Sharpe Ratio'],
        'max_drawdown': ['Max Drawdown [%]', 'Max Drawdown'],
        'total_return': ['Total Return [%]', 'Total Return'],
        'win_rate': ['Win Rate [%]', 'Win Rate'],
        'trades': ['Total Trades', 'Total Closed Trades'],
    }

    def _pick(candidates):
        for c in candidates:
            if c in stats.index:
                return stats[c]
        return np.nan

    selected = {k: _pick(v) for k, v in key_candidates.items()}
    print(
        f"{prefix}pf.stats selected -> "
        f"Sharpe Ratio: {selected['sharpe_ratio']}, "
        f"Max Drawdown: {selected['max_drawdown']}, "
        f"Total Return: {selected['total_return']}, "
        f"Win Rate: {selected['win_rate']}, "
        f"# Trades: {selected['trades']}"
    )


def vectorbt_trade_returns_gapaware(
    open_df, high_df, low_df, close_df,
    entries_long, exits_long, entries_short, exits_short,
    volume_df=None,
    tp=0.02, sl=0.01, init_cash=1.0,
    stop_exits_on_open_only: bool = True,
    use_smart_slippage: bool = True,
    smart_slippage_kwargs: dict | None = None,
    debug=False,
    debug_show_examples=5,
    return_pf: bool = False,
):
    # Enums live in the docs under portfolio.enums (StopEntryPrice / StopExitPrice) :contentReference[oaicite:2]{index=2}
    StopEntryPrice = vbt.portfolio.enums.StopEntryPrice
    StopExitPrice  = vbt.portfolio.enums.StopExitPrice

    # With daily bars, setting high/low to open enforces SL/TP checks at next-session open only.
    stop_high = open_df if stop_exits_on_open_only else high_df
    stop_low = open_df if stop_exits_on_open_only else low_df

    # Match sizing semantics used in other simulation functions:
    # - normalize same-day candidate weights to sum to 1.0
    # - pass percent sizes with shared cash pool across columns
    w_raw = (entries_long.astype(float) + entries_short.astype(float)).where(
        (entries_long | entries_short),
        0.0,
    )
    w_sum = w_raw.sum(axis=1).replace(0.0, np.nan)
    size_frac = w_raw.div(w_sum, axis=0).fillna(0.0)
    size_frac = size_frac.where((entries_long | entries_short), 0.0)

    slippage_value: float | pd.DataFrame
    if use_smart_slippage and volume_df is not None:
        slippage_value = calculate_smart_slippage(
            open_df=open_df,
            high_df=high_df,
            low_df=low_df,
            close_df=close_df,
            volume_df=volume_df.reindex_like(close_df),
            size_frac=size_frac,
            init_cash=init_cash,
            **(smart_slippage_kwargs or {}),
        )
    else:
        slippage_value = 0.0

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        open=open_df,
        high=stop_high,
        low=stop_low,

        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,

        size=size_frac,
        size_type="percent",

        # Percent stops relative to entry
        sl_stop=sl,
        tp_stop=tp,

        # Entry at close of signal bar (event day close)
        stop_entry_price=StopEntryPrice.Close,

        # If gapped through stop/TP, exit at next bar open; otherwise intra-bar via OHLC.
        # StopMarket applies slippage; StopLimit does not :contentReference[oaicite:3]{index=3}
        stop_exit_price=StopExitPrice.StopMarket,

        init_cash=init_cash,
        cash_sharing=True,
        fees=0.0,
        slippage=slippage_value,
        freq="1D",
        direction="both"
    )

    trades = pf.trades.records_readable.copy()

    _dbg(f"[stops] stop_exits_on_open_only={stop_exits_on_open_only}", debug)

    expected_keys = _expected_event_keys_from_signals(entries_long, entries_short)
    _trade_alignment_report(
        trades,
        expected_keys=expected_keys,
        enabled=debug,
        max_examples=debug_show_examples,
    )

    if return_pf:
        return trades, pf

    return trades
