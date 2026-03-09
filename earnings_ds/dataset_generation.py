import numpy as np
import pandas as pd
import yfinance as yf

def add_earnings_edge_features(
    df: pd.DataFrame,
    close: pd.Series,
    ret: pd.Series,
    vol: pd.Series | None = None,
    rets_panel: pd.DataFrame | None = None,      # your `rets` DataFrame (all tickers)
    earnings_ts: np.ndarray | None = None,       # output of _extract_earnings_ts(t)
    *,
    market_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    trend_windows: tuple[int, ...] = (20, 60, 252),
    hl_windows: tuple[int, ...] = (20, 60, 252),
    stat_windows: tuple[int, ...] = (20, 60),
    vwap_windows: tuple[int, ...] = (20, 60),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Adds extra features into df (aligned on df.index), no forward leakage for "earnings memory":
      - rolling log-price linear trend slope + R^2 (vectorized via convolution)
      - drawdown / run-up vs rolling max/min
      - return shape (skew/kurt), autocorr(1), downside realized vol
      - realized vol regime (rv_5/10/60 and ratios vs 20d)
      - liquidity/impact: dollar volume, dollar-volume z, Amihud illiquidity
      - OBV pressure and OBV z-score
      - rolling VWAP (price*vol / vol) deviation
      - short-horizon corr/beta vs SPY (and VIX corr if present)
      - earnings memory: previous earnings reactions, trailing mean/std/winrate, trading-days since prev earnings
    """

    idx = df.index
    close = close.reindex(idx).astype(float)
    ret = ret.reindex(idx).astype(float)

    # ---------- helpers ----------
    def _rolling_linreg_slope_r2(y: np.ndarray, w: int):
        """
        y: 1D array of floats (may contain NaN)
        returns slope and r2 arrays aligned to y length, where value appears at window end (t),
        i.e., first non-NaN at t = w-1
        """
        n = y.size
        slope = np.full(n, np.nan)
        r2 = np.full(n, np.nan)
        if n < w:
            return slope, r2

        x = np.arange(w, dtype=float)
        sum_x = x.sum()
        sum_x2 = (x * x).sum()
        denom = (w * sum_x2 - sum_x * sum_x)

        ones = np.ones(w, dtype=float)
        wts_rev = x[::-1].astype(float)

        y0 = np.nan_to_num(y, nan=0.0)
        finite = np.isfinite(y).astype(float)

        cnt = np.convolve(finite, ones, mode="valid")
        sum_y = np.convolve(y0, ones, mode="valid")
        sum_y2 = np.convolve(y0 * y0, ones, mode="valid")
        sum_xy = np.convolve(y0, wts_rev, mode="valid")

        ok = cnt == w
        b = np.full(sum_y.shape, np.nan)
        a = np.full(sum_y.shape, np.nan)

        b[ok] = (w * sum_xy[ok] - sum_x * sum_y[ok]) / (denom + eps)
        a[ok] = (sum_y[ok] - b[ok] * sum_x) / w

        ss_tot = np.full(sum_y.shape, np.nan)
        ss_res = np.full(sum_y.shape, np.nan)

        ss_tot[ok] = sum_y2[ok] - (sum_y[ok] * sum_y[ok]) / w
        ss_res[ok] = (
            sum_y2[ok]
            + w * (a[ok] * a[ok])
            + (b[ok] * b[ok]) * sum_x2
            - 2.0 * a[ok] * sum_y[ok]
            - 2.0 * b[ok] * sum_xy[ok]
            + 2.0 * a[ok] * b[ok] * sum_x
        )

        rr = 1.0 - (ss_res / (ss_tot + eps))
        rr = np.clip(rr, -1.0, 1.0)

        slope[w - 1 :] = b
        r2[w - 1 :] = rr
        return slope, r2

    # ---------- calendar / seasonality (numerical) ----------
    # (useful if your event-days skew by weekday/month, and it's not in your existing feature set)
    dow = idx.dayofweek.to_numpy(dtype=float)  # Mon=0
    mon = idx.month.to_numpy(dtype=float)      # 1..12
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    df["month_sin"] = np.sin(2 * np.pi * (mon - 1.0) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (mon - 1.0) / 12.0)

    # ---------- trend fit on log-price ----------
    lp = np.log(close.replace(0.0, np.nan))
    y = lp.to_numpy(dtype=float)

    for w in trend_windows:
        sl, rr = _rolling_linreg_slope_r2(y, w)
        df[f"trend_lp_slope_{w}"] = sl                        # log-price per day
        df[f"trend_lp_r2_{w}"] = rr                           # fit quality
        df[f"trend_ann_{w}"] = np.exp(sl * 252.0) - 1.0       # implied annualized drift

    # ---------- drawdown / run-up vs rolling extrema ----------
    for w in hl_windows:
        roll_max = close.rolling(w).max()
        roll_min = close.rolling(w).min()
        df[f"dd_{w}"] = (close / (roll_max + eps)) - 1.0      # <= 0
        df[f"ru_{w}"] = (close / (roll_min + eps)) - 1.0      # >= 0

    # ---------- return distribution + dependence ----------
    for w in stat_windows:
        df[f"ret_skew_{w}"] = ret.rolling(w).skew()
        df[f"ret_kurt_{w}"] = ret.rolling(w).kurt()
        df[f"ret_autocorr1_{w}"] = ret.rolling(w).corr(ret.shift(1))
        neg = ret.clip(upper=0.0)
        df[f"downside_rv_{w}"] = np.sqrt((neg * neg).rolling(w).mean())

    # ---------- realized vol regime (short vs long) ----------
    rv_5 = ret.rolling(5).std()
    rv_10 = ret.rolling(10).std()
    rv_60 = ret.rolling(60).std()
    df["rv_5"] = rv_5
    df["rv_10"] = rv_10
    df["rv_60"] = rv_60

    # use your existing vol_20 if present; otherwise compute
    rv_20 = df["vol_20"] if "vol_20" in df.columns else ret.rolling(20).std()
    df["rv_5_over_20"] = rv_5 / (rv_20 + eps)
    df["rv_10_over_60"] = rv_10 / (rv_60 + eps)

    # ---------- volume / liquidity / impact (if volume provided) ----------
    if vol is not None:
        vol = vol.reindex(idx).astype(float)
        dvol = (close * vol).replace([np.inf, -np.inf], np.nan)
        df["dollar_vol"] = dvol

        # dollar-volume z-score (liquidity regime)
        dv_m = dvol.rolling(20).mean()
        dv_s = dvol.rolling(20).std()
        df["dollar_vol_z_20"] = (dvol - dv_m) / (dv_s + eps)

        # Amihud illiquidity (price impact proxy)
        # abs(ret) / dollar_vol, rolling mean; scaled for nicer magnitudes
        ill = (ret.abs() / (dvol + eps))
        df["amihud_20"] = ill.rolling(20).mean() * 1e6
        df["amihud_60"] = ill.rolling(60).mean() * 1e6

        # OBV + OBV z-score (volume pressure with direction)
        obv = (np.sign(ret.fillna(0.0)) * vol.fillna(0.0)).cumsum()
        df["obv"] = obv
        obv_m = obv.rolling(60).mean()
        obv_s = obv.rolling(60).std()
        df["obv_z_60"] = (obv - obv_m) / (obv_s + eps)

        # rolling VWAP proxy + deviation (different from your earnings-anchored VWAP)
        for w in vwap_windows:
            pv = (close * vol).rolling(w).sum()
            vv = vol.rolling(w).sum()
            rvwap = pv / (vv + eps)
            df[f"rvwap_{w}"] = rvwap
            df[f"px_vs_rvwap_{w}"] = (close / (rvwap + eps)) - 1.0

    # ---------- short-horizon market linkage (vs SPY) ----------
    if rets_panel is not None and market_ticker in rets_panel.columns:
        mret = rets_panel[market_ticker].reindex(idx).astype(float)
        df["corr_20_vs_spy"] = ret.rolling(20).corr(mret)

        # beta_20 (you already have beta_60 via anchors; this adds a shorter regime beta)
        cov20 = ret.rolling(20).cov(mret)
        var20 = mret.rolling(20).var()
        df["beta_20_vs_spy"] = cov20 / (var20 + eps)

    # (optional) “risk-off” linkage: corr to VIX changes (if present)
    if rets_panel is not None and vix_ticker in rets_panel.columns:
        vret = rets_panel[vix_ticker].reindex(idx).astype(float)
        df["corr_20_vs_vix"] = ret.rolling(20).corr(vret)

    # ---------- earnings memory (NO leakage) ----------
    if earnings_ts is not None and len(earnings_ts) > 0:
        trade_days = idx.values.astype("datetime64[ns]")
        ts = pd.to_datetime(pd.Series(earnings_ts)).values.astype("datetime64[ns]")

        pos_e = np.searchsorted(trade_days, ts, side="right") - 1
        pos_e = pos_e[(pos_e >= 0) & (pos_e < trade_days.size)]
        pos_e = np.unique(pos_e)

        if pos_e.size > 0:
            # trading-days since previous earnings day (strictly before current day)
            pos_series = pd.Series(np.nan, index=idx)
            pos_series.iloc[pos_e] = pos_e.astype(float)
            prev_pos = pos_series.ffill().shift(1)
            df["tdays_since_prev_ea"] = np.arange(len(idx), dtype=float) - prev_pos.to_numpy(dtype=float)

            # realized reaction of each earnings (after-close -> measured from event-day close)
            react1 = close.shift(-1) / close - 1.0
            react5 = close.shift(-5) / close - 1.0
            r1_e = react1.iloc[pos_e].to_numpy(dtype=float)
            r5_e = react5.iloc[pos_e].to_numpy(dtype=float)

            # trailing stats across earnings events (computed at event granularity)
            s_r1 = pd.Series(r1_e)
            mean4 = s_r1.rolling(4, min_periods=1).mean().to_numpy()
            std8 = s_r1.rolling(8, min_periods=1).std().to_numpy()
            win8 = s_r1.rolling(8, min_periods=1).apply(lambda a: np.mean(a > 0.0), raw=True).to_numpy()

            # write these into daily index as step functions that update the DAY AFTER earnings
            prev_r1 = pd.Series(np.nan, index=idx)
            prev_r5 = pd.Series(np.nan, index=idx)
            tr_mean4 = pd.Series(np.nan, index=idx)
            tr_std8 = pd.Series(np.nan, index=idx)
            tr_win8 = pd.Series(np.nan, index=idx)

            for k, p in enumerate(pos_e):
                j = p + 1  # start using this info AFTER the reaction is known (next day) -> no leakage
                if j < len(idx):
                    prev_r1.iloc[j] = r1_e[k]
                    prev_r5.iloc[j] = r5_e[k]
                    tr_mean4.iloc[j] = mean4[k]
                    tr_std8.iloc[j] = std8[k]
                    tr_win8.iloc[j] = win8[k]

            df["prev_ea_react_1d"] = prev_r1.ffill()
            df["prev_ea_react_5d"] = prev_r5.ffill()
            df["ea_trailing_mean_react1_4"] = tr_mean4.ffill()
            df["ea_trailing_std_react1_8"] = tr_std8.ffill()
            df["ea_trailing_winrate_8"] = tr_win8.ffill()

    return df


def _download_ohlc(tickers, start, end=None):
    # sanitize tickers
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("No valid string tickers found.")

    frames = {}
    for t in tickers:
        df = yf.download(
            t,
            start=start, end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            ignore_tz=True,
            group_by="column",
            threads=False,
        )
        if df is None or df.empty:
            continue

        # force tz-naive
        idx = pd.to_datetime(df.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        df = df.copy()
        df.index = idx
        df = df[~df.index.isna()].sort_index()

        frames[t] = df

    if not frames:
        raise ValueError("yfinance returned no data for any ticker.")

    # build panel
    panel = pd.concat(frames, axis=1)  # columns MultiIndex: (ticker, field)

    # pick close field
    fields = panel.columns.get_level_values(1).unique().tolist()
    use_close = "Adj Close" if "Adj Close" in fields else "Close"

    close = panel.xs(use_close, axis=1, level=1).sort_index().ffill(limit=3)
    high  = panel.xs("High", axis=1, level=1).sort_index().ffill(limit=3)
    low   = panel.xs("Low",  axis=1, level=1).sort_index().ffill(limit=3)

    return close, high, low


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def macd(series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h


def bollinger_z(series, window=20):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - ma) / (sd + 1e-12)


def rolling_beta(x_ret, y_ret, window=60):
    cov = x_ret.rolling(window).cov(y_ret)
    var = y_ret.rolling(window).var()
    return cov / (var + 1e-12)


def anchored_vwap(px: pd.Series, vol: pd.Series, anchor_mask: pd.Series) -> pd.Series:
    """AVWAP that resets on each True in anchor_mask (inclusive)."""
    px = px.astype(float)
    vol = vol.astype(float)
    anchor_mask = anchor_mask.reindex(px.index).fillna(False).astype(bool)

    # avoid divide-by-zero / NaN propagation
    v = vol.fillna(0.0)
    pv = (px.fillna(method="ffill") * v).fillna(0.0)

    seg = anchor_mask.cumsum()  # segment id increases at each anchor
    cum_pv = pv.groupby(seg).cumsum()
    cum_v = v.groupby(seg).cumsum()

    out = cum_pv / (cum_v.replace(0.0, np.nan))
    return out


def _extract_earnings_ts(ticker: str) -> np.ndarray:
    ed = yf.Ticker(ticker).earnings_dates
    if ed is None or ed.empty:
        return np.array([], dtype="datetime64[ns]")

    idx = ed.index  # tz-aware ET
    after_close = ed[idx.time >= pd.to_datetime("16:00").time()]

    return (
        after_close.index
        .tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )


def build_earnings_reaction_dataset(
    tickers: list[str],
    horizon: int = 1,
    start: str = "2010-01-01",
    end: str | None = None,
    anchor: list[str] | None = ["SPY"],
    min_hist_days: int = 120,
    ma_windows: tuple[int, int, int] = (5, 20, 60),  # <-- 3 MAs
) -> pd.DataFrame:

    anchor = anchor or []
    all_tickers = list(dict.fromkeys(tickers + anchor))

    raw = yf.download(
        all_tickers,
        start=start, end=end,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        ignore_tz=True,
    )

    # ---- prices panel ----
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(1).unique().tolist()
        use_px = "Adj Close" if "Adj Close" in fields else "Close"
        px = raw.xs(use_px, axis=1, level=1).sort_index()
        vol = raw.xs("Volume", axis=1, level=1).sort_index() if "Volume" in fields else None
    else:
        use_px = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = pd.DataFrame({all_tickers[0]: raw[use_px]}).sort_index()
        vol = pd.DataFrame({all_tickers[0]: raw["Volume"]}).sort_index() if "Volume" in raw.columns else None

    px   = raw.xs(use_px, axis=1, level=1).sort_index()
    opn  = raw.xs("Open", axis=1, level=1).sort_index()
    high = raw.xs("High", axis=1, level=1).sort_index()
    low  = raw.xs("Low", axis=1, level=1).sort_index()

    px = px.ffill(limit=3)
    if vol is not None:
        vol = vol.ffill(limit=3)

    rets = px.pct_change()
    logrets = np.log(px).diff()

    # Cache earnings timestamps per ticker (avoid double yfinance calls)
    earnings_ts = {}
    for t in tickers:
        try:
            earnings_ts[t] = _extract_earnings_ts(t)
        except Exception:
            earnings_ts[t] = np.array([], dtype="datetime64[ns]")

    # 3) Per-ticker daily features
    daily_feats = {}
    for t in tickers:
        if t not in px.columns:
            continue

        s = px[t]
        r = rets[t]
        lr = logrets[t]

        df = pd.DataFrame(index=px.index)
        df["px"] = s
        df["ret_1"] = r
        df["logret_1"] = lr
        df["ret_5"] = s.pct_change(5)
        df["vol_20"] = r.rolling(20).std()
        df["z_bb_20"] = bollinger_z(s, 20)

        df["rsi_14"] = rsi(s, 14)
        m, sig, hist = macd(s, 12, 26, 9)
        df["macd"] = m
        df["macd_signal"] = sig
        df["macd_hist"] = hist

        o = opn[t]
        h = high[t]
        l = low[t]
        c = s  # close

        rng = (h - l).replace(0, np.nan)

        # Intraday structure
        df["hl_range"] = rng / (c + 1e-12)
        df["body"] = (c - o) / (c + 1e-12)
        df["body_abs"] = (c - o).abs() / (c + 1e-12)

        df["close_loc"] = (c - l) / (rng + 1e-12)

        # Wicks
        df["upper_wick"] = (h - np.maximum(c, o)) / (c + 1e-12)
        df["lower_wick"] = (np.minimum(c, o) - l) / (c + 1e-12)

        # Gaps
        df["gap_open"] = o / c.shift(1) - 1.0

        # True Range / ATR proxy
        tr = np.maximum(
            h - l,
            np.maximum(
                (h - c.shift(1)).abs(),
                (l - c.shift(1)).abs()
            )
        )

        df["atr_14"] = tr.rolling(14).mean()
        df["atr_rel"] = df["atr_14"] / (c + 1e-12)


        df = add_regime_drift_features(
                df=df,
                o=o,
                h=h,
                l=l,
                c=c,
                r=r,
                vol_series=(vol[t] if vol is not None and t in vol.columns else None),
                anchors_rets=rets,
                anchor_list=anchor,
                add_percentiles=True,
                pct_window=252,
            )




        if "GLD" in rets.columns and "SLV" in rets.columns:
          g_ret = rets["GLD"]
          s_ret = rets["SLV"]

          df["gold_silver_ret_1"] = g_ret - s_ret
          df["gold_silver_ret_5"] = px["GLD"].pct_change(5) - px["SLV"].pct_change(5)
          df["gold_silver_corr_60"] = g_ret.rolling(60).corr(s_ret)

        # ---- volume features + AVWAP anchored to prior earnings ----
        if vol is not None and t in vol.columns:
            v = vol[t].astype(float)
            v_pct=vol.pct_change()[t].astype(float)
            df["vol"] = v
            df['vol_ret1']=v_pct
            for w in ma_windows:
                df[f"vol_ma_{w}"] = v.rolling(w).mean()
            df["vol_rel_20"] = df["vol"] / (df["vol_ma_20"] + 1e-12)


            # AVWAP anchor = most recent earnings trading day
            ts = earnings_ts.get(t)
            if ts is not None and ts.size:
                trade_days = df.index.values.astype("datetime64[ns]")
                pos = np.searchsorted(trade_days, ts, side="right") - 1
                pos = pos[(pos >= 0) & (pos < trade_days.size)]
                if pos.size:
                    anchor_mask = pd.Series(False, index=df.index)
                    anchor_mask.iloc[np.unique(pos)] = True
                    avwap = anchored_vwap(s, v, anchor_mask)
                    df["avwap_ea"] = avwap
                    df["px_vs_avwap_ea"] = (s / avwap) - 1.0

        # performance = rolling mean of daily returns
        for w in ma_windows:
            df[f"perf_ma_{w}"] = r.rolling(w).mean()

        # anchors
        for a in anchor:
            if a not in rets.columns:
                continue
            a_ret = rets[a]
            df[f"rel_ret_1_vs_{a}"] = r - a_ret
            df[f"rel_ret_5_vs_{a}"] = s.pct_change(5) - px[a].pct_change(5)
            df[f"corr_60_vs_{a}"] = r.rolling(60).corr(a_ret)
            df[f"beta_60_vs_{a}"] = rolling_beta(r, a_ret, 60)

        df["fwd_ret_h"] = s.shift(-horizon) / s - 1.0

        df = add_earnings_edge_features(
          df,
          close=s,
          ret=r,
          vol=(vol[t] if vol is not None and t in vol.columns else None),
          rets_panel=rets,
          earnings_ts=earnings_ts.get(t)  # in synthetic, pass ts from _extract_earnings_ts(t)
            )

        daily_feats[t] = df


    # 4) Earnings timestamps -> vectorized event mapping
    rows = []
    for t in tickers:
        feat_df = daily_feats.get(t)
        if feat_df is None:
            continue

        ts = earnings_ts.get(t)
        if ts is None or ts.size == 0:
            continue

        trade_days = feat_df.index.values.astype("datetime64[ns]")
        pos = np.searchsorted(trade_days, ts, side="right") - 1

        # filters
        ok = (pos >= min_hist_days) & (pos >= 0) & ((pos + horizon) < trade_days.size)
        if not ok.any():
            continue

        pos_ok = pos[ok]
        event_days = pd.to_datetime(trade_days[pos_ok])

        # label availability
        fwd = feat_df["fwd_ret_h"].iloc[pos_ok].to_numpy(dtype=float)
        ok2 = np.isfinite(fwd)


        if not ok2.any():
            continue

        pos_ok = pos_ok[ok2]
        event_days = event_days[ok2]
        ts_ok = pd.to_datetime(ts[ok][ok2])

        fwd = fwd[ok2]
        y_up = (fwd > 0).astype(int)

        X = feat_df.iloc[pos_ok].drop(columns=["fwd_ret_h"])

        out = X.copy()
        out.insert(0, "ticker", t)
        out.insert(1, "earnings_ts", ts_ok.to_numpy())
        out.insert(2, "event_day", event_days.to_numpy())
        out.insert(3, "horizon_days", horizon)
        out.insert(4, "y_up", y_up)
        out.insert(5, "fwd_ret_h", fwd)

        rows.append(out.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows, ignore_index=True)
    data = data.sort_values(["ticker", "event_day"]).reset_index(drop=True)
    return data,px


def build_synthetic_earnings_test_dataset(
    tickers: list[str],
    asof: str | pd.Timestamp | None = None,   # "assumed earnings date"
    start: str = "2010-01-01",
    anchor: list[str] | None = ["SPY"],
    min_hist_days: int = 120,
    ma_windows: tuple[int, int, int] = (5, 20, 60),
) -> pd.DataFrame:
    """
    One row per ticker, representing features computed as-of the last trading day <= asof.
    Labels are unknown, so y_up and fwd_ret_h are NaN.
    """

    anchor = anchor or []
    all_tickers = list(dict.fromkeys(tickers + anchor))

    if asof is None:
        asof_ts = pd.Timestamp.today().normalize()
    else:
        asof_ts = pd.Timestamp(asof).tz_localize(None).normalize()

    # Download ONLY up to asof (end is exclusive in yfinance; add 1 day)
    end = (asof_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        all_tickers,
        start=start, end=end,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        ignore_tz=True,
    )

    # ---- prices panel ----
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(1).unique().tolist()
        use_px = "Adj Close" if "Adj Close" in fields else "Close"
        px = raw.xs(use_px, axis=1, level=1).sort_index()
        vol = raw.xs("Volume", axis=1, level=1).sort_index() if "Volume" in fields else None
    else:
        use_px = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = pd.DataFrame({all_tickers[0]: raw[use_px]}).sort_index()
        vol = pd.DataFrame({all_tickers[0]: raw["Volume"]}).sort_index() if "Volume" in raw.columns else None

    px   = raw.xs(use_px, axis=1, level=1).sort_index()
    opn  = raw.xs("Open", axis=1, level=1).sort_index()
    high = raw.xs("High", axis=1, level=1).sort_index()
    low  = raw.xs("Low", axis=1, level=1).sort_index()

    px = px.ffill(limit=3)
    if vol is not None:
        vol = vol.ffill(limit=3)

    rets = px.pct_change()
    logrets = np.log(px).diff()

    rows = []
    for t in tickers:
        if t not in px.columns:
            continue

        s = px[t]
        r = rets[t]
        lr = logrets[t]

        df = pd.DataFrame(index=px.index)
        df["px"] = s
        df["ret_1"] = r
        df["logret_1"] = lr
        df["ret_5"] = s.pct_change(5)
        df["vol_20"] = r.rolling(20).std()
        df["z_bb_20"] = bollinger_z(s, 20)

        df["rsi_14"] = rsi(s, 14)
        m, sig, hist = macd(s, 12, 26, 9)
        df["macd"] = m
        df["macd_signal"] = sig
        df["macd_hist"] = hist


        o = opn[t]
        h = high[t]
        l = low[t]
        c = s  # close

        rng = (h - l).replace(0, np.nan)

        # Intraday structure
        df["hl_range"] = rng / (c + 1e-12)
        df["body"] = (c - o) / (c + 1e-12)
        df["body_abs"] = (c - o).abs() / (c + 1e-12)

        df["close_loc"] = (c - l) / (rng + 1e-12)

        # Wicks
        df["upper_wick"] = (h - np.maximum(c, o)) / (c + 1e-12)
        df["lower_wick"] = (np.minimum(c, o) - l) / (c + 1e-12)

        # Gaps
        df["gap_open"] = o / c.shift(1) - 1.0

        # True Range / ATR proxy
        tr = np.maximum(
            h - l,
            np.maximum(
                (h - c.shift(1)).abs(),
                (l - c.shift(1)).abs()
            )
        )

        df["atr_14"] = tr.rolling(14).mean()
        df["atr_rel"] = df["atr_14"] / (c + 1e-12)

        '''
        df = add_regime_drift_features(
              df=df,
              o=o,
              h=h,
              l=l,
              c=c,
              r=r,
              vol_series=(vol[t] if vol is not None and t in vol.columns else None),
              anchors_rets=rets,
              anchor_list=anchor,
              add_percentiles=True,
              pct_window=252,
                )
        '''

        if "GLD" in rets.columns and "SLV" in rets.columns:
          g_ret = rets["GLD"]
          s_ret = rets["SLV"]

          df["gold_silver_ret_1"] = g_ret - s_ret
          df["gold_silver_ret_5"] = px["GLD"].pct_change(5) - px["SLV"].pct_change(5)
          df["gold_silver_corr_60"] = g_ret.rolling(60).corr(s_ret)



        # volume features + AVWAP anchored to prior earnings
        if vol is not None and t in vol.columns:
            v = vol[t].astype(float)
            df["vol"] = v


            v_pct=vol.pct_change()[t].astype(float)

            df['vol_ret1']=v_pct
            for w in ma_windows:
                df[f"vol_ma_{w}"] = v.rolling(w).mean()
            df["vol_rel_20"] = df["vol"] / (df["vol_ma_20"] + 1e-12)

            try:
                ts = _extract_earnings_ts(t)
            except Exception:
                ts = np.array([], dtype="datetime64[ns]")

            if ts.size:
                trade_days = df.index.values.astype("datetime64[ns]")
                pos_e = np.searchsorted(trade_days, ts, side="right") - 1
                pos_e = pos_e[(pos_e >= 0) & (pos_e < trade_days.size)]
                if pos_e.size:
                    anchor_mask = pd.Series(False, index=df.index)
                    anchor_mask.iloc[np.unique(pos_e)] = True
                    avwap = anchored_vwap(s, v, anchor_mask)
                    df["avwap_ea"] = avwap
                    df["px_vs_avwap_ea"] = (s / avwap) - 1.0

        # performance MAs
        for w in ma_windows:
            df[f"perf_ma_{w}"] = r.rolling(w).mean()

        # anchors
        for a in anchor:
            if a not in rets.columns:
                continue
            a_ret = rets[a]
            df[f"rel_ret_1_vs_{a}"] = r - a_ret
            df[f"rel_ret_5_vs_{a}"] = s.pct_change(5) - px[a].pct_change(5)
            df[f"corr_60_vs_{a}"] = r.rolling(60).corr(a_ret)
            df[f"beta_60_vs_{a}"] = rolling_beta(r, a_ret, 60)


                # synthetic: one assumed earnings timestamp for this ticker
        earnings_ts_arr = np.array([np.datetime64(asof_ts)], dtype="datetime64[ns]")

        df = add_earnings_edge_features(
            df,
            close=s,
            ret=r,
            vol=(vol[t] if vol is not None and t in vol.columns else None),
            rets_panel=rets,
            earnings_ts=earnings_ts_arr,     # ✅ no .get()
        )


        # pick the synthetic event day = last trading day <= asof
        trade_days = df.index.values.astype("datetime64[ns]")
        asof_np = np.datetime64(asof_ts.to_datetime64())
        pos = np.searchsorted(trade_days, asof_np, side="right") - 1
        if pos < 0 or pos < min_hist_days:
            continue

        event_day = pd.to_datetime(trade_days[pos])
        X = df.iloc[[pos]].copy()

        out = X.copy()
        out.insert(0, "ticker", t)
        out.insert(1, "earnings_ts", np.datetime64(asof_ts))
        out.insert(2, "event_day", np.datetime64(event_day))
        out.insert(3, "horizon_days", np.nan)
        out.insert(4, "y_up", np.nan)
        out.insert(5, "fwd_ret_h", np.nan)

        rows.append(out.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows, ignore_index=True).sort_values(["ticker"]).reset_index(drop=True)
    return data


def get_ohlcv(
    tickers,
    start="1998-01-01",
    end=None,
    use_adj_close=True,
    ffill_limit=3,
):
    """
    Returns:
        close, high, low, open_, volume   (all pd.DataFrame)

    Each:
        index   = trading days
        columns = tickers
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

    def _extract(field):
        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(1):
                return None
            df = raw.xs(field, axis=1, level=1)
        else:
            if field not in raw.columns:
                return None
            df = raw[[field]]
            df.columns = tickers[:1]

        df = df.sort_index()
        df = df.ffill(limit=ffill_limit)
        return df

    # close / adj close
    if use_adj_close and "Adj Close" in raw.columns.get_level_values(1):
        close = _extract("Adj Close")
    else:
        close = _extract("Close")

    open_  = _extract("Open")
    high   = _extract("High")
    low    = _extract("Low")
    volume = _extract("Volume")

    return close, high, low, open_, volume


def update_ohlcv_incremental(
    tickers,
    close=None,
    high=None,
    low=None,
    open_=None,
    volume=None,
    end=None,
    use_adj_close=True,
    ffill_limit=3,
    refresh_lookback_days=10,
):
    """
    Incrementally refresh OHLCV by re-downloading only a short trailing window.

    Why this is useful:
      - first run can fetch a long history once
      - subsequent runs refresh only the most recent days (including today),
        while preserving the historical panel already in memory/storage

    Returns:
        close, high, low, open_, volume   (all pd.DataFrame)
    """

    frames = [close, high, low, open_, volume]
    existing_index = [f.index.max() for f in frames if isinstance(f, pd.DataFrame) and not f.empty]

    # If there is no prior panel, fall back to a full pull from default start.
    if not existing_index:
        return get_ohlcv(
            tickers=tickers,
            start="1998-01-01",
            end=end,
            use_adj_close=use_adj_close,
            ffill_limit=ffill_limit,
        )

    last_ts = pd.to_datetime(max(existing_index)).tz_localize(None)
    refresh_start = (last_ts - pd.Timedelta(days=refresh_lookback_days)).strftime("%Y-%m-%d")

    new_close, new_high, new_low, new_open, new_volume = get_ohlcv(
        tickers=tickers,
        start=refresh_start,
        end=end,
        use_adj_close=use_adj_close,
        ffill_limit=ffill_limit,
    )

    def _merge(old_df, new_df):
        if old_df is None or old_df.empty:
            return new_df
        if new_df is None or new_df.empty:
            return old_df

        old_df = old_df.copy()
        cutoff = pd.to_datetime(refresh_start)
        old_df = old_df.loc[old_df.index < cutoff]

        merged = pd.concat([old_df, new_df], axis=0).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]

        # keep requested ticker order and include any newly requested ticker columns
        ordered_cols = [t for t in tickers if t in merged.columns]
        return merged.reindex(columns=ordered_cols)

    close_m = _merge(close, new_close)
    high_m = _merge(high, new_high)
    low_m = _merge(low, new_low)
    open_m = _merge(open_, new_open)
    volume_m = _merge(volume, new_volume)

    return close_m, high_m, low_m, open_m, volume_m


def derive_exit_labels_first_touch_approx(
    X: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    horizon: int = 5,
    tp: float = 0.02,
    sl: float = 0.01,
    include_day1_intraday: bool = True,
    both_hit_rule: str = "sl_first",

    # --- NEW: volatility-based exits ---
    vol_mode: str = "fixed",          # "fixed" | "atr" | "close_vol"
    vol_lookback: int = 20,
    tp_vol_mult: float = 2.0,
    sl_vol_mult: float = 1.5,
    min_tp: float = 0.0,              # clamp (optional)
    min_sl: float = 0.0,
    max_tp: float = np.inf,
    max_sl: float = np.inf,
) -> pd.DataFrame:

    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels < 2:
        raise ValueError("X must have a MultiIndex like (ticker, earnings_ts).")
    if both_hit_rule not in {"sl_first", "tp_first", "skip"}:
        raise ValueError("both_hit_rule must be one of: 'sl_first', 'tp_first', 'skip'.")
    if vol_mode not in {"fixed", "atr", "close_vol"}:
        raise ValueError("vol_mode must be one of: 'fixed', 'atr', 'close_vol'.")

    tickers = X.index.get_level_values(0)
    earnings_ts = pd.to_datetime(X.index.get_level_values(1)).tz_localize(None)

    trade_days = close_df.index.values.astype("datetime64[ns]")
    ts_np = earnings_ts.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(trade_days, ts_np, side="right") - 1

    valid = (pos >= 0) & ((pos + horizon) < len(trade_days))
    event_day = pd.to_datetime(trade_days[np.clip(pos, 0, len(trade_days) - 1)])

    out = pd.DataFrame(index=X.index)
    out["event_day"] = event_day

    entry_px = []
    mfe_list = []
    mae_list = []
    exit_code_list = []
    exit_day_list = []
    tp_used_list = []
    sl_used_list = []
    vol_used_list = []

    # --- Precompute vol surfaces (fast + simple) ---
    # For ATR we need prev close; for close_vol we use rolling std of returns.
    if vol_mode == "atr":
        prev_close = close_df.shift(1)
        tr = pd.concat([
            (high_df - low_df),
            (high_df - prev_close).abs(),
            (low_df - prev_close).abs(),
        ], axis=0).groupby(level=0).max()  # (hacky) not correct because concat changes index
        # Better explicit TR:
        tr = np.maximum(high_df - low_df, np.maximum((high_df - prev_close).abs(), (low_df - prev_close).abs()))
        atr = tr.rolling(vol_lookback, min_periods=vol_lookback).mean()
        # Convert to % of price (use close as a stable denominator; entry close is even better per-trade)
        atr_pct = atr / close_df

    elif vol_mode == "close_vol":
        ret = close_df.pct_change()
        vol = ret.rolling(vol_lookback, min_periods=vol_lookback).std()
        vol_pct = vol  # already in return units (daily)

    for (tkr, ev_pos, ok) in zip(tickers, pos, valid):
        if (not ok) or (tkr not in close_df.columns):
            entry_px.append(np.nan)
            mfe_list.append(np.nan)
            mae_list.append(np.nan)
            exit_code_list.append(np.nan)
            exit_day_list.append(np.nan)
            tp_used_list.append(np.nan)
            sl_used_list.append(np.nan)
            vol_used_list.append(np.nan)
            continue

        entry = float(close_df.iloc[ev_pos][tkr])
        entry_px.append(entry)

        start = ev_pos + 1
        end = ev_pos + 1 + horizon  # exclusive

        # --- per-trade TP/SL ---
        if vol_mode == "fixed":
            tp_i, sl_i = float(tp), float(sl)
            vol_i = np.nan

        elif vol_mode == "atr":
            # use ATR% from event day (computed using prior lookback)
            vol_i = float(atr_pct.iloc[ev_pos][tkr])
            if not np.isfinite(vol_i) or vol_i <= 0:
                tp_i = sl_i = np.nan
            else:
                tp_i = np.clip(tp_vol_mult * vol_i, min_tp, max_tp)
                sl_i = np.clip(sl_vol_mult * vol_i, min_sl, max_sl)

        else:  # "close_vol"
            vol_i = float(vol_pct.iloc[ev_pos][tkr])
            if not np.isfinite(vol_i) or vol_i <= 0:
                tp_i = sl_i = np.nan
            else:
                tp_i = np.clip(tp_vol_mult * vol_i, min_tp, max_tp)
                sl_i = np.clip(sl_vol_mult * vol_i, min_sl, max_sl)

        tp_used_list.append(tp_i)
        sl_used_list.append(sl_i)
        vol_used_list.append(vol_i)

        # If we couldn't compute vol-based thresholds, mark invalid for this trade
        if not np.isfinite(tp_i) or not np.isfinite(sl_i):
            mfe_list.append(np.nan)
            mae_list.append(np.nan)
            exit_code_list.append(np.nan)
            exit_day_list.append(np.nan)
            continue

        # Diagnostics
        h_win = high_df.iloc[start:end][tkr].to_numpy(dtype=float)
        l_win = low_df.iloc[start:end][tkr].to_numpy(dtype=float)
        o_win = open_df.iloc[start:end][tkr].to_numpy(dtype=float)

        mfe = np.nanmax(h_win / entry - 1.0)
        mae = np.nanmin(np.minimum(o_win / entry - 1.0, l_win / entry - 1.0))

        mfe_list.append(mfe)
        mae_list.append(mae)

        # First-touch simulation
        exit_code = 0
        exit_day = np.nan

        for i, day_idx in enumerate(range(start, end), start=1):
            o = float(open_df.iloc[day_idx][tkr])
            o_ret = o / entry - 1.0

            if o_ret <= -sl_i:
                exit_code, exit_day = -1, i
                break
            if o_ret >= tp_i:
                exit_code, exit_day = 1, i
                break

            if (i == 1) and (not include_day1_intraday):
                continue

            h = float(high_df.iloc[day_idx][tkr])
            l = float(low_df.iloc[day_idx][tkr])

            touched_tp = (h / entry - 1.0) >= tp_i
            touched_sl = (l / entry - 1.0) <= -sl_i

            if touched_tp and not touched_sl:
                exit_code, exit_day = 1, i
                break
            if touched_sl and not touched_tp:
                exit_code, exit_day = -1, i
                break
            if touched_tp and touched_sl:
                if both_hit_rule == "sl_first":
                    exit_code, exit_day = -1, i
                elif both_hit_rule == "tp_first":
                    exit_code, exit_day = 1, i
                else:
                    exit_code, exit_day = np.nan, np.nan
                break

        exit_code_list.append(exit_code)
        exit_day_list.append(exit_day)

    out["entry_px"] = entry_px
    out["vol_used"] = vol_used_list
    out["tp_used"] = tp_used_list
    out["sl_used"] = sl_used_list
    out["MFE"] = mfe_list
    out["MAE"] = mae_list
    out["exit_code"] = exit_code_list
    out["exit_day"] = exit_day_list
    out["y_tp_first"] = np.where(valid, (out["exit_code"] == 1).astype(float), np.nan)
    return out

