import numpy as np
import pandas as pd

from earnings_ds.simulations import (
    simulate_earnings_bidir_vbt,
    simulate_earnings_long_vbt,
    make_event_signal_matrices,
    vectorbt_trade_returns_gapaware,
    attach_returns_to_events,
    simulate_event_returns_from_proba,
)


def _sample_ohlc(index, tickers):
    base = np.linspace(100, 120, len(index))
    close = pd.DataFrame({t: base + i for i, t in enumerate(tickers)}, index=index)
    open_ = close * 0.999
    high = close * 1.01
    low = close * 0.99
    return open_, high, low, close


def test_simulate_earnings_bidir_trade_alignment_with_preds_index():
    dates = pd.bdate_range("2024-01-02", periods=20)
    tickers = ["AAA", "BBB", "CCC"]
    open_, high, low, close = _sample_ohlc(dates, tickers)

    events = [
        ("AAA", pd.Timestamp("2024-01-03 16:00"), 0.80),
        ("BBB", pd.Timestamp("2024-01-04 16:00"), 0.75),
        ("CCC", pd.Timestamp("2024-01-05 16:00"), 0.20),
        ("AAA", pd.Timestamp("2024-01-08 16:00"), 0.30),
        ("BBB", pd.Timestamp("2024-01-09 16:00"), 0.85),
        ("CCC", pd.Timestamp("2024-01-10 16:00"), 0.15),
    ]

    idx = pd.MultiIndex.from_tuples([(t, ts) for t, ts, _ in events], names=["ticker", "earnings_ts"])
    proba = pd.Series([p for _, _, p in events], index=idx)

    pf = simulate_earnings_bidir_vbt(
        proba_last_class=proba,
        ohlcv={"open": open_, "high": high, "low": low, "close": close},
        horizon=2,
        min_proba_long=0.7,
        max_proba_short=0.35,
        weighting="equal",
        debug=True,
    )

    trades = pf.trades.records_readable.copy()
    assert not trades.empty

    expected = []
    for t, ts, p in events:
        d = ts.normalize()
        if d in dates and (p >= 0.7 or p <= 0.35):
            expected.append((d, t))

    expected_idx = pd.MultiIndex.from_tuples(expected, names=["event_day", "ticker"]).unique()
    trades_idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "event_day": pd.to_datetime(trades["Entry Timestamp"]).dt.normalize(),
                "ticker": trades["Column"].astype(str),
            }
        )
    ).unique()

    overlap = expected_idx.intersection(trades_idx)
    overlap_ratio = len(overlap) / len(expected_idx)

    # Allow small slippage/missing fills, but reject large mismatches from bad joins.
    assert overlap_ratio >= 0.80


def test_event_join_and_trade_attachment_alignment():
    dates = pd.bdate_range("2024-02-01", periods=15)
    tickers = ["AAA", "BBB"]
    open_, high, low, close = _sample_ohlc(dates, tickers)

    index = pd.MultiIndex.from_tuples(
        [
            ("AAA", pd.Timestamp("2024-02-02 16:00")),
            ("BBB", pd.Timestamp("2024-02-05 16:00")),
            ("AAA", pd.Timestamp("2024-02-06 16:00")),
            ("BBB", pd.Timestamp("2024-02-09 16:00")),
        ],
        names=["ticker", "earnings_ts"],
    )
    p_primary = pd.Series([0.8, 0.2, 0.7, 0.3], index=index)
    index_df = pd.DataFrame(
        {
            "event_day": [
                pd.Timestamp("2024-02-02"),
                pd.Timestamp("2024-02-05"),
                pd.Timestamp("2024-02-06"),
                pd.Timestamp("2024-02-09"),
            ]
        },
        index=index,
    )

    tmp_events, el, xl, es, xs = make_event_signal_matrices(
        index_df=index_df,
        px_close=close,
        p_primary=p_primary,
        horizon=2,
        side_threshold=0.5,
        debug=True,
    )

    trades = vectorbt_trade_returns_gapaware(
        open_df=open_,
        high_df=high,
        low_df=low,
        close_df=close,
        entries_long=el,
        exits_long=xl,
        entries_short=es,
        exits_short=xs,
        debug=True,
    )

    out = attach_returns_to_events(tmp_events, trades, close)

    assert len(out) == len(tmp_events)
    matched = out["trade_ret"].notna().sum()
    assert matched / len(out) >= 0.80


def test_simulate_earnings_long_trade_alignment_with_preds_index():
    dates = pd.bdate_range("2024-03-01", periods=20)
    tickers = ["AAA", "BBB", "CCC"]
    open_, high, low, close = _sample_ohlc(dates, tickers)

    events = [
        ("AAA", pd.Timestamp("2024-03-04 16:00"), 0.80),
        ("BBB", pd.Timestamp("2024-03-05 16:00"), 0.65),
        ("CCC", pd.Timestamp("2024-03-06 16:00"), 0.20),
        ("AAA", pd.Timestamp("2024-03-07 16:00"), 0.30),
        ("BBB", pd.Timestamp("2024-03-08 16:00"), 0.85),
        ("CCC", pd.Timestamp("2024-03-11 16:00"), 0.55),
    ]

    idx = pd.MultiIndex.from_tuples([(t, ts) for t, ts, _ in events], names=["ticker", "earnings_ts"])
    proba = pd.Series([p for _, _, p in events], index=idx)

    pf = simulate_earnings_long_vbt(
        proba_last_class=proba,
        ohlcv={"open": open_, "high": high, "low": low, "close": close},
        horizon=2,
        min_proba=0.5,
        weighting="equal",
        debug=True,
    )

    trades = pf.trades.records_readable.copy()
    assert not trades.empty

    expected = []
    for t, ts, p in events:
        d = ts.normalize()
        if d in dates and p >= 0.5:
            expected.append((d, t))

    expected_idx = pd.MultiIndex.from_tuples(expected, names=["event_day", "ticker"]).unique()
    trades_idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "event_day": pd.to_datetime(trades["Entry Timestamp"]).dt.normalize(),
                "ticker": trades["Column"].astype(str),
            }
        )
    ).unique()

    overlap = expected_idx.intersection(trades_idx)
    overlap_ratio = len(overlap) / len(expected_idx)

    assert overlap_ratio >= 0.80


def test_illiquidity_gate_blocks_events_above_tp_price_cap():
    dates = pd.bdate_range("2024-04-01", periods=8)
    tickers = ["AAA", "BBB"]
    open_, high, low, close = _sample_ohlc(dates, tickers)

    idx = pd.MultiIndex.from_tuples(
        [
            ("AAA", pd.Timestamp("2024-04-02 16:00")),
            ("BBB", pd.Timestamp("2024-04-03 16:00")),
        ],
        names=["ticker", "earnings_ts"],
    )
    p_primary = pd.Series([0.8, 0.8], index=idx)
    index_df = pd.DataFrame(
        {"event_day": [pd.Timestamp("2024-04-02"), pd.Timestamp("2024-04-03")]},
        index=idx,
    )

    spread_df = pd.DataFrame(0.5, index=dates, columns=tickers)
    spread_df.loc[pd.Timestamp("2024-04-02"), "AAA"] = 10.0  # should be blocked for tp=3%

    events_with_ret, *_ = simulate_event_returns_from_proba(
        index_df=index_df,
        p_primary=p_primary,
        px_open=open_,
        px_high=high,
        px_low=low,
        px_close=close,
        px_volume=None,
        horizon=2,
        side_threshold=0.5,
        tp=0.03,
        sl=0.03,
        long_only=True,
        use_smart_slippage=False,
        use_illiquidity_gate=True,
        illiquidity_spread_df=spread_df,
    )

    # AAA is blocked by illiquidity gate; BBB remains tradable.
    assert pd.isna(events_with_ret.loc[("AAA", pd.Timestamp("2024-04-02 16:00")), "trade_ret"])
    assert pd.notna(events_with_ret.loc[("BBB", pd.Timestamp("2024-04-03 16:00")), "trade_ret"])
