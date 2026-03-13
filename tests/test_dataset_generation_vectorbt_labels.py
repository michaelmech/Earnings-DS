import numpy as np
import pandas as pd

from earnings_ds.dataset_generation import derive_exit_labels_first_touch_approx


def _sample_inputs():
    dates = pd.bdate_range("2024-01-02", periods=8)
    tickers = ["AAA", "BBB"]

    close = pd.DataFrame(
        {
            "AAA": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "BBB": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    open_ = close.copy()
    high = close.copy()
    low = close.copy()

    # AAA: TP hit on day+1 intraday; BBB: SL hit on day+1 intraday.
    high.loc[pd.Timestamp("2024-01-04"), "AAA"] = 104.0
    low.loc[pd.Timestamp("2024-01-05"), "BBB"] = 96.0

    idx = pd.MultiIndex.from_tuples(
        [
            ("AAA", pd.Timestamp("2024-01-03 16:00")),
            ("BBB", pd.Timestamp("2024-01-04 16:00")),
        ],
        names=["ticker", "earnings_ts"],
    )
    X = pd.DataFrame(index=idx)
    return X, open_, high, low, close


def test_derive_exit_labels_mandatory_vectorbt_uses_first_touch_exit_days(monkeypatch):
    X, open_, high, low, close = _sample_inputs()

    captured = {}

    def _fake_vbt(*, entries_long, exits_long, tp, sl, use_smart_slippage, weighting, max_trade_size, smart_slippage_kwargs, **kwargs):
        captured["entries_long"] = entries_long.copy()
        captured["exits_long"] = exits_long.copy()
        captured["tp"] = tp
        captured["sl"] = sl
        captured["use_smart_slippage"] = use_smart_slippage
        captured["weighting"] = weighting
        captured["max_trade_size"] = max_trade_size
        captured["smart_slippage_kwargs"] = smart_slippage_kwargs
        captured["kwargs"] = kwargs

        return pd.DataFrame(
            {
                "Column": ["AAA", "BBB"],
                "Entry Timestamp": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")],
                "Exit Timestamp": [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
                "Return": [0.04, -0.04],
            }
        )

    monkeypatch.setattr("earnings_ds.simulations.vectorbt_trade_returns_gapaware", _fake_vbt)

    out = derive_exit_labels_first_touch_approx(
        X=X,
        open_df=open_,
        high_df=high,
        low_df=low,
        close_df=close,
        horizon=2,
        tp=0.03,
        sl=0.03,
        use_smart_slippage=False,
        smart_slippage_kwargs={"participation_cap": 0.1},
        weighting="conviction",
        max_trade_size=0.25,
        debug=True,
    )

    assert captured["tp"] == np.inf
    assert captured["sl"] == np.inf
    assert captured["use_smart_slippage"] is False
    assert captured["weighting"] == "conviction"
    assert captured["max_trade_size"] == 0.25
    assert captured["smart_slippage_kwargs"] == {"participation_cap": 0.1}
    assert captured["kwargs"] == {}

    assert captured["entries_long"].at[pd.Timestamp("2024-01-03"), "AAA"]
    assert captured["entries_long"].at[pd.Timestamp("2024-01-04"), "BBB"]
    assert captured["exits_long"].at[pd.Timestamp("2024-01-04"), "AAA"]
    assert captured["exits_long"].at[pd.Timestamp("2024-01-05"), "BBB"]

    assert out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "exit_code"] == 1
    assert out.loc[("BBB", pd.Timestamp("2024-01-04 16:00")), "exit_code"] == -1
    assert out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "exit_day"] == 1
    assert out.loc[("BBB", pd.Timestamp("2024-01-04 16:00")), "exit_day"] == 1
    assert out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "trade_ret"] == 0.04
    assert out.loc[("BBB", pd.Timestamp("2024-01-04 16:00")), "trade_ret"] == -0.04


def test_derive_exit_labels_mandatory_vectorbt_illiquidity_gate_blocks_event(monkeypatch):
    X, open_, high, low, close = _sample_inputs()

    spread = pd.DataFrame(0.01, index=close.index, columns=close.columns)
    spread.loc[pd.Timestamp("2024-01-03"), "AAA"] = 9.0

    captured = {}

    def _fake_vbt(*, entries_long, **kwargs):
        captured["entries_long"] = entries_long.copy()
        return pd.DataFrame(
            {
                "Column": ["BBB"],
                "Entry Timestamp": [pd.Timestamp("2024-01-04")],
                "Exit Timestamp": [pd.Timestamp("2024-01-05")],
                "Return": [-0.04],
            }
        )

    monkeypatch.setattr("earnings_ds.simulations.vectorbt_trade_returns_gapaware", _fake_vbt)

    out = derive_exit_labels_first_touch_approx(
        X=X,
        open_df=open_,
        high_df=high,
        low_df=low,
        close_df=close,
        horizon=2,
        tp=0.03,
        sl=0.03,
        use_illiquidity_gate=True,
        illiquidity_spread_df=spread,
    )

    assert not captured["entries_long"].at[pd.Timestamp("2024-01-03"), "AAA"]
    assert captured["entries_long"].at[pd.Timestamp("2024-01-04"), "BBB"]
    assert np.isnan(out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "y_tp_first"])
    assert np.isnan(out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "trade_ret"])


def test_derive_exit_labels_mandatory_vectorbt_respects_both_hit_rule_skip(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=6)
    close = pd.DataFrame({"AAA": [100.0] * len(dates)}, index=dates)
    open_ = close.copy()
    high = close.copy()
    low = close.copy()

    # On day+1, both TP and SL touched for AAA.
    high.loc[pd.Timestamp("2024-01-04"), "AAA"] = 104.0
    low.loc[pd.Timestamp("2024-01-04"), "AAA"] = 96.0

    idx = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2024-01-03 16:00"))],
        names=["ticker", "earnings_ts"],
    )
    X = pd.DataFrame(index=idx)

    def _fake_vbt(*, entries_long, exits_long, **kwargs):
        # skip rule -> no first-touch label hit, so exit should be horizon exit (2 bars)
        assert exits_long.at[pd.Timestamp("2024-01-05"), "AAA"]
        return pd.DataFrame(
            {
                "Column": ["AAA"],
                "Entry Timestamp": [pd.Timestamp("2024-01-03")],
                "Exit Timestamp": [pd.Timestamp("2024-01-05")],
                "Return": [0.0],
            }
        )

    monkeypatch.setattr("earnings_ds.simulations.vectorbt_trade_returns_gapaware", _fake_vbt)

    out = derive_exit_labels_first_touch_approx(
        X=X,
        open_df=open_,
        high_df=high,
        low_df=low,
        close_df=close,
        horizon=2,
        tp=0.03,
        sl=0.03,
        both_hit_rule="skip",
    )

    assert np.isnan(out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "exit_code"])
    assert np.isnan(out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "exit_day"])


def test_derive_exit_labels_mandatory_vectorbt_illiquidity_gate_allows_vol_override(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=8)
    close = pd.DataFrame(
        {
            "AAA": [100.0, 140.0, 80.0, 120.0, 90.0, 100.0, 100.0, 100.0],
            "BBB": [100.0] * 8,
        },
        index=dates,
    )
    open_ = close.copy()
    high = close.copy()
    low = close.copy()

    idx = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2024-01-03 16:00"))],
        names=["ticker", "earnings_ts"],
    )
    X = pd.DataFrame(index=idx)

    spread = pd.DataFrame(0.01, index=close.index, columns=close.columns)
    spread.loc[pd.Timestamp("2024-01-03"), "AAA"] = 5.0

    captured = {}

    def _fake_vbt(*, entries_long, **kwargs):
        captured["entries_long"] = entries_long.copy()
        return pd.DataFrame(
            {
                "Column": ["AAA"],
                "Entry Timestamp": [pd.Timestamp("2024-01-03")],
                "Exit Timestamp": [pd.Timestamp("2024-01-05")],
                "Return": [0.01],
            }
        )

    monkeypatch.setattr("earnings_ds.simulations.vectorbt_trade_returns_gapaware", _fake_vbt)

    out = derive_exit_labels_first_touch_approx(
        X=X,
        open_df=open_,
        high_df=high,
        low_df=low,
        close_df=close,
        horizon=2,
        tp=0.03,
        sl=0.03,
        use_illiquidity_gate=True,
        illiquidity_spread_df=spread,
    )

    assert captured["entries_long"].at[pd.Timestamp("2024-01-03"), "AAA"]
    assert not np.isnan(out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "y_tp_first"])
    assert out.loc[("AAA", pd.Timestamp("2024-01-03 16:00")), "trade_ret"] == 0.01
