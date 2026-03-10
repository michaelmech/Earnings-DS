import pandas as pd

from earnings_ds.execution_alpaca import _passes_spread_gate


def test_spread_gate_uses_live_bid_ask_when_current_spread_missing():
    row = pd.Series({
        "bid_price": 99.9,
        "ask_price": 100.1,
        "current_price": 100.0,
    })

    ok, info = _passes_spread_gate(
        row,
        spread_multiplier=0.003,
        current_price_col="current_price",
        current_spread_col="current_spread",
    )

    assert ok is True
    assert info["current_spread"] == 0.2


def test_spread_gate_can_use_explicit_tp_spread_cap_column():
    row = pd.Series({
        "bid_price": 100.0,
        "ask_price": 100.3,
        "spread_cap": 0.25,
    })

    ok, info = _passes_spread_gate(
        row,
        spread_multiplier=None,
        current_price_col="current_price",
        current_spread_col="current_spread",
        spread_cap_col="spread_cap",
    )

    assert ok is False
    assert info["reason"] == "spread_gate_blocked"
    assert info["threshold_value"] == 0.25
    assert info["current_spread"] == 0.3
