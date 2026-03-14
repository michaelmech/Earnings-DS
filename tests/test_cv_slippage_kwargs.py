import numpy as np
import pandas as pd

from earnings_ds import cv


def _toy_inputs():
    idx = pd.MultiIndex.from_arrays(
        [
            ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-08",
                    "2024-01-15",
                    "2024-01-15",
                ]
            ),
        ],
        names=["ticker", "earnings_ts"],
    )
    X = pd.DataFrame({"f1": np.arange(len(idx), dtype=float)}, index=idx)
    y = pd.Series([0, 1, 0, 1, 0, 1], index=idx)

    ds = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "earnings_ts": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-08",
                    "2024-01-15",
                    "2024-01-15",
                ]
            ),
            "event_day": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-08",
                    "2024-01-15",
                    "2024-01-15",
                ]
            ),
        }
    )

    px_index = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
    cols = ["AAA", "BBB"]
    px = pd.DataFrame(100.0, index=px_index, columns=cols)

    return X, y, ds, px, cols


def test_meta_cvs_passes_min_max_slippage(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()
    seen = {}

    def fake_run_primary_plus_meta(*args, **kwargs):
        seen["smart_slippage_kwargs"] = kwargs.get("smart_slippage_kwargs")
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "cvs", lambda *args, **kwargs: np.array([0.5, 0.6]))

    out = cv.meta_cvs(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        min_slippage=0.001,
        max_slippage=0.01,
        smart_slippage_kwargs={"base_spread": 0.0002},
    )

    assert np.allclose(out, np.array([0.5, 0.6]))
    assert seen["smart_slippage_kwargs"] == {
        "base_spread": 0.0002,
        "min_slippage": 0.001,
        "max_slippage": 0.01,
    }


def test_meta_cvs_composite_passes_min_max_slippage(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()
    seen = {}

    def fake_run_primary_plus_meta(*args, **kwargs):
        seen["smart_slippage_kwargs"] = kwargs.get("smart_slippage_kwargs")
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.6]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.8]))

    out = cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        min_slippage=0.002,
        max_slippage=0.02,
        smart_slippage_kwargs={"impact_k": 0.05},
    )

    assert out == 0.7
    assert seen["smart_slippage_kwargs"] == {
        "impact_k": 0.05,
        "min_slippage": 0.002,
        "max_slippage": 0.02,
    }


def test_meta_cvs_passes_illiquidity_gate_kwargs(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()
    seen = {}
    spread_df = px * 0.0001

    def fake_run_primary_plus_meta(*args, **kwargs):
        seen["use_illiquidity_gate"] = kwargs.get("use_illiquidity_gate")
        seen["illiquidity_spread_df"] = kwargs.get("illiquidity_spread_df")
        seen["illiquidity_spread_kwargs"] = kwargs.get("illiquidity_spread_kwargs")
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "cvs", lambda *args, **kwargs: np.array([0.5, 0.6]))

    cv.meta_cvs(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        use_illiquidity_gate=True,
        illiquidity_spread_df=spread_df,
        illiquidity_spread_kwargs={"window": 20},
    )

    assert seen["use_illiquidity_gate"] is True
    assert seen["illiquidity_spread_df"] is spread_df
    assert seen["illiquidity_spread_kwargs"] == {"window": 20}


def test_meta_cvs_passes_threshold_and_trade_size(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()
    seen = {}

    def fake_run_primary_plus_meta(*args, **kwargs):
        seen["primary_threshold"] = kwargs.get("primary_threshold")
        seen["meta_threshold"] = kwargs.get("meta_threshold")
        seen["max_trade_size"] = kwargs.get("max_trade_size")
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "cvs", lambda *args, **kwargs: np.array([0.5, 0.6]))

    cv.meta_cvs(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        primary_threshold=0.61,
        meta_threshold=0.72,
        max_trade_size=0.33,
    )

    assert seen["primary_threshold"] == 0.61
    assert seen["meta_threshold"] == 0.72
    assert seen["max_trade_size"] == 0.33


def test_meta_cvs_composite_passes_illiquidity_gate_kwargs(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()
    seen = {}
    spread_df = px * 0.0002

    def fake_run_primary_plus_meta(*args, **kwargs):
        seen["use_illiquidity_gate"] = kwargs.get("use_illiquidity_gate")
        seen["illiquidity_spread_df"] = kwargs.get("illiquidity_spread_df")
        seen["illiquidity_spread_kwargs"] = kwargs.get("illiquidity_spread_kwargs")
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.6]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.8]))

    cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        use_illiquidity_gate=True,
        illiquidity_spread_df=spread_df,
        illiquidity_spread_kwargs={"window": 10},
    )

    assert seen["use_illiquidity_gate"] is True
    assert seen["illiquidity_spread_df"] is spread_df
    assert seen["illiquidity_spread_kwargs"] == {"window": 10}


def test_meta_cvs_composite_recall_floor_penalizes_when_below(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.4]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.9]))

    out = cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        use_primary_recall_floor=True,
        primary_recall_floor=0.5,
    )

    assert out == -0.1


def test_meta_cvs_composite_recall_floor_keeps_weighted_score_when_met(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.6]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.8]))

    out = cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        use_primary_recall_floor=True,
        primary_recall_floor=0.5,
        recall_weight=0.25,
    )

    assert out == 0.25 * 0.6 + 0.75 * 0.8


def test_meta_cvs_primary_prints_distribution(monkeypatch, capsys):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)

    def fake_cvs(*args, **kwargs):
        label = kwargs.get("label")
        if label == "Primary CV":
            return np.array([0.2, 0.4, 0.6])
        return np.array([0.5, 0.6])

    monkeypatch.setattr(cv, "cvs", fake_cvs)

    cv.meta_cvs(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        primary_cvs=True,
    )

    out = capsys.readouterr().out
    assert "Primary CV average_precision distribution:" in out
    assert "Primary CV average_precision fold scores:" in out


def test_meta_cvs_composite_uses_weighted_average(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.6]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.8]))

    out = cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        recall_weight=0.25,
    )

    assert out == 0.25 * 0.6 + 0.75 * 0.8


def test_meta_cvs_composite_returns_primary_meta_tuple_when_requested(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.61]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.83]))

    out = cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        return_primary_meta_scores=True,
    )

    assert out == (0.61, 0.83)


def test_meta_cvs_composite_component_scores_take_precedence_over_tuple(monkeypatch):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.55]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.75]))

    out = cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        return_component_scores=True,
        return_primary_meta_scores=True,
    )

    assert isinstance(out, dict)
    assert out["primary_score"] == 0.55
    assert out["meta_score"] == 0.75


def test_meta_cvs_composite_prints_adjusted_distributions(monkeypatch, capsys):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "_cv_recall_skill", lambda *args, **kwargs: np.array([0.5, 0.6]))
    monkeypatch.setattr(cv, "_cv_average_precision_skill", lambda *args, **kwargs: np.array([0.7, 0.8]))

    cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        adjust_for_imbalance=True,
    )

    out = capsys.readouterr().out
    assert "Primary recall skill (adjusted) CV distribution:" in out
    assert "Meta AP skill (adjusted) CV distribution:" in out


def test_meta_cvs_composite_prints_unadjusted_recall_distribution(monkeypatch, capsys):
    X, y, ds, px, tickers = _toy_inputs()

    def fake_run_primary_plus_meta(*args, **kwargs):
        return X, y

    monkeypatch.setattr("earnings_ds.meta_labeling.run_primary_plus_meta", fake_run_primary_plus_meta)
    monkeypatch.setattr(cv, "cross_val_score", lambda *args, **kwargs: np.array([0.4, 0.5]))
    monkeypatch.setattr(cv, "cvs", lambda *args, **kwargs: np.array([0.7, 0.8]))

    cv.meta_cvs_composite(
        X,
        y,
        ds,
        close=px,
        high=px,
        low=px,
        open_=px,
        earnings_tickers=tickers,
        adjust_for_imbalance=False,
    )

    out = capsys.readouterr().out
    assert "Primary recall CV distribution:" in out
    assert "Primary recall CV fold scores:" in out
