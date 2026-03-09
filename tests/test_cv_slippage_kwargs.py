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
