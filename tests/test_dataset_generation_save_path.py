import pandas as pd

from earnings_ds.dataset_generation import build_synthetic_earnings_test_dataset


def test_build_synthetic_earnings_test_dataset_saves_csv_with_asof_date(monkeypatch, tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=5)
    ticker = "AAA"

    raw = pd.DataFrame(
        {
            (ticker, "Open"): [99, 100, 101, 102, 103],
            (ticker, "High"): [101, 102, 103, 104, 105],
            (ticker, "Low"): [98, 99, 100, 101, 102],
            (ticker, "Close"): [100, 101, 102, 103, 104],
            (ticker, "Adj Close"): [100, 101, 102, 103, 104],
            (ticker, "Volume"): [1_000, 1_200, 1_100, 1_300, 1_400],
        },
        index=dates,
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)

    monkeypatch.setattr("earnings_ds.dataset_generation.yf.download", lambda *args, **kwargs: raw)
    monkeypatch.setattr("earnings_ds.dataset_generation._extract_earnings_ts", lambda *_: pd.Series([], dtype="datetime64[ns]").to_numpy())

    asof = "2024-01-05 15:30:00"
    out = build_synthetic_earnings_test_dataset(
        tickers=[ticker],
        asof=asof,
        start="2024-01-01",
        anchor=[],
        min_hist_days=1,
        save_path=tmp_path,
    )

    assert not out.empty
    saved_file = tmp_path / "2024-01-05.csv"
    assert saved_file.exists()

    saved = pd.read_csv(saved_file)
    assert len(saved) == len(out)
    assert "ticker" in saved.columns
