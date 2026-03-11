import numpy as np
import pandas as pd

from earnings_ds.meta_labeling import size_from_run_primary_out


def test_size_from_run_primary_out_accepts_proba_alias_of_score():
    out = pd.DataFrame(
        {
            "size": [0.2, -0.1, 0.0],
            "is_tradable": [True, True, True],
        },
        index=["AAA", "BBB", "CCC"],
    )

    w_proba = size_from_run_primary_out(out, weighting_scheme="proba")
    w_score = size_from_run_primary_out(out, weighting_scheme="score")

    assert np.allclose(w_proba.values, w_score.values)
    assert np.isclose(w_proba.abs().sum(), 1.0)


def test_size_from_run_primary_out_rejects_unknown_weighting_scheme():
    out = pd.DataFrame({"size": [0.2]}, index=["AAA"])

    try:
        size_from_run_primary_out(out, weighting_scheme="unknown")
    except ValueError as exc:
        assert "weighting_scheme" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown weighting_scheme")
