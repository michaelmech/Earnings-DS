import numpy as np
import pandas as pd

from earnings_ds.simulations import build_size_fractions


def test_build_size_fractions_equal_respects_max_trade_size():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    cols = ["AAA", "BBB", "CCC"]

    entries_long = pd.DataFrame(
        [[True, False, False], [True, True, True]],
        index=idx,
        columns=cols,
    )
    entries_short = pd.DataFrame(False, index=idx, columns=cols)

    out = build_size_fractions(
        entries_long=entries_long,
        entries_short=entries_short,
        weighting="equal",
        max_trade_size=0.25,
    )

    assert out.loc[idx[0], "AAA"] == 0.25
    assert np.isclose(out.loc[idx[1], "AAA"], 0.25)
    assert np.isclose(out.loc[idx[1], "BBB"], 0.25)
    assert np.isclose(out.loc[idx[1], "CCC"], 0.25)
