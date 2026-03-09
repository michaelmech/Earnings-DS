import numpy as np
import pandas as pd

from earnings_ds.simulations import calculate_agk_spread_proxy


def test_calculate_agk_spread_proxy_uses_close_df():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    cols = ["AAA", "BBB"]

    open_df = pd.DataFrame(100.0, index=idx, columns=cols)
    high_df = pd.DataFrame([[101.0, 102.0], [103.0, 104.0]], index=idx, columns=cols)
    low_df = pd.DataFrame([[99.0, 98.0], [100.0, 101.0]], index=idx, columns=cols)
    close_df = pd.DataFrame([[100.0, 100.0], [102.0, 103.0]], index=idx, columns=cols)

    out = calculate_agk_spread_proxy(open_df, high_df, low_df, close_df)

    hl_rel = 2.0 * (high_df - low_df) / (high_df + low_df).replace(0.0, np.nan)
    expected = (hl_rel * close_df).abs().clip(lower=0.0, upper=100.0).fillna(0.0)

    pd.testing.assert_frame_equal(out, expected)
