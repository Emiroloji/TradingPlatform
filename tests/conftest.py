import numpy as np
import pandas as pd
import pytest

from config import load_config


@pytest.fixture
def cfg():
    return load_config()


def make_bars(n: int = 450, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Random-walk OHLCV for one ticker, indexed by date."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.lognormal(13, 0.4, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.bdate_range(start, periods=n, name="date"),
    )


@pytest.fixture
def bars():
    return make_bars()


@pytest.fixture
def benchmark():
    return make_bars(seed=1)


def assert_no_lookahead(fn, data: pd.DataFrame, cut_points: list[int]) -> None:
    """fn(data)[t] must equal fn(data[:t+1])[t]: future bars may not change past values."""
    full = fn(data)
    for t in cut_points:
        partial = fn(data.iloc[: t + 1])
        pd.testing.assert_series_equal(
            _row(partial, -1), _row(full, t), check_names=False, rtol=1e-9, atol=1e-9
        )


def _row(result, i: int) -> pd.Series:
    if isinstance(result, pd.DataFrame):
        return result.iloc[i]
    return pd.Series([result.iloc[i]])
