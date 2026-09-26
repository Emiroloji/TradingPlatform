import pytest

from features.relative import relative_strength
from tests.conftest import assert_no_lookahead


def test_no_lookahead(bars, benchmark, cfg):
    # truncate both series together: the benchmark must not leak future bars either
    def fn(d):
        return relative_strength(d, benchmark.loc[: d.index[-1]], cfg)

    assert_no_lookahead(fn, bars, [30, 130, 300, 449])


def test_excess_return_math(bars, cfg):
    n = cfg.features.relative.periods[0]
    out = relative_strength(bars, bars, cfg)
    assert out[f"rs_{n}"].dropna().abs().max() == pytest.approx(0)  # stock vs itself = 0

    doubled = bars.copy()
    doubled["close"] = bars["close"] * 2  # same returns, different level
    assert relative_strength(doubled, bars, cfg)["relative_strength"].dropna().abs().max() == pytest.approx(0)


def test_benchmark_gap_uses_last_known_close(bars, benchmark, cfg):
    gappy = benchmark.drop(benchmark.index[200])
    out = relative_strength(bars, gappy, cfg)
    assert out.loc[bars.index[200:260], "rs_20"].notna().all()
