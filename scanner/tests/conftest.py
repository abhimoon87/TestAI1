"""Shared test fixtures for the HMAxEMA scanner test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Seeded RNG for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_ohlcv(rng):
    """Generate a realistic synthetic OHLCV DataFrame (200 bars, daily-like).

    The close series is a random walk with drift to ensure:
      - Enough data for all indicators (min ~100 bars)
      - Some trending periods for crossover detection
      - Positive volume values
    """
    n = 200
    # Random walk with slight upward drift
    returns = rng.randn(n) * 0.02 + 0.001
    close = 500 * np.exp(np.cumsum(returns))

    # Build OHLV from close
    noise = rng.randn(n) * 0.005
    high = close * (1 + np.abs(noise))
    low = close * (1 - np.abs(noise))
    open_ = close * (1 + rng.randn(n) * 0.003)
    volume = (rng.rand(n) * 1_000_000 + 500_000).astype(int)

    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


@pytest.fixture
def trending_up_ohlcv():
    """OHLCV with a clear uptrend (close always rising).

    Useful for testing that bullish signals fire correctly.
    """
    n = 200
    close = np.linspace(100, 200, n)
    high = close * 1.02
    low = close * 0.98
    open_ = close * 1.001
    volume = np.full(n, 1_000_000.0)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def trending_down_ohlcv():
    """OHLCV with a clear downtrend (close always falling)."""
    n = 200
    close = np.linspace(200, 100, n)
    high = close * 1.02
    low = close * 0.98
    open_ = close * 1.001
    volume = np.full(n, 1_000_000.0)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def crossover_ohlcv():
    """OHLCV engineered to produce a bullish MA crossover at a known bar.

    The close series flatlines below the slow MA, then jumps above it
    to trigger a crossover at bar 150 (within lookback=20 from the end at 200).
    """
    n = 200
    # Slow decline then sharp rise
    close = np.concatenate([
        np.linspace(120, 90, 150),   # bars 0-149: declining
        np.linspace(90, 130, 50),    # bars 150-199: rising sharply
    ])
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    volume = np.full(n, 1_000_000.0)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def short_ohlcv():
    """Very short OHLCV (30 bars) — below minimum for most indicators."""
    n = 30
    close = np.linspace(100, 110, n)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    volume = np.full(n, 1_000_000.0)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def flat_ohlcv():
    """Sideways / flat OHLCV (constant close) — tests edge cases."""
    n = 200
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    open_ = np.full(n, 100.0)
    volume = np.full(n, 1_000_000.0)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
