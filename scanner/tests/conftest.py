"""Shared test fixtures for the HMAxEMA scanner test suite."""

import logging
import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as _pq
import pytest

_PARQUET_LOCK = threading.Lock()


def safe_to_parquet(df, path, **kwargs):
    """Write a DataFrame to parquet, tolerating pyarrow 25 extension-type race."""
    with _PARQUET_LOCK:
        try:
            df.to_parquet(path, **kwargs)
        except pa.lib.ArrowKeyError:
            preserve = kwargs.pop("index", None)
            kwargs.pop("engine", None)
            table = pa.Table.from_pandas(
                df.reset_index(drop=not preserve), preserve_index=bool(preserve)
            )
            _pq.write_table(table, path)


@pytest.fixture(scope="session", autouse=True)
def _warm_pyarrow_extension():
    """Pre-register the pandas.period extension type once, thread-safely.

    pyarrow 25 + pandas 3 has a race condition where concurrent
    ``to_parquet`` calls register ``pandas.period`` multiple times,
    triggering ``ArrowKeyError``.  Doing it once at session start under
    a lock avoids the collision entirely.
    """
    with _PARQUET_LOCK:
        tmp = pd.DataFrame(
            {"x": [1]}, index=pd.bdate_range("2024-01-01", periods=1)
        )
        import os
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "_pyarrow_warmup.parquet")
        try:
            tmp.to_parquet(path, index=True)
        except pa.lib.ArrowKeyError:
            pass
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.fixture(scope="session", autouse=True)
def silence_daemon_thread_logging():
    """Keep daemon threads from logging into pytest's closed capture stream.

    The abortable-download tests spawn executor threads that may emit log
    records (e.g. "Chunk %d/%d done") after the test that created them has
    finished and pytest closed its handler — producing cosmetic
    "Logging error: I/O operation on closed file" noise at the end of a run.
    Routing scanner.data_fetcher records to a NullHandler keeps the suite
    output clean without affecting any assertions (no test uses caplog).
    """
    lg = logging.getLogger("scanner.data_fetcher")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False


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
