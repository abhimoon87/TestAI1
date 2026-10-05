"""setup_trace configuration invariants."""

import logging

from scanner.shared.trace import setup_trace


def test_setup_trace_silences_yfinance_logger():
    setup_trace()
    assert logging.getLogger("yfinance").getEffectiveLevel() == logging.CRITICAL
