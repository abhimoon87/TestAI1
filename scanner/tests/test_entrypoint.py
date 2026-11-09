"""Tests for the single-instance lock in scanner.__main__."""

import pytest

from scanner.__main__ import _acquire_single_instance_lock


def test_second_instance_rejected_while_first_holds_lock():
    first = _acquire_single_instance_lock(port=0)  # port 0: OS picks a free port
    try:
        with pytest.raises(SystemExit, match="already running"):
            _acquire_single_instance_lock(port=first.getsockname()[1])
    finally:
        first.close()


def test_lock_is_freed_on_close_no_stale_lock():
    # The OS releases the bind when the socket closes (crash or clean exit),
    # so a restart must never hit a stale "already running".
    sock = _acquire_single_instance_lock(port=0)
    port = sock.getsockname()[1]
    sock.close()
    second = _acquire_single_instance_lock(port=port)
    second.close()
