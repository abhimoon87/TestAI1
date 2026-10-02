"""Streaming fast-path pagination: rows that leave the page are pruned.

The stream branch patches rows in place instead of rebuilding, so a page
change must drop the previous page's rows (and re-display pooled rows)
or the grid shows two pages stacked together with restarted ranks.
"""

from scanner.tests.ui.conftest import make_app


def _rows(n):
    return [{"ticker": f"T{i}", "close": 100.0, "total": 50.0} for i in range(n)]


def test_page_change_prunes_stale_rows_in_stream_path():
    app = make_app()
    app.page_size = 5
    app._display_results(_rows(7))  # full rebuild → page 1 (5 rows)

    assert len(app.table_column.controls) == 5

    app._change_page(1)  # stream path: prune old page + append new rows
    assert app.current_page == 1
    assert [ctl._pool_ticker for ctl in app.table_column.controls] == ["T5", "T6"]

    app._change_page(-1)  # pooled rows must come back, not just their cells
    assert app.current_page == 0
    assert [ctl._pool_ticker for ctl in app.table_column.controls] == [
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
    ]
