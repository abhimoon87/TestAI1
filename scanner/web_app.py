"""
HMAxEMA Stock Scanner — Web Application
Flask-based web interface for scanning Indian stocks.

Usage:
    python -m scanner.web_app

Or double-click run.bat (Windows) / run.sh (macOS/Linux)
"""

import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit

from .universes import UNIVERSES
from .data_fetcher import fetch_stock_data, fetch_index_data, resample_ohlcv
from .scoring import compute_scores
from .report import generate_html_report, save_report

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hmaxema-scanner-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")

# ── Default Settings ─────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "fast_ma_type": "HMA",
    "fast_ma_len": 20,
    "slow_ma_type": "EMA",
    "slow_ma_len": 50,
    "crossover_lookback": 4,
    "rsi_len": 14,
    "rs_length": 14,
    "vol_ma_len": 20,
    "atr_len": 14,
    "index_symbol": "NSEI",
    "vp_lookback": 200,
    "vp_rows": 30,
    "vp_width": 40,
    "adx_len": 14,
    "adx_threshold": 20.0,
    "chop_len": 14,
    "chop_threshold": 61.8,
    "slope_ma_type": "EMA",
    "slope_ma_len": 50,
    "slope_lookback": 10,
    "flat_threshold": 0.5,
    "sc_pivot_len": 3,
    "sc_bands_mult": 0.6,
    "min_score": 50.0,
    "data_period": "1y",
    "timeframe": "D",
    "trend_filter": "All",
}

def load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception:
            pass
    return settings

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main scanner page."""
    settings = load_settings()
    universes = {name: len(tickers) for name, tickers in UNIVERSES.items()}
    return render_template('index.html', settings=settings, universes=universes)

@app.route('/api/universes')
def get_universes():
    """Get available stock universes."""
    universes = {name: len(tickers) for name, tickers in UNIVERSES.items()}
    return jsonify(universes)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings."""
    settings = load_settings()
    return jsonify(settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings."""
    settings = request.json
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start a scan in background thread."""
    data = request.json
    universe_name = data.get('universe', 'NIFTY 50')
    settings = data.get('settings', load_settings())

    # Run scan in background
    thread = threading.Thread(
        target=_run_scan_thread,
        args=(universe_name, settings),
        daemon=True
    )
    thread.start()

    return jsonify({"status": "started"})

@app.route('/api/export/html', methods=['POST'])
def export_html():
    """Export results as HTML report."""
    data = request.json
    results = data.get('results', [])
    title = data.get('title', 'HMAxEMA Scanner Report')
    threshold = data.get('threshold', 50.0)

    html = generate_html_report(results, title=title, threshold=threshold)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scanner_report_{timestamp}.html"
    filepath = os.path.join(SCANNER_DIR, filename)
    save_report(html, filepath)

    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export results as CSV."""
    import csv
    import io

    data = request.json
    results = data.get('results', [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank", "Ticker", "Score", "Rating", "Price", "MA Signal", "POC",
        "Trend", "Momentum", "RSI", "MACD", "Volume", "RS", "Fundamentals",
        "Direction", "RSI Val", "ADX", "Sideways", "1M Change", "3M Change"
    ])

    for i, r in enumerate(results, 1):
        sideways_reasons = ", ".join(r.get("sideways_reasons", []))
        writer.writerow([
            i, r["ticker"], r["total"],
            r.get("combined_rating", "N/A"),
            r.get("close"), r.get("ma_bullish", False), r.get("above_poc", False),
            r["trend"], r["momentum"], r["rsi"], r["macd"],
            r["volume"], r["rel_str"], r.get("fundamentals", 0),
            r["trend_dir"], r.get("rsi_val"), r.get("adx_val"),
            "Yes" if r.get("is_sideways") else "No",
            r.get("pc1m"), r.get("pc3m")
        ])

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scanner_results_{timestamp}.csv"

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

# ── Background Scan Thread ───────────────────────────────────────────────────

def _run_scan_thread(universe_name: str, settings: dict):
    """Run scan in background thread with WebSocket progress updates."""
    try:
        tickers = UNIVERSES.get(universe_name, [])
        period = settings.get("data_period", "1y")
        timeframe = settings.get("timeframe", "D")
        trend_filter = settings.get("trend_filter", "All")

        socketio.emit('scan_status', {
            'message': f'Starting scan: {universe_name} ({len(tickers)} stocks)',
            'type': 'info'
        })
        socketio.emit('scan_status', {
            'message': f'Timeframe: {timeframe} | Period: {period} | Filter: {trend_filter}',
            'type': 'info'
        })

        # Fetch NIFTY index
        socketio.emit('scan_progress', {'percent': 0, 'text': 'Fetching NIFTY 50 index...'})
        index_df = fetch_index_data("^NSEI", period=period)
        if index_df is not None:
            socketio.emit('scan_status', {
                'message': f'NIFTY 50 index loaded ({len(index_df)} bars)',
                'type': 'success'
            })

        # Fetch and score stocks
        results = []
        total = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            progress = (i / total) * 100
            socketio.emit('scan_progress', {
                'percent': progress,
                'text': f'[{i}/{total}] {ticker}'
            })

            try:
                df = fetch_stock_data(ticker, period=period, timeframe=timeframe)
                if df is not None and not df.empty:
                    scores = compute_scores(
                        df, index_df=index_df,
                        fast_ma_type=settings["fast_ma_type"],
                        fast_ma_len=settings["fast_ma_len"],
                        slow_ma_type=settings["slow_ma_type"],
                        slow_ma_len=settings["slow_ma_len"],
                        rsi_len=settings["rsi_len"],
                        vol_ma_len=settings["vol_ma_len"],
                        atr_len=settings["atr_len"],
                        rs_length=settings["rs_length"],
                        adx_len=settings["adx_len"],
                        adx_threshold=settings["adx_threshold"],
                        chop_len=settings["chop_len"],
                        chop_threshold=settings["chop_threshold"],
                        slope_ma_type=settings["slope_ma_type"],
                        slope_ma_len=settings["slope_ma_len"],
                        slope_lookback=settings["slope_lookback"],
                        flat_threshold=settings["flat_threshold"],
                        sc_pivot_len=settings["sc_pivot_len"],
                        sc_bands_mult=settings["sc_bands_mult"],
                        vp_lookback=settings["vp_lookback"],
                        vp_rows=settings["vp_rows"],
                        vp_width=settings["vp_width"],
                        crossover_lookback=settings["crossover_lookback"],
                    )
                    if scores is not None:
                        scores["ticker"] = ticker

                        # Apply trend filter
                        if trend_filter == "Bullish Only" and scores["trend_dir"] != "Bull":
                            continue
                        elif trend_filter == "Bearish Only" and scores["trend_dir"] != "Bear":
                            continue
                        elif trend_filter == "MA + POC Only":
                            if not (scores.get("ma_bullish", False) and scores.get("above_poc", False)):
                                continue

                        results.append(scores)
                        socketio.emit('scan_result', {
                            'ticker': ticker,
                            'score': scores['total'],
                            'rating': scores.get('combined_rating', 'N/A'),
                            'trend': scores['trend_dir'],
                            'ma_bullish': scores.get('ma_bullish', False),
                            'above_poc': scores.get('above_poc', False),
                        })
            except Exception as e:
                socketio.emit('scan_status', {
                    'message': f'{ticker}: ERROR - {str(e)}',
                    'type': 'error'
                })

            # Rate limiting
            import time
            time.sleep(0.15)

        # Sort and store results
        results.sort(key=lambda x: x["total"], reverse=True)

        # Calculate stats
        passed = len([r for r in results if r["total"] >= settings.get("min_score", 50)])
        ma_poc_count = sum(1 for r in results if r.get("ma_bullish") and r.get("above_poc"))

        socketio.emit('scan_complete', {
            'results': results,
            'stats': {
                'total': len(results),
                'passed': passed,
                'ma_poc': ma_poc_count,
                'threshold': settings.get("min_score", 50),
            }
        })

    except Exception as e:
        socketio.emit('scan_status', {
            'message': f'Scan error: {str(e)}',
            'type': 'error'
        })

# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HMAxEMA Stock Scanner — Web Application")
    print("=" * 60)
    print()
    print("  Starting server at http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print()

    # Open browser
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
