from scanner.scanner_engine import ScannerEngine
from scanner.settings_store import load_settings

engine = ScannerEngine()
settings = load_settings()
result = engine.scan(universe='NIFTY 50', settings=settings, period='1y', trend_filter='All')
print(f'Scan complete: {len(result.results)} results, filtered_out={result.filtered_out}, error={result.error}')
if result.results:
    print('Top 3:')
    for r in result.results[:3]:
        print(f'  {r["ticker"]}: {r["total"]:.1f} ({r["trend_dir"]})')
