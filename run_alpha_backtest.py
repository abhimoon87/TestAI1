from scanner.backtest import BacktestEngine
from scanner.universes import UNIVERSES
from scanner.settings_store import DEFAULT_SETTINGS

# Get Alpha 50 tickers
alpha50 = UNIVERSES['NIFTY ALPHA 50']
print(f'Running backtest on {len(alpha50)} Alpha 50 stocks...')

engine = BacktestEngine(DEFAULT_SETTINGS)
engine.load_data(alpha50, period='3y')
results = engine.run()

print(f'Total Trades: {results["total_trades"]}')
print(f'Win Rate: {results["win_rate"]:.1f}%')
print(f'Total PnL: {results["total_pnl"]:,.0f}')
print(f'Total Return: {results["total_return_pct"]:.1f}%')
print(f'Profit Factor: {results["profit_factor"]:.2f}')
print(f'Max Drawdown: {results["max_drawdown_pct"]:.1f}%')
print(f'Avg Win: {results["avg_win_pct"]:.1f}%')
print(f'Avg Loss: {results["avg_loss_pct"]:.1f}%')