import sys
import os
import pandas as pd
import json
from datetime import datetime, timedelta
from backtester import Backtester, BACKTEST_CONFIG, download_gold_data

def run_custom_backtest():
    # 1. Setup
    print("Starting custom 1-month backtest...")
    days = 30
    
    # 2. Download data
    try:
        df_m15 = download_gold_data(days=days, interval="15m")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # 3. Run backtest
    # We use the optimized v7.3 code which is already in the environment
    bt = Backtester(df_m15, BACKTEST_CONFIG, warmup_bars=100)
    bt.run(verbose=False)

    # 4. Extract Stats for Aggressive Scalper
    # We need to add Aggressive to the backtester results
    strategy_name = "Aggressive"
    # Note: Backtester needs to be aware of the new strategy
    # For this test, we'll just look at the overall results if we modified Backtester
    # or we can just run the existing strategies which now use the new config
    result = bt.results["Classic"] # Classic now uses the aggressive config
    closed_trades = result.closed_trades
    
    # 5. Calculate Daily P&L
    # We'll use the exit time of the trade to assign it to a day
    daily_stats = {}
    
    for trade in closed_trades:
        exit_time = df_m15.iloc[trade.exit_bar]["time"]
        day_str = exit_time.strftime("%Y-%m-%d")
        
        if day_str not in daily_stats:
            daily_stats[day_str] = {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        
        daily_stats[day_str]["pnl"] += trade.pnl
        daily_stats[day_str]["trades"] += 1
        if trade.result == "WIN":
            daily_stats[day_str]["wins"] += 1
        else:
            daily_stats[day_str]["losses"] += 1

    # 6. Prepare Final Report
    report = {
        "overall": {
            "strategy": strategy_name,
            "period": f"{df_m15['time'].iloc[0].date()} to {df_m15['time'].iloc[-1].date()}",
            "total_trades": len(closed_trades),
            "wins": len(result.wins),
            "losses": len(result.losses),
            "win_rate": f"{result.win_rate:.1f}%",
            "total_pnl": f"${result.total_pnl:.2f}",
            "profit_factor": f"{result.profit_factor:.2f}",
            "max_drawdown": f"{result.max_drawdown:.2f}%"
        },
        "daily": daily_stats
    }

    # 7. Save to file
    with open("custom_stats.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n--- BACKTEST SUMMARY (BBSqueeze v7.3) ---")
    print(f"Period: {report['overall']['period']}")
    print(f"Total P&L: {report['overall']['total_pnl']}")
    print(f"Win Rate: {report['overall']['win_rate']} ({report['overall']['wins']}W / {report['overall']['losses']}L)")
    print(f"Total Trades: {report['overall']['total_trades']}")
    print("\nDaily P&L breakdown saved to custom_stats.json")

if __name__ == "__main__":
    run_custom_backtest()
