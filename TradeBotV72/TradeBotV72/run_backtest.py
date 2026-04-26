import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import logging
import sys
from unittest.mock import MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Mock MT5 and other dependencies
mock_mt5 = MagicMock()
sys.modules['MetaTrader5'] = mock_mt5

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategies
from bb_squeeze_strategy import check_squeeze_signal
from gold_strategy import check_gold_signal

def run_backtest():
    print("Starting High-Performance Backtest (Target: $30 Daily)...")
    
    # Download 1 year of H1 data for trend and M15 for execution
    gold = yf.Ticker("GC=F")
    df_h1 = gold.history(period="1y", interval="1h")
    df_m15 = gold.history(period="60d", interval="15m") # Yahoo only allows 60d for 15m
    
    if df_m15.empty:
        print("Error: Could not download M15 data.")
        return

    # Prepare dataframes
    df_h1.columns = [c.lower() for c in df_h1.columns]
    df_m15.columns = [c.lower() for c in df_m15.columns]
    
    # Ensure timezone-naive
    if df_h1.index.tz is not None: df_h1.index = df_h1.index.tz_localize(None)
    if df_m15.index.tz is not None: df_m15.index = df_m15.index.tz_localize(None)
    
    initial_balance = 10000
    balance = initial_balance
    trades = []
    daily_profits = {}

    # Simulation loop on M15 data
    window_size = 100
    i = window_size
    while i < len(df_m15):
        current_time = df_m15.index[i]
        df_m15_slice = df_m15.iloc[i-window_size:i+1]
        df_h1_slice = df_h1[df_h1.index <= current_time].tail(100)
        
        if df_h1_slice.empty: 
            i += 1
            continue
        
        config = {"mt5_symbol": "XAUUSD"}
        signal = check_squeeze_signal(config, df_m15_override=df_m15_slice, df_h1_override=df_h1_slice)
        
        if signal:
            # Simulate trade
            entry = signal.entry
            sl = signal.sl
            tp = signal.tp
            action = signal.action
            
            # Check future bars for SL/TP
            outcome = None
            exit_idx = i
            for j in range(i+1, len(df_m15)):
                high = df_m15['high'].iloc[j]
                low = df_m15['low'].iloc[j]
                
                if action == "BUY":
                    if low <= sl: outcome = "SL"; exit_idx = j; break
                    if high >= tp: outcome = "TP"; exit_idx = j; break
                else:
                    if high >= sl: outcome = "SL"; exit_idx = j; break
                    if low <= tp: outcome = "TP"; exit_idx = j; break
            
            if outcome:
                pnl = (tp - entry) if outcome == "TP" else (sl - entry)
                if action == "SELL": pnl = -pnl
                
                # 0.01 lot = $1 per $1 move
                pnl_usd = pnl * 1.0 
                balance += pnl_usd
                
                trade_date = current_time.date().isoformat()
                daily_profits[trade_date] = daily_profits.get(trade_date, 0) + pnl_usd
                
                trades.append({
                    "time": current_time.isoformat(),
                    "action": action,
                    "entry": entry,
                    "outcome": outcome,
                    "pnl": pnl_usd
                })
                
                # Skip to end of trade to avoid multiple entries
                i = exit_idx + 1
                continue
        
        i += 1

    # Calculate Metrics
    total_pnl = balance - initial_balance
    win_rate = len([t for t in trades if t['outcome'] == "TP"]) / len(trades) * 100 if trades else 0
    days_traded = len(daily_profits)
    avg_daily_profit = total_pnl / days_traded if days_traded else 0
    
    print(f"\n--- Backtest Results (60 Days M15) ---")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Avg Daily Profit: ${avg_daily_profit:.2f}")
    print(f"Max Daily Profit: ${max(daily_profits.values()) if daily_profits else 0:.2f}")
    print(f"Final Balance: ${balance:.2f}")

    # Save results
    output_dir = "/home/ubuntu/backtest_output"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/hp_backtest_results.json", "w") as f:
        json.dump({
            "total_trades": len(trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_daily_profit": avg_daily_profit,
            "trades": trades
        }, f)

if __name__ == "__main__":
    run_backtest()
