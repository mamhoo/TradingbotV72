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

# Mock MT5
mock_mt5 = MagicMock()
sys.modules['MetaTrader5'] = mock_mt5

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategies
from bb_squeeze_strategy import check_squeeze_signal

def run_backtest():
    print("Starting Evolution Backtest (Dual-Engine Mode)...")
    
    # Try GC=F first, then GLD as fallback
    try:
        df_h1 = yf.download("GC=F", period="1y", interval="1h", progress=False)
        df_m15 = yf.download("GC=F", period="60d", interval="15m", progress=False)
        df_m5 = yf.download("GC=F", period="60d", interval="5m", progress=False)
    except:
        df_h1 = pd.DataFrame()
        df_m15 = pd.DataFrame()
        df_m5 = pd.DataFrame()

    if df_h1.empty or df_m15.empty:
        print("GC=F failed, trying IAU...")
        df_h1 = yf.download("IAU", period="1y", interval="1h", progress=False)
        df_m15 = yf.download("IAU", period="60d", interval="15m", progress=False)
        df_m5 = yf.download("IAU", period="60d", interval="5m", progress=False)
    
    if df_m15.empty or df_m5.empty:
        print("Error: Could not download data.")
        return
    
    # Flatten multi-index columns if necessary
    for df in [df_h1, df_m15, df_m5]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Prepare dataframes
    for df in [df_h1, df_m15, df_m5]:
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
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
        df_m5_slice = df_m5[df_m5.index <= current_time].tail(100)
        
        if df_h1_slice.empty or df_m5_slice.empty: 
            i += 1
            continue
        
        config = {"mt5_symbol": "XAUUSD"}
        signal = check_squeeze_signal(config, 
                                     df_m15_override=df_m15_slice, 
                                     df_h1_override=df_h1_slice,
                                     df_m5_override=df_m5_slice)
        
        if signal:
            entry = signal.entry
            sl = signal.sl
            tp = signal.tp
            action = signal.action
            
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
                pnl_usd = pnl * 1.0 
                balance += pnl_usd
                
                trade_date = current_time.date().isoformat()
                daily_profits[trade_date] = daily_profits.get(trade_date, 0) + pnl_usd
                
                trades.append({
                    "time": current_time.isoformat(),
                    "action": action,
                    "entry": entry,
                    "outcome": outcome,
                    "pnl": pnl_usd,
                    "reason": signal.reason
                })
                i = exit_idx + 1
                continue
        i += 1

    # Calculate Metrics
    total_pnl = balance - initial_balance
    win_rate = len([t for t in trades if t['outcome'] == "TP"]) / len(trades) * 100 if trades else 0
    days_traded = len(daily_profits)
    avg_daily_profit = total_pnl / days_traded if days_traded else 0
    
    print(f"\n--- Evolution Results (60 Days) ---")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Avg Daily Profit: ${avg_daily_profit:.2f}")
    print(f"Final Balance: ${balance:.2f}")

    output_dir = "/home/ubuntu/backtest_output"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/evo_backtest_results.json", "w") as f:
        json.dump({
            "total_trades": len(trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_daily_profit": avg_daily_profit,
            "trades": trades
        }, f)

if __name__ == "__main__":
    run_backtest()
