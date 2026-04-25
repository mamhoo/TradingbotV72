"""
run_backtest.py — Comparative Backtest for TradingbotV72
Compares the Original Strategy vs the Optimized BB Squeeze Strategy.
"""

import os
import sys
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gold_strategy import check_gold_signal
from smc_gold_strategy import check_gold_signal_combined
from bb_squeeze_strategy import check_squeeze_signal

def download_gold_data(period="1y", interval="1h"):
    log.info(f"Downloading Gold H1 data: period={period}, interval={interval}")
    # GC=F is Gold Futures on Yahoo Finance
    df_h1 = yf.download("GC=F", period=period, interval=interval, progress=False)
    if df_h1.empty:
        df_h1 = yf.download("GLD", period=period, interval=interval, progress=False)
    
    log.info(f"Downloading Gold D1 data for trend filtering...")
    df_d1 = yf.download("GC=F", period="2y", interval="1d", progress=False)
    if df_d1.empty:
        df_d1 = yf.download("GLD", period="2y", interval="1d", progress=False)

    # Flatten columns if multi-index
    for df in [df_h1, df_d1]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        # Ensure timezone-naive for comparison
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    
    return df_h1, df_d1

class FastBacktester:
    def __init__(self, df_h1, df_d1, initial_balance=10000):
        self.df_h1 = df_h1
        self.df_d1 = df_d1
        self.initial_balance = initial_balance
        self.config = {
            "mt5_symbol": "XAUUSD",
            "gold_risk_pct": 0.5,
            "gold_account_balance": initial_balance,
            "gold_lot_base": 0.01,
            "gold_max_lot": 5.0,
            "gold_max_spread_pips": 100,
            "gold_min_volume_ratio": 1.1
        }

    def run(self, strategy_func, name):
        balance = self.initial_balance
        trades = []
        active_trade = None
        
        # Warmup period
        warmup = 100
        
        for i in range(warmup, len(self.df_h1)):
            current_row = self.df_h1.iloc[i]
            current_time = self.df_h1.index[i]
            
            # Check active trade
            if active_trade:
                high = current_row['high']
                low = current_row['low']
                
                if active_trade['action'] == 'BUY':
                    if low <= active_trade['sl']:
                        # SL hit
                        pnl = (active_trade['sl'] - active_trade['entry']) * active_trade['lot'] * 100
                        balance += pnl
                        trades.append({**active_trade, 'exit_time': current_time, 'exit_price': active_trade['sl'], 'pnl': pnl, 'result': 'LOSS'})
                        active_trade = None
                    elif high >= active_trade['tp']:
                        # TP hit
                        pnl = (active_trade['tp'] - active_trade['entry']) * active_trade['lot'] * 100
                        balance += pnl
                        trades.append({**active_trade, 'exit_time': current_time, 'exit_price': active_trade['tp'], 'pnl': pnl, 'result': 'WIN'})
                        active_trade = None
                else: # SELL
                    if high >= active_trade['sl']:
                        # SL hit
                        pnl = (active_trade['entry'] - active_trade['sl']) * active_trade['lot'] * 100
                        balance += pnl
                        trades.append({**active_trade, 'exit_time': current_time, 'exit_price': active_trade['sl'], 'pnl': pnl, 'result': 'LOSS'})
                        active_trade = None
                    elif low <= active_trade['tp']:
                        # TP hit
                        pnl = (active_trade['entry'] - active_trade['tp']) * active_trade['lot'] * 100
                        balance += pnl
                        trades.append({**active_trade, 'exit_time': current_time, 'exit_price': active_trade['tp'], 'pnl': pnl, 'result': 'WIN'})
                        active_trade = None
            
            # Look for new signal if no active trade
            if not active_trade:
                # Prepare data slices
                h1_slice = self.df_h1.iloc[max(0, i-200):i+1].copy()
                m15_slice = h1_slice # Simplified
                h4_slice = h1_slice  # Simplified
                
                # Get D1 data up to current time
                d1_slice = self.df_d1[self.df_d1.index < current_time].tail(200).copy()
                
                # Update balance in config
                self.config["gold_account_balance"] = balance
                
                # Call strategy
                signal = strategy_func(
                    self.config, 
                    df_m5_override=h1_slice, 
                    df_m15_override=m15_slice, 
                    df_h1_override=h1_slice, 
                    df_h4_override=h4_slice, 
                    df_d1_override=d1_slice
                )
                
                if signal:
                    lot = getattr(signal, 'lot', getattr(signal, 'lot_or_qty', 0.01))
                    active_trade = {
                        'action': signal.action,
                        'entry': signal.entry,
                        'sl': signal.sl,
                        'tp': signal.tp,
                        'lot': lot,
                        'entry_time': current_time,
                        'reason': signal.reason
                    }
        
        # Calculate metrics
        wins = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = (len(wins) / len(trades) * 100) if trades else 0
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # Max Drawdown
        equity_curve = [self.initial_balance]
        curr = self.initial_balance
        for t in trades:
            curr += t['pnl']
            equity_curve.append(curr)
        
        peak = self.initial_balance
        max_dd = 0
        for e in equity_curve:
            if e > peak: peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd: max_dd = dd

        return {
            "strategy": name,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_dd, 2),
            "final_balance": round(balance, 2)
        }

if __name__ == "__main__":
    df_h1, df_d1 = download_gold_data(period="1y", interval="1h")
    if df_h1.empty or df_d1.empty:
        log.error("Failed to download data")
        sys.exit(1)
        
    backtester = FastBacktester(df_h1, df_d1)
    
    log.info("\nRunning Backtest for Original Strategy...")
    original_results = backtester.run(check_gold_signal, "Original")

    log.info("\nRunning Backtest for Combined SMC + Classic Strategy...")
    smc_results = backtester.run(check_gold_signal_combined, "Combined_SMC")
    
    log.info("\nRunning Backtest for Optimized BB Squeeze Strategy...")
    squeeze_results = backtester.run(check_squeeze_signal, "BBSqueeze_Optimized")
    
    results = {
        "Original": original_results,
        "Combined_SMC": smc_results,
        "BBSqueeze_Optimized": squeeze_results
    }
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS COMPARISON")
    print("="*50)
    for name, res in results.items():
        print(f"\nStrategy: {name}")
        print(f"Total Trades:  {res['total_trades']}")
        print(f"Win Rate:      {res['win_rate']}%")
        print(f"Total P&L:     ${res['total_pnl']}")
        print(f"Profit Factor: {res['profit_factor']}")
        print(f"Max Drawdown:  {res['max_drawdown']}%")
        print(f"Final Balance: ${res['final_balance']}")
    
    # Save results
    output_dir = "/home/ubuntu/backtest_output"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/comparison_data.json", "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to {output_dir}/comparison_data.json")
