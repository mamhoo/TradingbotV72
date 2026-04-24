"""
dashboard.py — Trading Bot Dashboard & Analytics
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate

DB_PATH = "trades.db"


def get_trade_stats(days=7):
    """Get trading statistics for the last N days."""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(pnl_usdt) as total_pnl,
            AVG(pnl_usdt) as avg_pnl,
            MAX(pnl_usdt) as max_win,
            MIN(pnl_usdt) as max_loss
        FROM trades
        WHERE timestamp >= datetime('now', ?)
        AND result != 'OPEN'
    """
    
    df = pd.read_sql_query(query, conn, params=(f"-{days} days",))
    conn.close()
    
    if df.empty or df.iloc[0]['total_trades'] == 0:
        return None
    
    row = df.iloc[0]
    win_rate = (row['wins'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
    
    return {
        'period_days': days,
        'total_trades': int(row['total_trades']),
        'wins': int(row['wins']),
        'losses': int(row['losses']),
        'win_rate': win_rate,
        'total_pnl': row['total_pnl'] or 0,
        'avg_pnl': row['avg_pnl'] or 0,
        'max_win': row['max_win'] or 0,
        'max_loss': row['max_loss'] or 0,
    }


def get_symbol_breakdown(days=7):
    """Get performance breakdown by symbol."""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl_usdt) as pnl
        FROM trades
        WHERE timestamp >= datetime('now', ?)
        AND result != 'OPEN'
        GROUP BY symbol
        ORDER BY pnl DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(f"-{days} days",))
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    df['win_rate'] = (df['wins'] / df['trades'] * 100).round(1)
    df['pnl'] = df['pnl'].round(2)
    return df


def get_recent_trades(limit=10):
    """Get most recent trades."""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            timestamp,
            market,
            symbol,
            action,
            entry,
            sl,
            tp,
            result,
            pnl_usdt,
            score,
            reason
        FROM trades
        ORDER BY id DESC
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def print_dashboard():
    """Print a nice dashboard to console."""
    print("=" * 70)
    print("                    TAWMHOOBOT TRADING DASHBOARD")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Overall Stats
    stats_7d = get_trade_stats(7)
    stats_30d = get_trade_stats(30)
    
    if stats_7d:
        print("📊 LAST 7 DAYS")
        print("-" * 40)
        print(f"  Total Trades:  {stats_7d['total_trades']}")
        print(f"  Win Rate:      {stats_7d['win_rate']:.1f}%")
        print(f"  Total P&L:     ${stats_7d['total_pnl']:.2f}")
        print(f"  Avg P&L:       ${stats_7d['avg_pnl']:.2f}")
        print(f"  Best Win:      ${stats_7d['max_win']:.2f}")
        print(f"  Worst Loss:    ${stats_7d['max_loss']:.2f}")
        print()
    
    if stats_30d:
        print("📊 LAST 30 DAYS")
        print("-" * 40)
        print(f"  Total Trades:  {stats_30d['total_trades']}")
        print(f"  Win Rate:      {stats_30d['win_rate']:.1f}%")
        print(f"  Total P&L:     ${stats_30d['total_pnl']:.2f}")
        print(f"  Avg P&L:       ${stats_30d['avg_pnl']:.2f}")
        print()
    
    # Symbol Breakdown
    symbol_df = get_symbol_breakdown(7)
    if not symbol_df.empty:
        print("📈 SYMBOL PERFORMANCE (7 DAYS)")
        print("-" * 40)
        print(tabulate(symbol_df, headers='keys', tablefmt='psql', showindex=False))
        print()
    
    # Recent Trades
    recent = get_recent_trades(5)
    if not recent.empty:
        print("🕐 RECENT TRADES")
        print("-" * 40)
        for _, row in recent.iterrows():
            result_icon = "✅" if row['result'] == 'WIN' else "❌" if row['result'] == 'LOSS' else "⏳"
            pnl_str = f"${row['pnl_usdt']:.2f}" if row['pnl_usdt'] != 0 else "OPEN"
            print(f"  {result_icon} {row['symbol']} {row['action']} | Score: {row['score']} | P&L: {pnl_str}")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        print_dashboard()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure trades.db exists and has data.")
