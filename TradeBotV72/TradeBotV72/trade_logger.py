"""
trade_logger.py — Trade logging with analytics support

Stores all signal data including:
  - R:R ratio
  - ATR value at entry
  - Session info
  - Score breakdown
"""

import sqlite3
import logging
from datetime import datetime

log = logging.getLogger(__name__)
DB_PATH = "trades.db"


class TradeLogger:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize database with all columns."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    market        TEXT,
                    symbol        TEXT,
                    action        TEXT,
                    entry         REAL,
                    sl            REAL,
                    tp            REAL,
                    lot_qty       REAL,
                    score         INTEGER,
                    sr_type       TEXT,
                    sr_level      REAL,
                    zone_strength INTEGER,
                    trend_1h      TEXT,
                    rsi           REAL,
                    risk_usdt     REAL,
                    reason        TEXT,
                    result        TEXT DEFAULT 'OPEN',
                    pnl_usdt      REAL DEFAULT 0,
                    rr_ratio      REAL DEFAULT 1.5,
                    atr_value     REAL DEFAULT 0,
                    session       TEXT,
                    win_streak    INTEGER DEFAULT 0,
                    loss_streak   INTEGER DEFAULT 0
                )
            """)

            # Add new columns if table exists from older version
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN rr_ratio REAL DEFAULT 1.5")
            except Exception:
                pass  # Column exists

            try:
                conn.execute("ALTER TABLE trades ADD COLUMN atr_value REAL DEFAULT 0")
            except Exception:
                pass

            try:
                conn.execute("ALTER TABLE trades ADD COLUMN session TEXT")
            except Exception:
                pass

            try:
                conn.execute("ALTER TABLE trades ADD COLUMN win_streak INTEGER DEFAULT 0")
            except Exception:
                pass

            try:
                conn.execute("ALTER TABLE trades ADD COLUMN loss_streak INTEGER DEFAULT 0")
            except Exception:
                pass

            conn.commit()

    def log(self, signal):
        """Log a new trade signal."""
        try:
            # Extract session from reason string
            session = ""
            if hasattr(signal, 'reason'):
                parts = signal.reason.split('|')
                if parts:
                    session = parts[0].strip()

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO trades
                    (timestamp, market, symbol, action, entry, sl, tp, lot_qty,
                     score, sr_type, sr_level, zone_strength, trend_1h, rsi,
                     risk_usdt, reason, rr_ratio, atr_value, session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.timestamp.isoformat(),
                    signal.market, signal.symbol, signal.action,
                    signal.entry, signal.sl, signal.tp, signal.lot_or_qty,
                    signal.score, signal.sr_type, signal.sr_level,
                    signal.zone_strength, signal.trend_1h, signal.rsi,
                    signal.risk_usdt, signal.reason,
                    getattr(signal, 'rr_ratio', 1.5),
                    getattr(signal, 'atr_value', 0),
                    session
                ))
                conn.commit()
            log.info(f"[LOG] Trade saved: {signal.symbol} {signal.action}")
        except Exception as e:
            log.error(f"[LOG] DB error: {e}")

    def update_result(self, symbol: str, result: str, pnl_usdt: float, action: str = None):
        """Update a trade with result and P&L."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Find most recent open trade for this symbol (and action if provided)
                if action:
                    row = conn.execute("""
                        SELECT id FROM trades
                        WHERE symbol = ? AND action = ? AND result = 'OPEN'
                        ORDER BY id DESC
                        LIMIT 1
                    """, (symbol, action)).fetchone()
                else:
                    row = conn.execute("""
                        SELECT id FROM trades
                        WHERE symbol = ? AND result = 'OPEN'
                        ORDER BY id DESC
                        LIMIT 1
                    """, (symbol,)).fetchone()

                if row is None:
                    log.warning(f"[LOG] No open trade found for {symbol}")
                    return

                trade_id = row[0]

                # Update by ID
                conn.execute("""
                    UPDATE trades
                    SET result = ?, pnl_usdt = ?
                    WHERE id = ?
                """, (result, pnl_usdt, trade_id))
                conn.commit()
                log.info(f"[LOG] Trade #{trade_id} updated: {symbol} {result} P&L:${pnl_usdt:.2f}")

        except Exception as e:
            log.error(f"[LOG] Update error: {e}")

    def get_summary(self, days: int = 7) -> str:
        """Get trading summary for the last N days."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT COUNT(*), SUM(pnl_usdt),
                           SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END),
                           AVG(CASE WHEN pnl_usdt > 0 THEN rr_ratio END),
                           AVG(CASE WHEN pnl_usdt < 0 THEN rr_ratio END)
                    FROM trades
                    WHERE timestamp >= datetime('now', ?)
                    AND result != 'OPEN'
                """, (f"-{days} days",)).fetchone()

            total, pnl, wins, avg_rr_win, avg_rr_loss = rows
            total = total or 0
            pnl = pnl or 0.0
            wins = wins or 0
            losses = total - wins
            wr = (wins / total * 100) if total > 0 else 0

            summary = (
                f"Last {days} days\n"
                f"━━━━━━━━━━━━━━\n"
                f"Trades  : {total}\n"
                f"Wins    : {wins}\n"
                f"Losses  : {losses}\n"
                f"WR      : {wr:.0f}%\n"
                f"P&L     : ${pnl:.2f}\n"
            )

            if avg_rr_win and avg_rr_loss:
                summary += f"Avg R:R W: {avg_rr_win:.1f} | L: {avg_rr_loss:.1f}"

            return summary

        except Exception as e:
            return f"[LOG] Summary error: {e}"

    def get_detailed_stats(self, days: int = 30) -> dict:
        """Get detailed trading statistics."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Basic stats
                basic = conn.execute("""
                    SELECT COUNT(*), SUM(pnl_usdt), AVG(pnl_usdt),
                           MAX(pnl_usdt), MIN(pnl_usdt),
                           SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END)
                    FROM trades
                    WHERE timestamp >= datetime('now', ?)
                    AND result != 'OPEN'
                """, (f"-{days} days",)).fetchone()

                total, pnl, avg_pnl, max_win, min_loss, wins = basic

                # Stats by session
                sessions = conn.execute("""
                    SELECT session, COUNT(*), SUM(pnl_usdt)
                    FROM trades
                    WHERE timestamp >= datetime('now', ?)
                    AND result != 'OPEN'
                    GROUP BY session
                """, (f"-{days} days",)).fetchall()

                # Stats by score range
                score_ranges = conn.execute("""
                    SELECT
                        CASE
                            WHEN score >= 80 THEN 'A+ (80+)'
                            WHEN score >= 65 THEN 'A (65-79)'
                            WHEN score >= 50 THEN 'B (50-64)'
                            ELSE 'C (<50)'
                        END as range,
                        COUNT(*), SUM(pnl_usdt)
                    FROM trades
                    WHERE timestamp >= datetime('now', ?)
                    AND result != 'OPEN'
                    GROUP BY range
                """, (f"-{days} days",)).fetchall()

                return {
                    "total_trades": total or 0,
                    "total_pnl": pnl or 0,
                    "avg_pnl": avg_pnl or 0,
                    "best_win": max_win or 0,
                    "worst_loss": min_loss or 0,
                    "wins": wins or 0,
                    "win_rate": (wins / (total or 1)) * 100,
                    "by_session": {s[0]: {"count": s[1], "pnl": s[2]} for s in (sessions or [])},
                    "by_score": {r[0]: {"count": r[1], "pnl": r[2]} for r in (score_ranges or [])},
                }

        except Exception as e:
            log.error(f"[STATS] Error: {e}")
            return {}
