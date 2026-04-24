"""
main.py — SUPER TRADER v7.2 Entry Point

FIXES from v6.0:
  [CRITICAL] P&L sync loop rewritten — matches closed ticket to signal by ticket ID,
             no more double close_trade() calls or symbol-set-difference guessing
  [CRITICAL] _known_open_tickets now populated at startup from existing MT5 positions,
             so pre-existing open trades are tracked correctly across restarts
  [CRITICAL] Removed duplicate UTC 21-22 dead-hour skip — session_config.py already
             handles all time filtering; double-filtering was causing missed Sydney/Tokyo signals
  [FIX]      Signal now stores ticket so sync can match cleanly
"""

import logging
import sys
import schedule
import time
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from config import CONFIG
from gold_strategy import (
    check_gold_signal, execute_gold_trade,
    calculate_trailing_stop, get_mt5_ohlcv,
    register_sl_hit,
)

from smc_gold_strategy import check_gold_signal_combined

from crypto_strategy import check_crypto_signal, execute_crypto_trade, get_exchange
from risk_manager import RiskManager
from notifier import TelegramNotifier
from trade_logger import TradeLogger
from indicators import atr

# ── Logging ──────────────────────────────────────────────────────────────────
log_handlers = [logging.FileHandler("bot.log", encoding="utf-8")]
try:
    utf8_stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                       buffering=1, closefd=False)
    log_handlers.append(logging.StreamHandler(utf8_stream))
except Exception:
    log_handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
log = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
risk = RiskManager(CONFIG)
notifier = TelegramNotifier(CONFIG)
logger = TradeLogger()
exchange = None

WARMUP_COMPLETE = False
STARTUP_TIME = None
MANAGED_POSITIONS: List[dict] = []

# Maps MT5 ticket → Signal object so we can match on close
# [FIX] was just a set of tickets; now maps ticket → signal for clean P&L attribution
_ticket_to_signal: dict = {}


# ── MT5 Connection ────────────────────────────────────────────────────────────

def auto_detect_symbol(base: str = "XAUUSD") -> str:
    candidates = [base, base + "m", base + ".", base + "+", "XAU/USD", "GOLD", "XAUUSDm"]
    for sym in candidates:
        info = mt5.symbol_info(sym)
        if info is not None and info.visible:
            log.info("[MT5] Symbol detected: %s", sym)
            return sym
    log.warning("[MT5] Could not detect symbol — using: %s", base)
    return base


def init_mt5() -> bool:
    if not mt5.initialize():
        log.error("[MT5] Initialize failed: %s", mt5.last_error())
        return False
    authorized = mt5.login(
        CONFIG["mt5_login"],
        password=CONFIG["mt5_password"],
        server=CONFIG["mt5_server"],
    )
    if not authorized:
        log.error("[MT5] Login failed: %s", mt5.last_error())
        return False
    info = mt5.account_info()
    if info:
        log.info("[MT5] Connected: %s | Balance: $%.2f | Equity: $%.2f",
                 info.name, info.balance, info.equity)
        CONFIG["gold_account_balance"] = float(info.balance)
    detected = auto_detect_symbol(CONFIG["mt5_symbol"])
    CONFIG["mt5_symbol"] = detected
    mt5.symbol_select(CONFIG["mt5_symbol"], True)
    return True


def ensure_mt5_connected() -> bool:
    try:
        if mt5.account_info() is not None:
            return True
    except Exception:
        pass
    log.warning("[MT5] Reconnecting...")
    mt5.shutdown()
    time.sleep(2)
    return init_mt5()


# ── P&L from actual MT5 deal history ─────────────────────────────────────────

def get_closed_pnl_for_ticket(ticket: int) -> float:
    """
    Retrieve actual P&L from MT5 deal history for a specific position ticket.
    Searches last 48h of deals to handle weekend/holiday gaps.
    """
    try:
        now = datetime.now()
        from_time = now - timedelta(hours=48)
        deals = mt5.history_deals_get(from_time, now)
        if deals is None:
            return 0.0
        total_pnl = sum(
            d.profit
            for d in deals
            if d.position_id == ticket
            and d.entry == mt5.DEAL_ENTRY_OUT
            and d.magic == CONFIG["mt5_magic"]
        )
        return total_pnl
    except Exception as e:
        log.error("[MT5] P&L fetch error for ticket %d: %s", ticket, e)
        return 0.0


# ── Position Sync (REWRITTEN) ─────────────────────────────────────────────────

def sync_open_positions():
    """
    [FIX v6.1] Completely rewritten sync logic:
    - Iterates over closed TICKETS (not symbol set difference)
    - Each closed ticket is matched to its Signal via _ticket_to_signal dict
    - risk.close_trade() called exactly once per closed position
    - No duplicate processing, no symbol guessing
    """
    global _ticket_to_signal, MANAGED_POSITIONS

    if not ensure_mt5_connected():
        return

    positions = mt5.positions_get(magic=CONFIG["mt5_magic"])
    if positions is None:
        positions = []

    current_tickets = {p.ticket: p for p in positions}

    # ── Detect closed tickets ─────────────────────────────────────────────────
    known_tickets = set(_ticket_to_signal.keys())
    closed_tickets = known_tickets - set(current_tickets.keys())

    for ticket in closed_tickets:
        signal = _ticket_to_signal.pop(ticket, None)
        if signal is None:
            log.warning("[SYNC] Closed ticket %d has no matching signal — skip", ticket)
            continue

        real_pnl = get_closed_pnl_for_ticket(ticket)
        result_word = "WIN" if real_pnl > 0 else "LOSS"

        risk.close_trade(signal.symbol, real_pnl, signal.action)
        logger.update_result(signal.symbol, result_word, real_pnl, signal.action)

        log.info("[SYNC] %s %s closed | %s | P&L: $%.2f (ticket %d)",
                 signal.symbol, signal.action, result_word, real_pnl, ticket)
        notifier.send_plain(
            f"[{result_word}] {signal.symbol} {signal.action} closed\n"
            f"P&L: ${real_pnl:.2f} | Score was: {signal.score}"
        )

        if real_pnl <= 0:
            register_sl_hit(signal.symbol)

    # ── Build managed positions list ──────────────────────────────────────────
    MANAGED_POSITIONS = [
        {
            "ticket":        p.ticket,
            "symbol":        p.symbol,
            "action":        "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
            "entry_price":   p.price_open,
            "current_price": p.price_current,
            "sl":            p.sl,
            "tp":            p.tp,
            "volume":        p.volume,
            "profit":        p.profit,
        }
        for p in positions
    ]

    if positions:
        log.info("[SYNC] %d open position(s) | tracked tickets: %d",
                 len(positions), len(_ticket_to_signal))


# ── [FIX] Populate _ticket_to_signal at startup from existing positions ───────

def load_existing_positions():
    """
    [FIX v6.1] On startup, register any already-open positions so sync
    can detect their close correctly — even after a bot restart.
    Creates a minimal placeholder Signal for each existing position.
    """
    global _ticket_to_signal

    positions = mt5.positions_get(magic=CONFIG["mt5_magic"])
    if not positions:
        return

    # Import here to avoid circular at module level
    from signal_model import Signal
    from datetime import datetime as dt

    for p in positions:
        action = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
        # Create a minimal signal stub so close tracking works
        stub = Signal(
            market="GOLD",
            symbol=p.symbol,
            action=action,
            entry=p.price_open,
            sl=p.sl,
            tp=p.tp,
            lot_or_qty=p.volume,
            score=0,
            reason="pre-existing position on restart",
            sr_level=p.price_open,
            sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
            zone_strength=0,
            trend_1h="UNKNOWN",
            rsi=50.0,
            risk_usdt=0.0,
            timestamp=dt.now(timezone.utc),
        )
        _ticket_to_signal[p.ticket] = stub
        log.info("[SYNC] Registered pre-existing position: %s %s ticket=%d",
                 p.symbol, action, p.ticket)

        # Also register in risk manager so max_open_trades is respected
        if not any(t.symbol == p.symbol and t.action == action
                   for t in risk.open_trades):
            risk.open_trades.append(stub)
            log.info("[RISK] Pre-existing trade added to open_trades: %s %s",
                     p.symbol, action)


# ── Position Management (Trailing Stop) ───────────────────────────────────────

def manage_positions():
    if not MANAGED_POSITIONS:
        return
    for pos in MANAGED_POSITIONS:
        try:
            symbol      = pos["symbol"]
            ticket      = pos["ticket"]
            action      = pos["action"]
            entry       = pos["entry_price"]
            current     = pos["current_price"]
            current_sl  = pos["sl"]

            df_m5 = get_mt5_ohlcv(symbol, "M5", 100)
            if df_m5 is None:
                continue

            current_atr_val = atr(df_m5, 14).iloc[-1]
            new_trailing = calculate_trailing_stop(
                action, current, entry, current_atr_val,
                trail_mult=1.5, current_sl=current_sl,
            )

            # Only update if SL improves meaningfully (>0.5 pts)
            # Also ensure breakeven lock if price moves > 1.5 ATR in profit
            if action == "BUY":
                if current > entry + (current_atr_val * 1.5):
                    new_trailing = max(new_trailing, entry + 0.5)
                if new_trailing <= current_sl + 0.5:
                    continue
            if action == "SELL":
                if current < entry - (current_atr_val * 1.5):
                    new_trailing = min(new_trailing, entry - 0.5)
                if new_trailing >= current_sl - 0.5:
                    continue

            result = mt5.order_send({
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "position": ticket,
                "sl":       round(new_trailing, 2),
                "tp":       pos["tp"],
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info("[TRAIL] %s %s: SL moved %.2f → %.2f",
                         symbol, action, current_sl, new_trailing)
        except Exception as e:
            log.error("[MANAGE] Error: %s", e)


# ── Gold Trading ──────────────────────────────────────────────────────────────

def run_gold():
    global WARMUP_COMPLETE

    if not ensure_mt5_connected():
        log.error("[GOLD] MT5 not connected — skip")
        return

    # Warmup
    if not WARMUP_COMPLETE:
        elapsed = (datetime.now(timezone.utc) - STARTUP_TIME).total_seconds()
        remaining = max(0, 120 - elapsed)
        if remaining > 0:
            log.info("[GOLD] Warmup: %.0fs remaining (scanning only)", remaining)
            return
        else:
            WARMUP_COMPLETE = True
            log.info("[GOLD] Warmup complete — trading active")
            notifier.send_plain("[BOT] Warmup complete — now placing orders")

    try:
        # [FIX] Use combined SMC + Classic strategy for higher quality signals
        signal = check_gold_signal_combined(CONFIG)
        if signal is None:
            return
        if not risk.can_trade(signal):
            return

        log.info("[GOLD] Executing: %s | Score:%d | Lot:%.3f | Risk:$%.2f",
                 signal.action, signal.score, signal.lot_or_qty, signal.risk_usdt)

        notifier.send_plain(signal.summary())
        success = execute_gold_trade(signal, CONFIG)

        if success:
            # [FIX] Store ticket → signal mapping for clean sync
            time.sleep(0.5)  # small delay for MT5 to register position
            positions = mt5.positions_get(symbol=signal.symbol, magic=CONFIG["mt5_magic"])
            if positions:
                ticket = positions[-1].ticket
                _ticket_to_signal[ticket] = signal
                log.info("[GOLD] Mapped ticket %d → %s %s", ticket, signal.symbol, signal.action)
            else:
                log.warning("[GOLD] Could not find position ticket after order — sync may miss close")

            risk.register_trade(signal)
            logger.log(signal)
        else:
            notifier.send_plain(f"[WARN] Order Failed: {signal.symbol} {signal.action}")

    except Exception as e:
        log.error("[GOLD] Error: %s", e, exc_info=True)
        notifier.send_plain(f"[ERROR] Gold error: {str(e)[:150]}")


# ── Crypto Trading ────────────────────────────────────────────────────────────

def run_crypto():
    global exchange
    if not CONFIG.get("crypto_symbols"):
        return
    for symbol in CONFIG["crypto_symbols"]:
        try:
            signal = check_crypto_signal(symbol, CONFIG, exchange)
            if signal is None:
                continue
            if not risk.can_trade(signal):
                continue
            log.info("[CRYPTO] Executing: %s %s", signal.symbol, signal.action)
            notifier.send_plain(signal.summary())
            success = execute_crypto_trade(signal, CONFIG, exchange)
            if success:
                risk.register_trade(signal)
                logger.log(signal)
        except Exception as e:
            log.error("[CRYPTO] %s error: %s", symbol, e, exc_info=True)


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_all():
    # Sync first (detect closes + real P&L)
    sync_open_positions()

    # [FIX] REMOVED the UTC 21-22 dead-hour hardcode.
    # session_config.py already handles all session time-gating inside check_gold_signal().
    # Having two separate time filters caused "Low-volume hour 22 UTC — skip Gold"
    # even during the Sydney/Tokyo session which session_config marks as active.
    run_gold()
    run_crypto()


def send_daily_summary():
    msg = logger.get_summary(days=1)
    streak = f"\nWin streak: {risk.consecutive_wins} | Loss streak: {risk.consecutive_losses}"
    notifier.send_plain(f"Daily Summary\n{msg}{streak}")


def send_status():
    notifier.send_plain(f"Bot Status\n{risk.status()}")
    log.info("[BOT] Status:\n%s", risk.status())


def refresh_balance():
    if not ensure_mt5_connected():
        return
    info = mt5.account_info()
    if info:
        CONFIG["gold_account_balance"] = float(info.balance)
        log.info("[MT5] Balance: $%.2f | Equity: $%.2f", info.balance, info.equity)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  SUPER TRADER v7.2 - Starting")
    log.info("=" * 60)

    STARTUP_TIME = datetime.now(timezone.utc)

    mt5_ok = init_mt5()
    if not mt5_ok:
        log.error("[MT5] Init failed")
        notifier.send_plain("[WARN] MT5 failed to connect")
    else:
        log.info("[BOT] Symbol: %s | Balance: $%.2f",
                 CONFIG["mt5_symbol"], CONFIG["gold_account_balance"])
        notifier.send_plain(
            f"SUPER TRADER v7.2 Online\n"
            f"Symbol: {CONFIG['mt5_symbol']}\n"
            f"Balance: ${CONFIG['gold_account_balance']:.2f}\n"
            f"Max trades/dir: 1 | Daily loss limit: 1.5%\n"
            f"Fixes: Sync rewrite | Ticket tracking | Session fix\n"
            f"Warmup: 2 minutes"
        )

    if CONFIG.get("crypto_symbols"):
        try:
            exchange = get_exchange(CONFIG)
            log.info("[CCXT] %s connected", CONFIG["exchange"])
        except Exception as e:
            log.error("[CCXT] Exchange init failed: %s", e)

    # Initial balance + position sync
    refresh_balance()

    # [FIX] Load pre-existing positions BEFORE first sync
    # so restart doesn't lose track of open trades
    if mt5_ok:
        load_existing_positions()

    sync_open_positions()

    # Import session_config here
    from session_config import is_tradeable

    last_scan_time = 0
    last_manage_time = 0
    
    # Schedule daily/status tasks
    schedule.every(30).minutes.do(refresh_balance)
    schedule.every().day.at("00:00").do(send_daily_summary)
    schedule.every(6).hours.do(send_status)

    log.info("[BOT] Turbo Mode Active: Dynamic scanning (30s Overlap / 60s Normal)")
    log.info("[BOT] Warmup: 2 minutes — will observe but NOT trade")
    log.info("[BOT] Limits: max 1 trade/direction | 1.5%% daily loss")
    log.info("[BOT] Session filtering: handled by session_config.py")
    log.info("[BOT] Press Ctrl+C to stop")

    while True:
        try:
            now = time.time()
            
            # Get current session parameters for dynamic interval
            _, session_name, session_params = is_tradeable()
            scan_interval = session_params.get("scan_interval", 60)
            
            # 1. Run main scan (run_all)
            if now - last_scan_time >= scan_interval:
                run_all()
                last_scan_time = now
            
            # 2. Run position management (every 2 mins)
            if now - last_manage_time >= 120:
                manage_positions()
                last_manage_time = now

            # 3. Run other scheduled tasks
            schedule.run_pending()
            
            # Sleep a short amount to remain responsive
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("[BOT] Stopped by user")
            notifier.send_plain("Bot Stopped")
            mt5.shutdown()
            break
        except Exception as e:
            log.error("[BOT] Loop error: %s", e, exc_info=True)
            time.sleep(30)
