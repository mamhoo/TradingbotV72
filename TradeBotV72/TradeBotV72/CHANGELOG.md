# TawmhooBot — Changelog

---

## v7.3 — P&L Optimization & First-Period Loss Fix

### Root Causes Identified (Win Rate High, P&L Low)

| # | Root Cause | Impact |
|---|---|---|
| 1 | `bb_squeeze_strategy.py` hardcoded `lot_or_qty=0.01` | Fixed lot = tiny P&L per win regardless of account size |
| 2 | `risk_manager.py` halted after only 2 consecutive losses | Early warmup losses permanently disabled bot for the day |
| 3 | `smc_gold_strategy.py` passed `symbol` as 6th arg to `calculate_lot_size()` | Wrong parameter binding → minimum lot on all SMC trades |
| 4 | Base RR ratio was 1.5 across all strategies | Low reward-per-win suppressed total P&L despite good win rate |
| 5 | No session-based lot scaling in BB Squeeze | Same size during low-liquidity sessions as high-liquidity |

---

### v7.3 Fixes & Optimizations

#### CRITICAL — P&L fixes

| File | Change |
|---|---|
| `bb_squeeze_strategy.py` | **Dynamic lot sizing**: replaced hardcoded `0.01` with `calculate_lot_size()` based on account balance, risk %, and ATR |
| `bb_squeeze_strategy.py` | **Session scaling**: London/NY Overlap = 1.25× lot; Sydney/Tokyo = 0.75× lot |
| `bb_squeeze_strategy.py` | **Higher RR targets**: SQUEEZE_BREAKOUT = 2.5R (was 2.0R); M5_REVERSION = 1.8R |
| `bb_squeeze_strategy.py` | **Tighter SL**: `atr_sl_mult` 1.5 → 1.2 ATR (better risk-adjusted returns) |
| `bb_squeeze_strategy.py` | **Improved scores**: H1+H4 aligned breakout = 80 (was 70); M5_REVERSION = 55 (was 50) |
| `bb_squeeze_strategy.py` | **Partial TP levels** added (was missing entirely) |
| `smc_gold_strategy.py` | **Fixed `calculate_lot_size()` call**: removed erroneous `symbol` positional arg that was binding to `lot_base` parameter |
| `gold_strategy.py` | **Base RR raised** from 1.5 → 2.0 in `calculate_dynamic_rr()` |
| `gold_strategy.py` | **SL multiplier tightened** from 1.2 → 1.0 ATR for better R:R |
| `gold_strategy.py` | **Partial TP improved**: TP1=1R(40%), TP2=1.5R(35%), TP3=fullRR(25%) — was 1R(50%)/2R(30%)/fullRR(20%) |

#### CRITICAL — First-period loss fix

| File | Change |
|---|---|
| `risk_manager.py` | **`max_consecutive_losses` raised from 2 → 3**: 2 was too aggressive; early warmup losses halted the bot before recovery was possible |
| `risk_manager.py` | **`startup_grace_trades = 2`**: first 2 trades per day bypass the consecutive-loss halt so indicator warmup doesn't kill the session |
| `risk_manager.py` | **`max_daily_loss_pct` raised from 1.5% → 2.0%**: gives the bot room to recover from early losses |
| `risk_manager.py` | **Daily P&L-aware risk multiplier**: `get_risk_multiplier()` now also reduces size when daily P&L is negative |

#### Config defaults updated

| Key | Old | New | Reason |
|---|---|---|---|
| `max_daily_loss_pct` | 1.5% | 2.0% | Allow recovery from early losses |
| `gold_rr_ratio` | 1.5 | 2.0 | Better P&L per trade |
| `gold_scalp_rr` | 1.5 | 2.0 | Consistent with new base RR |

---

### Files Changed

```
bb_squeeze_strategy.py  — v2 → v3 (dynamic lots, higher RR, session scaling, partial TPs)
gold_strategy.py        — v6.1 → v7.3 (higher base RR, tighter SL, improved partial TP)
smc_gold_strategy.py    — v7.0 → v7.3 (fixed calculate_lot_size argument order)
risk_manager.py         — v7.0 → v7.3 (grace period, higher limits, daily P&L multiplier)
config.py               — v6.1 → v7.3 (updated defaults)
```

### Files Unchanged

```
indicators.py
indicators_ext.py
session_config.py
signal_model.py
sr_zones.py
smc_concepts.py
mean_reversion_strategy.py
trade_logger.py
notifier.py
main.py
backtester.py
dashboard.py
```

---

## v6.1 — Signal Quality & Trailing Stop Fixes

*(Previous changelog — see below)*

---

# TawmhooBot v6.0 — Changelog

## Root Cause of April 9 Loss (~75% drawdown in 4 hours)

1. **Volume filter didn't block** — logged "below threshold" but kept generating signals
2. **Score was stuck at 90/100 every candle** — EMA_PULLBACK threshold too loose (0.5%)
3. **normal_atr = 1.0 hardcoded** — Gold ATR is ~10, so lot sizing was broken
4. **SELL SL had wrong sign** — spread_buf subtracted instead of added
5. **P&L tracking returned $0.00** — sync didn't read MT5 deal history properly
6. **No cooldown after loss** — re-entered immediately after SL hit
7. **Telegram 400 errors** — Markdown parse_mode rejected by API

---

## v6.0 Fixes

### CRITICAL — account protection

| File | Fix |
|---|---|
| `gold_strategy.py` | Volume filter now **returns None** (blocks signal), not just reduces score |
| `gold_strategy.py` | `normal_atr` changed `1.0` → `10.0` (realistic Gold baseline) |
| `gold_strategy.py` | SELL SL: `- spread_buf` → `+ spread_buf` (correct direction) |
| `gold_strategy.py` | `EMA_PULLBACK` tightened: `dist < 0.005` → `dist < 0.002` so score varies more |
| `gold_strategy.py` | **Anti-chasing**: block entry if price > 0.4% from EMA21 |
| `gold_strategy.py` | **Cooldown**: 45 min block after any SL hit |
| `gold_strategy.py` | **D1 trend gate**: BUY blocked if Daily trend DOWN, SELL blocked if Daily UP |
| `risk_manager.py` | `max_trades_per_direction`: 2 → **1** |
| `risk_manager.py` | `max_daily_loss_pct`: 3.0% → **1.5%** |
| `risk_manager.py` | `max_consecutive_losses`: 3 → **2** |
| `risk_manager.py` | `max_daily_trades`: 6 → **4** |
| `risk_manager.py` | `normal_atr`: 1.0 → **10.0** (ATR multiplier fixed) |
| `risk_manager.py` | Daily reset now properly resets `halted = False` |
| `main.py` | P&L tracking reads **actual MT5 deal history** by ticket (not $0.00) |
| `main.py` | `register_sl_hit()` called when trade closes at loss → triggers cooldown |
| `notifier.py` | `parse_mode` changed `Markdown` → `HTML` (fixes 400 errors) |
| `notifier.py` | Added `send_plain()` with retry on parse failure |

### IMPORTANT — signal quality

| Fix | Detail |
|---|---|
| D1 trend gate | Only BUY if Daily ≥ NEUTRAL, only SELL if Daily ≤ NEUTRAL |
| Anti-chase | Max 0.4% distance from EMA21 at entry time |
| Score distribution | Now varies 50–110 (was stuck at 90/100 every candle) |
| Volume threshold | Raised 1.2x → 1.3x |
| RSI thresholds | BUY block at 72 (was 75), SELL block at 28 (was 25) — more symmetric |

### New config keys (.env)

```
GOLD_MIN_VOLUME_RATIO=1.3       # volume gate (default 1.3)
GOLD_MAX_ENTRY_DIST_PCT=0.004   # anti-chase (default 0.004 = 0.4%)
GOLD_MIN_SCORE=55               # minimum score threshold
MAX_DAILY_LOSS_PCT=1.5          # daily halt (default 1.5%)
```

---

## Recommended: Test on Demo 1-2 weeks before Live
