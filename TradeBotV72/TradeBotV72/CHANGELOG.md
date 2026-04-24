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

## Files changed

```
gold_strategy.py   — major rewrite (v5 → v6)
risk_manager.py    — limits tightened + P&L fix
main.py            — sync rewrite + cooldown integration
notifier.py        — parse_mode fix + send_plain()
config.py          — new v6.0 keys added
```

## Files unchanged

```
crypto_strategy.py
indicators.py
session_config.py
signal_model.py
sr_zones.py
trade_logger.py
dashboard.py
```

---

## Recommended: Test on Demo 1-2 weeks before Live
