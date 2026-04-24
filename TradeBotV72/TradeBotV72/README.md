# TradeBotV6

⚠️ **CRITICAL SECURITY NOTICE** ⚠️
If you previously cloned this repository, your MT5 credentials and Telegram tokens may have been exposed. 
1. **Change your MT5 password immediately.**
2. **Revoke your Telegram Bot Token.**
3. **Delete your local `.env` file and use `.env.example` as a template.**

## Security Fixes (v6.1)
- **Credential Protection**: `.env` is now ignored by git.
- **Trailing Stop Fix**: Corrected logic for SELL positions to prevent premature stop-outs.
- **Volume Filter Fix**: Corrected M5 volume extrapolation logic.
- **SMC Strategy Activation**: The bot now uses the combined SMC + Classic strategy for higher quality signals.
- **Score Logic Fix**: Removed double-penalty on D1 trend filter.
- **Session-Specific Gates**: Now correctly enforces `min_score` per trading session.

## Setup
1. Copy `.env.example` to `.env`
2. Fill in your credentials in `.env`
3. Install dependencies: `pip install -r requirements.txt` (if applicable)
4. Run the bot: `python main.py`
