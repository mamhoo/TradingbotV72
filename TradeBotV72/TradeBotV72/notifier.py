"""
notifier.py — Telegram Bot notifications

FIXES from v5.0:
  [CRITICAL] parse_mode changed Markdown → HTML to fix 400 Bad Request errors
  [NEW] send_plain() method — sends without any parse_mode (safest fallback)
  [FIX] Encoding: removed BOM/garbage chars from original file
"""

import logging
import requests

log = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, config: dict):
        self.token = config.get("telegram_token", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        raw = config.get("telegram_chat_ids", config.get("telegram_chat_id", ""))
        if isinstance(raw, list):
            self.chat_ids = [str(c) for c in raw if c]
        else:
            self.chat_ids = [str(raw)] if raw else []

    def _send_to(self, chat_id: str, message: str, parse_mode: str = None) -> bool:
        try:
            payload = {
                "chat_id": chat_id,
                "text":    message[:4096],   # Telegram max message length
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            log.error("[TELEGRAM] Send error to %s: %s", chat_id, e)
            return False

    def send(self, message: str) -> bool:
        """Send with HTML parse mode."""
        return self._broadcast(message, parse_mode="HTML")

    def send_plain(self, message: str) -> bool:
        """
        [NEW] Send without any parse_mode — safest option.
        Use this for trade alerts that may contain special characters.
        """
        return self._broadcast(message, parse_mode=None)

    def _broadcast(self, message: str, parse_mode: str = None) -> bool:
        if not self.token or self.token in ("", "YOUR_BOT_TOKEN"):
            log.info("[TELEGRAM] Not configured — skipping: %s", message[:80])
            return False

        success = True
        for chat_id in self.chat_ids:
            if chat_id and chat_id not in ("", "0"):
                ok = self._send_to(chat_id, message, parse_mode)
                if not ok:
                    # Retry once without parse_mode on failure
                    if parse_mode:
                        log.info("[TELEGRAM] Retrying %s without parse_mode", chat_id)
                        ok = self._send_to(chat_id, message, parse_mode=None)
                    if not ok:
                        success = False
        return success

    def send_status(self, risk_manager) -> bool:
        msg = f"Bot Status\n{risk_manager.status()}"
        return self.send_plain(msg)
