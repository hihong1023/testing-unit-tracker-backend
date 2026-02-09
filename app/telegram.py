import requests
import os

TELEGRAM_BOT_TOKEN = os.getenv("8438811074:AAFR-FAMKmSEPU3QQ8S-NBcgcx2hgerZdUs")
TELEGRAM_CHAT_ID = os.getenv("-5145199200")

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram env vars not set")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": int(TELEGRAM_CHAT_ID),
        "text": text,
        "parse_mode": "Markdown"
    }

    r = requests.post(url, json=payload, timeout=5)
    r.raise_for_status()
