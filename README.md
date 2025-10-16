# Trump-China-tariffs-autotrader-using-chatgpt-API

Real-time trading bot that watches two Telegram channels (Trump's TruthSocial scraper and a test channel) for the word **“tariff”**, classifies each hit with OpenAI, and trades **ETHUSDT** on **Binance USDT-M Futures** accordingly.

> **Speed note:** **ChatGPT 5 Nano typically takes ~6–7 seconds** per analysis. **ChatGPT 3.5 Turbo is ~2–3 seconds** but may be **less accurate**.

---

## What it does

* Monitors two Telegram channels via **Telethon**.
* Sends matched messages to OpenAI with a strict **one-word label**: `positive`, `negative`, `neutral`, or `irrelevant`.
* Trading logic (Binance USDT-M Futures):

  * **positive → LONG**, **negative → SHORT** with **50 ETH** (configurable).
  * Sets **TP/SL** (Take Profit / Stop Loss) using multipliers; retries once, then **fallback TP/SL**; if still failing, closes the position immediately.
  * **Skips** trading if an **ETHUSDT position already exists**.
  * **Cooldown** (default 300s) avoids duplicate entries on bursty messages.
  * **Idempotent** per-session order IDs to prevent duplicates.
* **Alert bot** (separate Telegram bot) sends order/position/TP-SL notifications.
* **Health checks**: Telegram auth, Binance ping, and time sync (startup + every 10 min).
* **Resilience**: Graceful shutdown (Ctrl+C), auto-restart on network errors, structured logging with **durations** for each step.

---

## Requirements

* Python **3.10+**
* A Telegram account + API ID/HASH (for Telethon)
* A Telegram **bot token** and **chat ID** (for alerts)
* **Binance USDT-M Futures** API key with **Futures enabled** (mainnet or testnet)
* OpenAI API key

Install dependencies:

```bash
pip install telethon httpx python-dotenv openai
```

---

## Configuration

You already added **`config.env`**. Fill in your keys and settings there.

Key variables (brief):

* `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TG_PHONE`
* `BINANCE_API_KEY`, `BINANCE_API_SECRET`
* `OPENAI_API_KEY`
* Trading:

  * `ORDER_QUANTITY`, `CONTRACT_PAIR`
  * `LONG_TP_MULTIPLIER`, `LONG_SL_MULTIPLIER`
  * `SHORT_TP_MULTIPLIER`, `SHORT_SL_MULTIPLIER`
  * Fallbacks: `*_FALLBACK_MULTIPLIER`
* Alert bot:

  * `ALERT_BOT_TOKEN`, `ALERT_CHAT_ID`
* Ops:

  * `HEALTHCHECK_INTERVAL`, `RESTART_DELAY`, `TRADE_COOLDOWN_SEC`

> Tip: If you use **testnet**, swap the base URL in the script (`BINANCE_BASE`) and use testnet keys.

---

## Change the OpenAI model (speed vs. accuracy)

The script defaults to `gpt-5-nano` in the classification function.
You can switch to **ChatGPT 3.5 Turbo** by changing:

```python
resp = await openai_client.responses.create(
    model="gpt-3.5-turbo",   # was "gpt-5-nano"
    input=prompt
)
```

* **Performance note:** **ChatGPT 5 Nano ~6–7s** per analysis vs. **3.5 Turbo ~2–3s**.
  **3.5 Turbo may be less accurate.**

---

## Run

```bash
python tariff_eth_bot.py
```

First run will prompt Telethon to sign in (one-time code). Next runs reuse the local session file.

---

## Logs & Monitoring

* Logs in `tariff_eth_bot.log` with timestamps and per-stage **durations**:

  * receive → OpenAI
  * order send / open confirm
  * TP/SL placement
* Telegram **alerts** for:

  * Order sending/opened
  * TP/SL placed (or fallback/close on failure)
  * Critical errors (e.g., Binance auth/signature issues)

---

## Safety & Notes

* **API (Application Programming Interface) keys:** ensure Binance key has **Futures enabled** and IP whitelist is correct.
* **USDT-M** = Tether-margined futures.
* **TP/SL** = Take Profit / Stop Loss.
* The bot **does not fetch contract filters** before opening to reduce latency.
* On 4xx auth errors, trading **aborts** and you’ll get an alert (no silent “fake position”).

---

## Disclaimer

This code is for educational purposes. Crypto trading is risky. Use at your own risk, test thoroughly (prefer **testnet**) before trading real funds.

