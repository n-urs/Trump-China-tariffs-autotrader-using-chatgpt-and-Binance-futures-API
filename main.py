import asyncio
import os
import re
import signal
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple

import hmac
import hashlib
import uuid
import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import RPCError
from openai import AsyncOpenAI  # Responses API client (async)

# =========================
# Config & Environment
# =========================
load_dotenv("config.env")

def _clean(s: Optional[str]) -> str:
    return (s or "").strip().strip('"').strip("'")

TELEGRAM_API_ID = int(_clean(os.getenv("TELEGRAM_API_ID")))
TELEGRAM_API_HASH = _clean(os.getenv("TELEGRAM_API_HASH"))
TG_PHONE = _clean(os.getenv("TG_PHONE"))

BINANCE_API_KEY = _clean(os.getenv("BINANCE_API_KEY"))
BINANCE_API_SECRET = _clean(os.getenv("BINANCE_API_SECRET"))

OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))

ORDER_QTY = Decimal(_clean(os.getenv("ORDER_QUANTITY") or "50"))
# Superpositive size (defaults to ORDER_QTY if not set)
ORDER_QTY_SUPER = Decimal(_clean(os.getenv("ORDER_QUANTITY_SUPER") or str(ORDER_QTY)))

CONTRACT_PAIR = _clean(os.getenv("CONTRACT_PAIR") or "ETHUSDT").upper()

LONG_TP_MULT = Decimal(_clean(os.getenv("LONG_TP_MULTIPLIER") or "1.029"))
LONG_SL_MULT = Decimal(_clean(os.getenv("LONG_SL_MULTIPLIER") or "0.9939"))
SHORT_TP_MULT = Decimal(_clean(os.getenv("SHORT_TP_MULTIPLIER") or "0.971"))
SHORT_SL_MULT = Decimal(_clean(os.getenv("SHORT_SL_MULTIPLIER") or "1.0061"))

LONG_TP_FALLBACK_MULT = Decimal(_clean(os.getenv("LONG_TP_FALLBACK_MULTIPLIER") or "1.049"))
LONG_SL_FALLBACK_MULT = Decimal(_clean(os.getenv("LONG_SL_FALLBACK_MULTIPLIER") or "0.989"))
SHORT_TP_FALLBACK_MULT = Decimal(_clean(os.getenv("SHORT_TP_FALLBACK_MULTIPLIER") or "0.951"))
SHORT_SL_FALLBACK_MULT = Decimal(_clean(os.getenv("SHORT_SL_FALLBACK_MULTIPLIER") or "1.011"))

ALERT_BOT_TOKEN = _clean(os.getenv("ALERT_BOT_TOKEN"))
ALERT_CHAT_ID = _clean(os.getenv("ALERT_CHAT_ID"))

HEALTHCHECK_INTERVAL = int(_clean(os.getenv("HEALTHCHECK_INTERVAL") or "30"))
RESTART_DELAY = int(_clean(os.getenv("RESTART_DELAY") or "5"))
TRADE_COOLDOWN_SEC = int(_clean(os.getenv("TRADE_COOLDOWN_SEC") or "300"))

# Original GPT-classified channels
TARGET_CHANNEL_IDS = {-1002442330266, -1002833482708}
# New logic-only channels
LOGIC_CHANNEL_IDS = {-1002481959011, -1002364176580}
ALL_CHANNEL_IDS = set(TARGET_CHANNEL_IDS) | set(LOGIC_CHANNEL_IDS)

TARIFF_PATTERN = re.compile(r"tariff", flags=re.IGNORECASE)

# ===== Logic trigger regexes for 10–25% =====
LOGIC_NEED_TARIFF = re.compile(r"tariff", re.IGNORECASE)
LOGIC_NEED_CHIN = re.compile(r"chin", re.IGNORECASE)  # matches china, chinese, chin...
# numeric: 10..25 followed by %, percent, per cent, or parcent
NUMERIC_PERCENT = re.compile(
    r"\b(?:1[0-9]|2[0-5])\s*(?:%|percent|per\s*cent|parcent)\b",
    re.IGNORECASE
)
# spelled-out 10..25: ten..nineteen, twenty, twenty-one..twenty-five
WORDS_10_19 = r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
WORDS_20_25 = r"(?:twenty(?:[-\s]?(?:one|two|three|four|five))?)"
SPELL_PERCENT = re.compile(
    rf"\b(?:{WORDS_10_19}|{WORDS_20_25})\s*(?:percent|per\s*cent|parcent)\b",
    re.IGNORECASE
)

def logic_superpositive_trigger(text: str) -> bool:
    """Return True if text satisfies logic trigger: tariff + chin + (10–25%)."""
    return (
        LOGIC_NEED_TARIFF.search(text) is not None
        and LOGIC_NEED_CHIN.search(text) is not None
        and (NUMERIC_PERCENT.search(text) is not None or SPELL_PERCENT.search(text) is not None)
    )

# =========================
# Logging
# =========================
LOG_PATH = "tariff_eth_bot.log"
logger = logging.getLogger("tariff_eth_bot")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S%z")
fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
fh.setFormatter(fmt)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)

# =========================
# Globals
# =========================
SESSION_ID = uuid.uuid4().hex[:10]
trade_lock = asyncio.Lock()
last_trade_ts: Optional[float] = None
binance_time_offset_ms: int = 0
stop_event = asyncio.Event()

# HTTP clients
BINANCE_BASE = "https://fapi.binance.com"  # USDT-M Futures mainnet
TELEGRAM_BOT_API = "https://api.telegram.org"

binance_client = httpx.AsyncClient(base_url=BINANCE_BASE, timeout=5.0)
generic_client = httpx.AsyncClient(timeout=5.0)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Responses API

# =========================
# Utilities
# =========================
def utcnow_ms() -> int:
    return int(time.time() * 1000) + binance_time_offset_ms

def hmac_sign(query: str) -> str:
    return hmac.new(BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def signed_params(params: Dict) -> Dict:
    params["timestamp"] = utcnow_ms()
    qs = urlencode(params, doseq=True)
    params["signature"] = hmac_sign(qs)
    return params

def price2(p: Decimal) -> str:
    return str(p.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

async def send_alert(text: str) -> None:
    if not ALERT_BOT_TOKEN or not ALERT_CHAT_ID:
        logger.warning("Alert bot not configured; skipping alert: %s", text)
        return
    try:
        url = f"{TELEGRAM_BOT_API}/bot{ALERT_BOT_TOKEN}/sendMessage"
        await generic_client.post(url, data={"chat_id": ALERT_CHAT_ID, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.exception("Failed to send alert: %s", e)

async def binance_ping() -> bool:
    try:
        r = await binance_client.get("/fapi/v1/ping")
        return r.status_code == 200
    except Exception:
        return False

async def binance_server_time() -> Optional[int]:
    try:
        r = await binance_client.get("/fapi/v1/time")
        r.raise_for_status()
        return r.json()["serverTime"]
    except Exception as e:
        logger.warning("Failed to fetch Binance time: %s", e)
        return None

async def sync_time() -> None:
    global binance_time_offset_ms
    server_time = await binance_server_time()
    if server_time is None:
        return
    local = int(time.time() * 1000)
    binance_time_offset_ms = server_time - local
    logger.info("Time sync: offset_ms=%d", binance_time_offset_ms)

# =========================
# Binance helpers
# =========================
async def assert_binance_ready() -> bool:
    """Fail-fast auth check at startup."""
    try:
        params = signed_params({"recvWindow": 10000})
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        r = await binance_client.get("/fapi/v2/balance", params=params, headers=headers)
        if r.status_code != 200:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text}
            await send_alert(f"🛑 Binance Futures auth check failed ({r.status_code}): {body}")
            logger.error("Futures auth check failed: %s", body)
            return False
        return True
    except Exception as e:
        logger.exception("Futures auth check exception: %s", e)
    # Alert outside except to avoid duplicate alerts on JSON decode
        await send_alert("🛑 Binance Futures auth check exception; see logs.")
        return False

async def get_position_ethusdt() -> Tuple[Decimal, Decimal]:
    """
    Returns (positionAmt, entryPrice). On auth/signature errors we raise,
    so the pipeline can alert & abort (no trading when not authorized).
    """
    params = signed_params({"symbol": CONTRACT_PAIR, "recvWindow": 10000})
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    r = await binance_client.get("/fapi/v2/positionRisk", params=params, headers=headers)
    if r.status_code != 200:
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        logger.error("Binance positionRisk error %s: %s", r.status_code, body)
        r.raise_for_status()  # will raise HTTPStatusError
    data = r.json()
    if not data:
        return Decimal("0"), Decimal("0")
    item = data[0]
    return Decimal(item["positionAmt"]), Decimal(item["entryPrice"])

def build_cx_id(tag: str) -> str:
    return f"{tag}-{SESSION_ID}-{int(time.time()*1000)%10_000_000}".replace(" ", "")[:36]

async def place_market_order(side: str, qty: Decimal, resp_type: str = "RESULT") -> Optional[Dict]:
    try:
        params = {
            "symbol": CONTRACT_PAIR,
            "side": side,
            "type": "MARKET",
            "quantity": str(qty.normalize()),
            "newOrderRespType": resp_type,
            "recvWindow": 10000,
            "newClientOrderId": build_cx_id("MKT"),
        }
        params = signed_params(params)
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        r = await binance_client.post("/fapi/v1/order", params=params, headers=headers)
        if r.status_code != 200:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text}
            logger.error("Market order error %s: %s", r.status_code, body)
            return None
        return r.json()
    except Exception as e:
        logger.exception("Market order failed: %s", e)
        return None

async def close_position_immediately() -> None:
    try:
        amt, _ = await get_position_ethusdt()
    except Exception as e:
        logger.exception("Cannot fetch position to close: %s", e)
        return
    if amt == 0:
        return
    side = "BUY" if amt < 0 else "SELL"
    qty = abs(amt)
    try:
        params = {
            "symbol": CONTRACT_PAIR,
            "side": side,
            "type": "MARKET",
            "quantity": str(qty.normalize()),
            "reduceOnly": "true",
            "recvWindow": 10000,
            "newClientOrderId": build_cx_id("CLS"),
        }
        params = signed_params(params)
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        r = await binance_client.post("/fapi/v1/order", params=params, headers=headers)
        if r.status_code != 200:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text}
            logger.error("Close position error %s: %s", r.status_code, body)
        else:
            logger.info("Position closed immediately via market order.")
            await send_alert(f"⚠️ Closed position immediately (fallback failure). Side={side} Qty={qty}")
    except Exception as e:
        logger.exception("Failed to close position immediately: %s", e)

async def place_tp_sl(
    side_exit: str,
    tp_price: Decimal,
    sl_price: Decimal,
    working_type: str = "MARK_PRICE",
    attempt_tag: str = "A1",
) -> Tuple[bool, bool]:
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    tp_params = {
        "symbol": CONTRACT_PAIR,
        "side": side_exit,
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": price2(tp_price),
        "closePosition": "true",
        "workingType": working_type,
        "recvWindow": 10000,
        "newClientOrderId": build_cx_id(f"TP{attempt_tag}"),
    }
    sl_params = {
        "symbol": CONTRACT_PAIR,
        "side": side_exit,
        "type": "STOP_MARKET",
        "stopPrice": price2(sl_price),
        "closePosition": "true",
        "workingType": working_type,
        "recvWindow": 10000,
        "newClientOrderId": build_cx_id(f"SL{attempt_tag}"),
    }

    tp_ok = sl_ok = False
    try:
        r1 = await binance_client.post("/fapi/v1/order", params=signed_params(tp_params), headers=headers)
        if r1.status_code == 200:
            tp_ok = True
        else:
            try:
                logger.error("TP placement failed %s: %s", r1.status_code, r1.json())
            except Exception:
                logger.error("TP placement failed %s: %s", r1.status_code, r1.text)
    except Exception as e:
        logger.exception("TP placement exception: %s", e)

    try:
        r2 = await binance_client.post("/fapi/v1/order", params=signed_params(sl_params), headers=headers)
        if r2.status_code == 200:
            sl_ok = True
        else:
            try:
                logger.error("SL placement failed %s: %s", r2.status_code, r2.json())
            except Exception:
                logger.error("SL placement failed %s: %s", r2.status_code, r2.text)
    except Exception as e:
        logger.exception("SL placement exception: %s", e)

    return tp_ok, sl_ok

# =========================
# OpenAI classification
# =========================
async def classify_with_gpt5_nano(message_text: str) -> str:
    """
    Returns one of: superpositive | positive | negative | neutral | irrelevant
    (Prompt already updated by you; leaving as-is.)
    """
    import re as _re
    prompt = (
        f"{message_text}\n"
        "Analyze the message above written by Donald Trump on the presence of any information regarding U.S. tariffs for China. "
        'If there is clear information about tariffs on China, write if it is "superpositive", "positive", "negative", or "neutral". '
        'If the message is irrelevant to the tariffs on China, respond "irrelevant". ANSWER IN ONE WORD ONLY: superpositive, positive, negative, neutral, irrelevant.\n\n'
        "If there is only a picture or other irrelevant info, it is irrelevant. "
        "If the message is about China but there is no clear tariff rate indication, it is neutral. "
        "If the new tariff rate is less than 30%, then respond superpositive. "
        "If the new tariff rate is equal to 30%, then respond positive. "
        "If the new tariff rate is more than 30%, then respond negative. "
        'If the message explicitly says the U.S. and China have reached a deal (e.g., "reached an agreement", "deal finalized", "agreement signed"), respond positive.'
    )
    try:
        resp = await openai_client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        text = (resp.output_text or "").strip().lower()
        m = _re.search(r"\b(superpositive|positive|negative|neutral|irrelevant)\b", text)
        return m.group(1) if m else "irrelevant"
    except Exception as e:
        logger.exception("OpenAI classification failed: %s", e)
        return "irrelevant"

# =========================
# Healthcheck & Runner
# =========================
async def healthcheck_loop(tg_client: TelegramClient) -> None:
    last_sync = 0.0
    while not stop_event.is_set():
        try:
            if not await tg_client.is_user_authorized():
                logger.warning("Telegram client not authorized")
            ok = await binance_ping()
            if not ok:
                logger.warning("Binance ping failed")
            now = time.time()
            if now - last_sync > 600 or binance_time_offset_ms == 0:
                await sync_time()
                last_sync = now
        except Exception as e:
            logger.exception("Healthcheck error: %s", e)
        await asyncio.sleep(HEALTHCHECK_INTERVAL)

async def process_trade_pipeline(
    text: str,
    channel_id: int,
    message_id: int,
    event_time: float,
    override_label: Optional[str] = None,
    source: str = "GPT",
) -> None:
    global last_trade_ts

    # --- Label (logic or GPT)
    if override_label:
        label = override_label
        to_gpt_ms = 0
        logger.info("Using override label from %s: %s", source, label)
    else:
        t1 = time.perf_counter()
        label = await classify_with_gpt5_nano(text)
        t2 = time.perf_counter()
        to_gpt_ms = int((t2 - t1) * 1000)
        logger.info(
            "GPT classification: %s | channel=%s msg_id=%s | durations: to_gpt=%dms",
            label, channel_id, message_id, to_gpt_ms
        )

    # Alert on trigger labels (even if we later skip due to cooldown/position)
    if label in ("superpositive", "positive", "negative"):
        await send_alert(f"📣 Trigger ({source}): <b>{label.upper()}</b> | chat={channel_id} msg={message_id}")

    if label == "irrelevant":
        logger.info("Irrelevant message: logged only.")
        return
    if label == "neutral":
        await send_alert(f"ℹ️ <b>NEUTRAL</b> — Forwarded message:\n\n{text}")
        return

    async with trade_lock:
        now = time.time()
        if last_trade_ts and (now - last_trade_ts) < TRADE_COOLDOWN_SEC:
            logger.info("Cooldown active; skipping trade. %.0fs left", TRADE_COOLDOWN_SEC - (now - last_trade_ts))
            return

        # Ensure we can query position (auth). Abort with alert on error.
        try:
            pos_amt, _ = await get_position_ethusdt()
        except httpx.HTTPStatusError:
            await send_alert("🛑 Binance auth/signature error on positionRisk (401/4xx). "
                             "Check API key/secret, Futures permission, IP whitelist, and mainnet vs testnet.")
            logger.error("Aborting trade due to Binance auth/signature error.")
            return
        except Exception:
            await send_alert("🛑 Binance error on positionRisk. Aborting trade; see logs.")
            return

        if pos_amt != 0:
            logger.info("Existing ETHUSDT position detected (%.3f). Skipping new trade.", pos_amt)
            return

        # Decide side and quantity
        if label in ("positive", "superpositive"):
            side_open = "BUY"
            side_exit = "SELL"
            qty = ORDER_QTY_SUPER if label == "superpositive" else ORDER_QTY
        else:  # negative
            side_open = "SELL"
            side_exit = "BUY"
            qty = ORDER_QTY

        # --- Place market order
        t3 = time.perf_counter()
        await send_alert(f"⚙️ Trigger={source} — Sending MARKET {side_open} {qty} {CONTRACT_PAIR} (reason: {label})")
        order_resp = await place_market_order(side_open, qty)
        t4 = time.perf_counter()
        if not order_resp:
            logger.error("Market order failed; aborting.")
            return
        await send_alert(f"✅ Order sent: {side_open} {qty} {CONTRACT_PAIR}. Took {int((t4 - t3)*1000)}ms")

        # --- Confirm position open & get entry
        entry_price: Decimal = Decimal("0")
        opened = False
        t5 = time.perf_counter()
        for _ in range(20):  # ~2s
            await asyncio.sleep(0.1)
            try:
                amt, ep = await get_position_ethusdt()
            except Exception as e:
                logger.exception("Error re-checking position after order: %s", e)
                continue
            if amt != 0:
                entry_price = ep
                opened = True
                break
        t6 = time.perf_counter()
        if not opened or entry_price == 0:
            logger.error("Position failed to open or entry price unavailable; aborting.")
            return
        await send_alert(
            f"🟩 Position opened: side={side_open} qty={qty} entry≈{entry_price} ({int((t6 - t5)*1000)}ms confirm)"
        )

        # --- TP/SL math
        if label in ("positive", "superpositive"):  # LONG
            tp = Decimal(price2(entry_price * LONG_TP_MULT))
            sl = Decimal(price2(entry_price * LONG_SL_MULT))
            tp_fb = Decimal(price2(entry_price * LONG_TP_FALLBACK_MULT))
            sl_fb = Decimal(price2(entry_price * LONG_SL_FALLBACK_MULT))
        else:  # SHORT
            tp = Decimal(price2(entry_price * SHORT_TP_MULT))
            sl = Decimal(price2(entry_price * SHORT_SL_MULT))
            tp_fb = Decimal(price2(entry_price * SHORT_TP_FALLBACK_MULT))
            sl_fb = Decimal(price2(entry_price * SHORT_SL_FALLBACK_MULT))

        # --- Place TP/SL with retry & fallback
        t7 = time.perf_counter()
        tp_ok, sl_ok = await place_tp_sl(side_exit, tp, sl, attempt_tag="A1")
        if not (tp_ok and sl_ok):
            tp_ok2, sl_ok2 = await place_tp_sl(side_exit, tp, sl, attempt_tag="A2")
            tp_ok = tp_ok or tp_ok2
            sl_ok = sl_ok or sl_ok2
        if not (tp_ok and sl_ok):
            await send_alert(f"⚠️ TP/SL initial placement failed. Using fallbacks: TP={tp_fb} SL={sl_fb}")
            tp_ok_fb, sl_ok_fb = await place_tp_sl(side_exit, tp_fb, sl_fb, attempt_tag="FB")
            tp_ok = tp_ok or tp_ok_fb
            sl_ok = sl_ok or sl_ok_fb
        t8 = time.perf_counter()

        if tp_ok and sl_ok:
            await send_alert(f"🎯 TP/SL placed ({int((t8 - t7)*1000)}ms): TP={tp} SL={sl} (fallbacks may have been used)")
        else:
            await send_alert("🛑 TP/SL placement failed even after fallbacks. Closing position now.")
            await close_position_immediately()

        last_trade_ts = time.time()

async def telegram_runner() -> None:
    tg_client = TelegramClient("monitor", TELEGRAM_API_ID, TELEGRAM_API_HASH)

    async def handle_new_message(event):
        try:
            if event.chat_id not in ALL_CHANNEL_IDS:
                return
            text = (event.raw_text or "").strip()
            if not text:
                return

            # Logic-only channels first (no GPT call here)
            if event.chat_id in LOGIC_CHANNEL_IDS:
                if logic_superpositive_trigger(text):
                    logger.info(
                        "Logic hit: tariff+chin+10-25%% | chat=%s msg_id=%s len=%d",
                        event.chat_id, event.message.id, len(text),
                    )
                    await process_trade_pipeline(
                        text, event.chat_id, event.message.id, time.perf_counter(),
                        override_label="superpositive",
                        source="LOGIC",
                    )
                return  # Do not pass to GPT for logic-only channels

            # GPT-classified channels
            if event.chat_id in TARGET_CHANNEL_IDS:
                if not TARIFF_PATTERN.search(text):
                    return
                logger.info("Message matched 'tariff': chat=%s msg_id=%s len=%d",
                            event.chat_id, event.message.id, len(text))
                await process_trade_pipeline(
                    text, event.chat_id, event.message.id, time.perf_counter(),
                    override_label=None,
                    source="GPT",
                )
        except Exception as e:
            logger.exception("Error in message handler: %s", e)

    tg_client.add_event_handler(
        handle_new_message, events.NewMessage(chats=list(ALL_CHANNEL_IDS))
    )

    await tg_client.start(phone=TG_PHONE)
    logger.info("Telegram client started. Monitoring channels: %s", ", ".join(map(str, ALL_CHANNEL_IDS)))
    hc_task = asyncio.create_task(healthcheck_loop(tg_client))
    try:
        await stop_event.wait()
    finally:
        hc_task.cancel()
        await tg_client.disconnect()
        await binance_client.aclose()
        await generic_client.aclose()
        await openai_client.close()
        logger.info("Shutdown complete.")

def setup_signal_handlers():
    def _graceful_shutdown(*_):
        logger.info("SIGINT/SIGTERM received, shutting down gracefully...")
        stop_event.set()
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

async def main_loop():
    setup_signal_handlers()
    logger.info("Starting tariff ETH bot (session=%s)", SESSION_ID)
    await sync_time()

    # Fail-fast auth check (don’t trade if not ready)
    ready = await assert_binance_ready()
    if not ready:
        logger.error("Binance Futures not ready — fix credentials/permissions and restart.")
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            await telegram_runner()
        except (httpx.HTTPError, RPCError, OSError) as e:
            logger.exception("Network/API error: %s. Restarting in %ds", e, RESTART_DELAY)
            await asyncio.sleep(RESTART_DELAY)
        except Exception as e:
            logger.exception("Fatal error: %s. Restarting in %ds", e, RESTART_DELAY)
            await asyncio.sleep(RESTART_DELAY)

if __name__ == "__main__":
    asyncio.run(main_loop())

