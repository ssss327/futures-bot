import asyncio
import json
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import websockets


class WebSocketDataHub:
    """
    Binance Futures WebSocket hub that maintains rolling kline buffers per symbol/timeframe.

    - Uses combined streams on fstream for efficiency
    - Keeps an in-memory deque of recent candles (timestamp, open, high, low, close, volume)
    - Provides async waiters to get notified when a new candle/kline event arrives
    """

    BASE_WS_URL = "wss://fstream.binance.com/stream"

    def __init__(self, max_candles: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_candles = max_candles
        self._buffers: Dict[Tuple[str, str], Deque[List[float]]] = defaultdict(
            lambda: deque(maxlen=self.max_candles)
        )
        self._waiters: Dict[Tuple[str, str], List[asyncio.Future]] = defaultdict(list)
        self._ws_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._subscriptions: Dict[Tuple[str, str], str] = {}  # (symbol, tf) -> stream key
        self._current_streams: List[str] = []

    @staticmethod
    def _to_stream_symbol(symbol: str) -> str:
        # Convert formats like BTC/USDT or BTCUSDT to lowercase btcusdt
        s = symbol.replace("/", "").replace(":USDT", "")
        return s.lower()

    async def subscribe(self, symbol: str, timeframe: str):
        key = (symbol, timeframe)
        if key in self._subscriptions:
            return
        stream = f"{self._to_stream_symbol(symbol)}@kline_{timeframe}"
        self._subscriptions[key] = stream
        await self._restart_ws()

    async def unsubscribe(self, symbol: str, timeframe: str):
        key = (symbol, timeframe)
        if key not in self._subscriptions:
            return
        del self._subscriptions[key]
        await self._restart_ws()

    def get_buffer(self, symbol: str, timeframe: str) -> List[List[float]]:
        return list(self._buffers.get((symbol, timeframe), deque()))

    async def wait_for_update(self, symbol: str, timeframe: str, timeout: Optional[float] = None) -> bool:
        key = (symbol, timeframe)
        fut = asyncio.get_event_loop().create_future()
        self._waiters[key].append(fut)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            if not fut.done():
                fut.cancel()

    async def stop(self):
        self._stop_event.set()
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except Exception:
                pass

    async def _restart_ws(self):
        # recompute streams and restart background task
        streams = sorted(set(self._subscriptions.values()))
        if streams == self._current_streams and self._ws_task and not self._ws_task.done():
            return
        self._current_streams = streams
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except Exception:
                pass
        if not streams:
            return
        self._stop_event.clear()
        self._ws_task = asyncio.create_task(self._run_ws(streams))

    async def _run_ws(self, streams: List[str]):
        # build URL with combined streams
        stream_param = "/".join(streams)
        url = f"{self.BASE_WS_URL}?streams={stream_param}"
        self.logger.info(f"Connecting WS: {url}")
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self.logger.info("WS connected")
                    backoff = 1.0
                    async for msg in ws:
                        if self._stop_event.is_set():
                            break
                        self._handle_message(msg)
            except Exception as e:
                self.logger.warning(f"WS error: {e}. Reconnecting in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_message(self, raw: str):
        try:
            data = json.loads(raw)
            payload = data.get("data") or data
            if not payload:
                return
            if payload.get("e") == "kline":
                k = payload.get("k", {})
                s = payload.get("s") or k.get("s")
                tf = k.get("i")
                symbol = self._normalize_symbol_from_ws(s)
                key = (symbol, tf)
                ts = int(k.get("t"))
                o = float(k.get("o"))
                h = float(k.get("h"))
                l = float(k.get("l"))
                c = float(k.get("c"))
                v = float(k.get("v"))
                closed = bool(k.get("x"))
                # store/replace latest candle by timestamp
                buf = self._buffers[key]
                if buf and buf[-1][0] == ts:
                    buf[-1] = [ts, o, h, l, c, v]
                else:
                    buf.append([ts, o, h, l, c, v])
                # notify waiters only on closed candle to reduce churn
                if closed:
                    for fut in self._waiters.get(key, []):
                        if not fut.done():
                            fut.set_result(True)
                    self._waiters[key].clear()
        except Exception:
            # swallow parsing errors to keep stream alive
            pass

    @staticmethod
    def _normalize_symbol_from_ws(s: str) -> str:
        # Convert ws symbol like BTCUSDT to trading pair BTC/USDT
        if s.upper().endswith("USDT"):
            base = s[:-4]
            return f"{base}/USDT"
        return s


