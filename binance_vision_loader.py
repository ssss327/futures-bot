import asyncio
import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import zipfile


class BinanceVisionLoader:
    """
    Minimal Binance Vision dataset loader for historical klines.

    Downloads zipped monthly/daily kline CSVs on-demand and returns a DataFrame with
    columns: timestamp, open, high, low, close, volume (index: timestamp)
    """

    BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _symbol_to_vision(symbol: str) -> str:
        # Convert BTC/USDT -> BTCUSDT, BTCUSDT:USDT -> BTCUSDT
        s = symbol.replace("/", "").replace(":USDT", "")
        return s.upper()

    async def load_range(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        # Binance Vision provides daily zip files per date for many intervals
        # We'll iterate days and concatenate available files.
        sym = self._symbol_to_vision(symbol)
        cur = start_date.date()
        out = []
        session = requests.Session()
        while cur <= end_date.date():
            date_str = cur.strftime("%Y-%m-%d")
            # File pattern example: BTCUSDT-1m-2024-01-01.zip
            url = f"{self.BASE_URL}/{sym}/{timeframe}/{sym}-{timeframe}-{date_str}.zip"
            try:
                resp = await asyncio.get_event_loop().run_in_executor(None, session.get, url)
                if resp.status_code != 200:
                    cur = cur.fromordinal(cur.toordinal() + 1)
                    continue
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                # Inside zip: CSV with same base name
                for name in zf.namelist():
                    with zf.open(name) as f:
                        df = pd.read_csv(f, header=None)
                        # Columns per Binance Vision spec
                        df = df[[0,1,2,3,4,5]]
                        df.columns = ["timestamp","open","high","low","close","volume"]
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        out.append(df)
            except Exception:
                # skip missing days
                pass
            cur = cur.fromordinal(cur.toordinal() + 1)
        if not out:
            return None
        big = pd.concat(out, ignore_index=True)
        big = big.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        big = big[(big["timestamp"] >= pd.to_datetime(start_date)) & (big["timestamp"] <= pd.to_datetime(end_date))]
        big = big.set_index("timestamp")
        return big


