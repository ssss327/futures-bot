import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def detect_trend_and_consolidation(df_1d: pd.DataFrame, df_4h: pd.DataFrame) -> tuple[str, bool]:
    """
    Returns (trend, is_consolidating)
    - trend: "UP", "DOWN", or "SIDE"
    - is_consolidating: True if market is ranging/unclear
    Heuristics:
      - Trend by EMA21 slope alignment on 1D and 4H
      - Consolidation if 4H ATR% low and EMA21 flat (small slope)
    """
    if df_1d is None or df_4h is None or len(df_1d) < 50 or len(df_4h) < 50:
        return "SIDE", True

    d1 = df_1d.copy()
    h4 = df_4h.copy()
    d1_close = d1["close"].astype(float)
    h4_close = h4["close"].astype(float)
    d1_ema = ema(d1_close, 21)
    h4_ema = ema(h4_close, 21)

    def slope(x: pd.Series, lookback: int = 10) -> float:
        if len(x) < lookback + 1:
            return 0.0
        a = float(x.iloc[-1] - x.iloc[-1 - lookback]) / max(1e-9, abs(float(x.iloc[-1 - lookback])))
        return a

    d1_slope = slope(d1_ema, 10)
    h4_slope = slope(h4_ema, 10)

    # Determine trend by slope signs
    if d1_slope > 0 and h4_slope > 0:
        trend = "UP"
    elif d1_slope < 0 and h4_slope < 0:
        trend = "DOWN"
    else:
        trend = "SIDE"

    # Consolidation on 4H: low ATR% and flat EMA slope
    h4_atr = compute_atr(h4, 14)
    last_close = float(h4_close.iloc[-1])
    atr_pct = float(h4_atr.iloc[-1] / max(1e-9, last_close)) if not h4_atr.empty else 0.0
    flat = abs(h4_slope) < 0.01  # ~1% over 10 bars
    low_vol = atr_pct < 0.01     # <1% ATR
    is_consolidating = (trend == "SIDE") or (flat and low_vol)

    return trend, is_consolidating


