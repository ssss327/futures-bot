"""
Qudo SMC Strategy Implementation
Based on: HTF context (4H) → MTF liquidity grab + BOS (15m) → LTF confirmation (1m)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LiquidityLevel:
    """Represents a liquidity pool (e.g., Asian Low, PDH)"""
    price: float
    timestamp: datetime
    level_type: str  # "asian_low", "asian_high", "pdh", "pdl"
    touched: bool = False


@dataclass
class POI:
    """Point of Interest: OB, Breaker, or FVG"""
    poi_type: str  # "order_block", "breaker", "fvg"
    zone_high: float
    zone_low: float
    timestamp: datetime
    is_fresh: bool = True  # Not yet mitigated
    is_discount: bool = True  # Below 50% for bullish, above 50% for bearish


@dataclass
class QudoSignal:
    """Complete Qudo setup signal"""
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    htf_context: str
    liquidity_grabbed: str
    bos_confirmed: bool
    poi: POI
    ltf_confirmed: bool
    confidence: float


class QudoSMCStrategy:
    """
    Implements the full Qudo SMC strategy:
    1. HTF (4H) Order Flow context
    2. MTF (15m) Liquidity grab + BOS
    3. MTF (15m) Fresh POI in discount/premium
    4. LTF (1m) CHoCH confirmation
    """

    def __init__(self):
        self.liquidity_lookback = 20  # bars to scan for liquidity levels

    # ==================== STEP 1: HTF CONTEXT (4H) ====================
    
    def detect_htf_order_flow(self, df_4h: pd.DataFrame) -> str:
        """
        Determine 4H Order Flow direction.
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"
        
        Logic: Price moving from old lows → old highs (bullish) or vice versa
        """
        if df_4h is None or len(df_4h) < 50:
            return "NEUTRAL"
        
        df = df_4h.copy()
        df = df.tail(50)
        
        # Detect swing highs and lows
        highs = df['high'].rolling(5, center=True).max()
        lows = df['low'].rolling(5, center=True).min()
        
        swing_highs = df[df['high'] == highs]['high'].values
        swing_lows = df[df['low'] == lows]['low'].values
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"
        
        # Check if making higher highs and higher lows (bullish)
        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]
        
        if len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]:
            if len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
                return "BULLISH"
        
        # Check if making lower lows and lower highs (bearish)
        if len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]:
            if len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[-2]:
                return "BEARISH"
        
        return "NEUTRAL"

    # ==================== STEP 2: MTF LIQUIDITY GRAB (15m) ====================
    
    def detect_liquidity_levels(self, df_15m: pd.DataFrame) -> List[LiquidityLevel]:
        """
        Identify key liquidity levels: Asian Low/High, PDL/PDH
        """
        levels = []
        
        if df_15m is None or len(df_15m) < 100:
            return levels
        
        df = df_15m.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Asian session: typically 00:00-08:00 UTC
        df['hour'] = df['timestamp'].dt.hour
        asian_mask = (df['hour'] >= 0) & (df['hour'] < 8)
        
        if asian_mask.any():
            asian_data = df[asian_mask].tail(96)  # Last 24h of Asian sessions
            if not asian_data.empty:
                asian_low = asian_data['low'].min()
                asian_high = asian_data['high'].max()
                
                levels.append(LiquidityLevel(
                    price=asian_low,
                    timestamp=asian_data['timestamp'].iloc[-1],
                    level_type="asian_low"
                ))
                levels.append(LiquidityLevel(
                    price=asian_high,
                    timestamp=asian_data['timestamp'].iloc[-1],
                    level_type="asian_high"
                ))
        
        # Previous Day Low/High
        df['date'] = df['timestamp'].dt.date
        if len(df['date'].unique()) >= 2:
            prev_day = sorted(df['date'].unique())[-2]
            prev_day_data = df[df['date'] == prev_day]
            
            if not prev_day_data.empty:
                levels.append(LiquidityLevel(
                    price=prev_day_data['low'].min(),
                    timestamp=prev_day_data['timestamp'].iloc[-1],
                    level_type="pdl"
                ))
                levels.append(LiquidityLevel(
                    price=prev_day_data['high'].max(),
                    timestamp=prev_day_data['timestamp'].iloc[-1],
                    level_type="pdh"
                ))
        
        return levels

    def check_liquidity_grab(self, df_15m: pd.DataFrame, levels: List[LiquidityLevel], 
                            direction: str) -> Optional[LiquidityLevel]:
        """
        Check if price grabbed liquidity from a key level.
        For BUY: look for grab below (Asian Low, PDL)
        For SELL: look for grab above (Asian High, PDH)
        """
        if df_15m is None or len(df_15m) < 10:
            return None
        
        recent = df_15m.tail(10)
        
        for level in levels:
            tolerance = 0.001  # 0.1% tolerance
            
            if direction == "BUY":
                # Looking for downside liquidity grab
                if level.level_type in ["asian_low", "pdl"]:
                    # Price swept below the level
                    if any(recent['low'] <= level.price * (1 + tolerance)):
                        return level
            
            elif direction == "SELL":
                # Looking for upside liquidity grab
                if level.level_type in ["asian_high", "pdh"]:
                    # Price swept above the level
                    if any(recent['high'] >= level.price * (1 - tolerance)):
                        return level
        
        return None

    # ==================== STEP 3: MTF BOS DETECTION (15m) ====================
    
    def detect_bos(self, df_15m: pd.DataFrame, direction: str) -> bool:
        """
        Detect Break of Structure after liquidity grab.
        BUY: Price impulsively breaks above recent swing high
        SELL: Price impulsively breaks below recent swing low
        """
        if df_15m is None or len(df_15m) < 20:
            return False
        
        df = df_15m.tail(20).copy()
        
        # Find swing points
        highs = df['high'].rolling(3, center=True).max()
        lows = df['low'].rolling(3, center=True).min()
        
        swing_highs = df[df['high'] == highs]['high'].values
        swing_lows = df[df['low'] == lows]['low'].values
        
        if direction == "BUY":
            # Check if recent close broke above last swing high
            if len(swing_highs) >= 2:
                last_high = swing_highs[-2]
                current_close = df['close'].iloc[-1]
                return current_close > last_high
        
        elif direction == "SELL":
            # Check if recent close broke below last swing low
            if len(swing_lows) >= 2:
                last_low = swing_lows[-2]
                current_close = df['close'].iloc[-1]
                return current_close < last_low
        
        return False

    # ==================== STEP 4: MTF POI DETECTION (15m) ====================
    
    def find_poi(self, df_15m: pd.DataFrame, direction: str) -> Optional[POI]:
        """
        Find fresh POI (Order Block, Breaker, or FVG) in discount/premium zone.
        BUY: Discount zone (below 50% of impulse)
        SELL: Premium zone (above 50% of impulse)
        """
        if df_15m is None or len(df_15m) < 30:
            return None
        
        df = df_15m.tail(30).copy()
        
        # Find the impulse move (last strong directional candle sequence)
        impulse_start_idx = None
        impulse_end_idx = len(df) - 1
        
        for i in range(len(df) - 1, max(0, len(df) - 15), -1):
            if direction == "BUY":
                if df.iloc[i]['close'] > df.iloc[i]['open']:  # Bullish candle
                    impulse_start_idx = i
                else:
                    break
            elif direction == "SELL":
                if df.iloc[i]['close'] < df.iloc[i]['open']:  # Bearish candle
                    impulse_start_idx = i
                else:
                    break
        
        if impulse_start_idx is None:
            return None
        
        impulse_low = df.iloc[impulse_start_idx:impulse_end_idx + 1]['low'].min()
        impulse_high = df.iloc[impulse_start_idx:impulse_end_idx + 1]['high'].max()
        midpoint = (impulse_high + impulse_low) / 2
        
        # Find Order Block (last bearish candle before bullish move for BUY)
        if direction == "BUY":
            for i in range(impulse_start_idx - 1, max(0, impulse_start_idx - 10), -1):
                candle = df.iloc[i]
                if candle['close'] < candle['open']:  # Bearish OB
                    ob_high = candle['high']
                    ob_low = candle['low']
                    
                    # Check if in discount zone
                    ob_mid = (ob_high + ob_low) / 2
                    if ob_mid < midpoint:
                        return POI(
                            poi_type="order_block",
                            zone_high=ob_high,
                            zone_low=ob_low,
                            timestamp=candle['timestamp'],
                            is_fresh=True,
                            is_discount=True
                        )
        
        elif direction == "SELL":
            for i in range(impulse_start_idx - 1, max(0, impulse_start_idx - 10), -1):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:  # Bullish OB
                    ob_high = candle['high']
                    ob_low = candle['low']
                    
                    # Check if in premium zone
                    ob_mid = (ob_high + ob_low) / 2
                    if ob_mid > midpoint:
                        return POI(
                            poi_type="order_block",
                            zone_high=ob_high,
                            zone_low=ob_low,
                            timestamp=candle['timestamp'],
                            is_fresh=True,
                            is_discount=False
                        )
        
        return None

    # ==================== STEP 5: LTF CONFIRMATION (1m) ====================
    
    def check_ltf_choch(self, df_1m: pd.DataFrame, poi: POI, direction: str) -> bool:
        """
        Check for CHoCH (Change of Character) on 1m after price touches POI.
        BUY: Price touches POI zone, then breaks structure upward
        SELL: Price touches POI zone, then breaks structure downward
        """
        if df_1m is None or len(df_1m) < 10 or poi is None:
            return False
        
        df = df_1m.tail(20).copy()
        
        # Check if price recently touched POI zone
        touched = False
        touch_idx = None
        for i in range(len(df)):
            low = df.iloc[i]['low']
            high = df.iloc[i]['high']
            
            if low <= poi.zone_high and high >= poi.zone_low:
                touched = True
                touch_idx = i
                break
        
        if not touched or touch_idx is None:
            return False
        
        # After touch, look for micro BOS (CHoCH)
        post_touch = df.iloc[touch_idx:]
        
        if len(post_touch) < 5:
            return False
        
        if direction == "BUY":
            # Look for upward break
            highs = post_touch['high'].rolling(2).max()
            swing_high = highs.iloc[-3] if len(highs) >= 3 else post_touch['high'].iloc[0]
            current_close = post_touch['close'].iloc[-1]
            return current_close > swing_high
        
        elif direction == "SELL":
            # Look for downward break
            lows = post_touch['low'].rolling(2).min()
            swing_low = lows.iloc[-3] if len(lows) >= 3 else post_touch['low'].iloc[0]
            current_close = post_touch['close'].iloc[-1]
            return current_close < swing_low
        
        return False

    # ==================== MASTER ANALYZER ====================
    
    def analyze(self, df_4h: pd.DataFrame, df_15m: pd.DataFrame, df_1m: pd.DataFrame) -> Optional[QudoSignal]:
        """
        Full Qudo strategy analysis across all timeframes.
        Returns signal if all conditions are met, None otherwise.
        """
        # Step 1: HTF Context
        htf_flow = self.detect_htf_order_flow(df_4h)
        
        if htf_flow == "NEUTRAL":
            return None
        
        # Determine trade direction based on HTF
        direction = "BUY" if htf_flow == "BULLISH" else "SELL"
        
        # Step 2: MTF Liquidity Grab
        liquidity_levels = self.detect_liquidity_levels(df_15m)
        grabbed_level = self.check_liquidity_grab(df_15m, liquidity_levels, direction)
        
        if grabbed_level is None:
            return None
        
        # Step 3: MTF BOS after liquidity grab
        bos_confirmed = self.detect_bos(df_15m, direction)
        
        if not bos_confirmed:
            return None
        
        # Step 4: MTF POI in discount/premium
        poi = self.find_poi(df_15m, direction)
        
        if poi is None:
            return None
        
        # Step 5: LTF CHoCH confirmation
        ltf_confirmed = self.check_ltf_choch(df_1m, poi, direction)
        
        if not ltf_confirmed:
            return None
        
        # All conditions met - generate signal
        current_price = float(df_1m['close'].iloc[-1])
        
        # Calculate SL and TP
        if direction == "BUY":
            stop_loss = poi.zone_low * 0.999  # Below POI
            # Target next liquidity above
            next_liq = max([lv.price for lv in liquidity_levels if lv.level_type in ["asian_high", "pdh"]], default=current_price * 1.02)
            take_profit = next_liq
        else:
            stop_loss = poi.zone_high * 1.001  # Above POI
            # Target next liquidity below
            next_liq = min([lv.price for lv in liquidity_levels if lv.level_type in ["asian_low", "pdl"]], default=current_price * 0.98)
            take_profit = next_liq
        
        # Calculate confidence
        confidence = 0.85  # Base confidence for full setup
        
        return QudoSignal(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            htf_context=htf_flow,
            liquidity_grabbed=grabbed_level.level_type,
            bos_confirmed=bos_confirmed,
            poi=poi,
            ltf_confirmed=ltf_confirmed,
            confidence=confidence
        )

