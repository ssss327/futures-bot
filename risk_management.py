"""
Enhanced Risk Management and Position Sizing Module
Implements Kelly Criterion and multi-level take profit system
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from config import Config

@dataclass
class RiskParameters:
    """Risk management parameters for a trade"""
    entry_price: float
    stop_loss: float
    take_profit_1: float  # 1:1 RR
    take_profit_2: float  # 1:3 RR
    trail_start: float    # Break-even + 1 ATR
    position_size: float  # As percentage of account
    leverage: int
    max_risk_percent: float

@dataclass
class TradeAllocation:
    """Position allocation across different take profit levels"""
    tp1_allocation: float = 0.40  # 40% at 1:1
    tp2_allocation: float = 0.30  # 30% at 1:3
    trail_allocation: float = 0.30  # 30% trailing

class RiskManager:
    def __init__(self):
        self.kelly_fraction = 0.25  # Conservative Kelly fraction
        self.max_position_risk = 0.02  # Maximum 2% risk per trade
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
        
        Returns:
            Kelly percentage for position sizing
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.01  # Minimum position size
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_f = (b * p - q) / b
        
        # Apply fractional Kelly and cap at maximum risk
        fractional_kelly = kelly_f * self.kelly_fraction
        
        return min(max(fractional_kelly, 0.005), self.max_position_risk)  # Between 0.5% and 2%
    
    def calculate_atr_based_stops(self, df: pd.DataFrame, signal_type: str, 
                                 entry_price: float, order_block_level: float) -> Tuple[float, float]:
        """
        Calculate ATR-based stop loss and take profit levels
        
        Args:
            df: Price data
            signal_type: 'BUY' or 'SELL'
            entry_price: Entry price
            order_block_level: Order block level for stop calculation
        
        Returns:
            Tuple of (stop_loss, atr20)
        """
        # Calculate ATR20
        df = df.copy()
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close'].shift(1)) if len(df) > 1 else x['high'] - x['low'],
                abs(x['low'] - x['close'].shift(1)) if len(df) > 1 else x['high'] - x['low']
            ), axis=1
        )
        atr20 = df['tr'].rolling(window=20).mean().iloc[-1]
        
        # Stop loss beyond order block edge + 1.2 * ATR20
        if signal_type == 'BUY':
            stop_loss = order_block_level - (1.2 * atr20)
        else:
            stop_loss = order_block_level + (1.2 * atr20)
        
        return stop_loss, atr20
    
    def calculate_position_levels(self, signal_type: str, entry_price: float, 
                                 stop_loss: float, tier: int) -> RiskParameters:
        """
        Calculate comprehensive position sizing and risk parameters
        
        Args:
            signal_type: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss: Stop loss level
            tier: Signal tier (1=high, 2=medium, 3=low)
        
        Returns:
            RiskParameters object with all levels
        """
        risk_amount = abs(entry_price - stop_loss)
        
        if signal_type == 'BUY':
            # Take Profit levels
            tp1 = entry_price + (risk_amount * 1.0)  # 1:1 RR
            tp2 = entry_price + (risk_amount * 3.0)  # 1:3 RR
            trail_start = entry_price + (risk_amount * 0.5)  # Break-even + 0.5R
        else:
            # Take Profit levels
            tp1 = entry_price - (risk_amount * 1.0)  # 1:1 RR
            tp2 = entry_price - (risk_amount * 3.0)  # 1:3 RR
            trail_start = entry_price - (risk_amount * 0.5)  # Break-even + 0.5R
        
        # Position sizing based on confidence and Kelly
        base_kelly = self.calculate_kelly_criterion(
            win_rate=0.65,  # Estimated from 95% accuracy target
            avg_win=2.0,    # Average 2R win
            avg_loss=1.0    # 1R loss
        )
        
        # Adjust position size based on tier (lower tier number = higher confidence)
        tier_multipliers = {1: 1.2, 2: 1.0, 3: 0.8}  # Tier 1 gets 20% more, Tier 3 gets 20% less
        tier_multiplier = tier_multipliers.get(tier, 0.8)
        position_size = base_kelly * tier_multiplier
        
        # Calculate leverage based on tier (higher tier gets more leverage)
        tier_leverage = {1: Config.LEVERAGE_MAX, 2: max(3, Config.LEVERAGE_MAX // 2), 3: 2}
        leverage = tier_leverage.get(tier, 2)
        
        return RiskParameters(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            trail_start=trail_start,
            position_size=position_size,
            leverage=leverage,
            max_risk_percent=self.max_position_risk
        )
    
    def validate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                  take_profit: float) -> float:
        """
        Validate and calculate risk-reward ratio
        
        Returns:
            Risk-reward ratio (reward/risk)
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def calculate_slippage_impact(self, entry_price: float, signal_type: str) -> float:
        """
        Calculate slippage impact on entry price
        
        Returns:
            Adjusted entry price with slippage
        """
        slippage = Config.SLIPPAGE_PERCENT
        
        if signal_type == 'BUY':
            return entry_price * (1 + slippage)
        else:
            return entry_price * (1 - slippage)
    
    def calculate_total_fees(self, position_value: float) -> float:
        """
        Calculate total trading fees (entry + exit)
        
        Returns:
            Total fees as percentage of position
        """
        return Config.MAKER_TAKER_FEES * 2  # Entry + exit
    
    def get_trade_allocation(self) -> TradeAllocation:
        """
        Get position allocation across take profit levels
        
        Returns:
            TradeAllocation object
        """
        return TradeAllocation()
    
    def calculate_portfolio_heat(self, active_positions: List[Dict]) -> float:
        """
        Calculate total portfolio heat (risk exposure)
        
        Args:
            active_positions: List of active position dictionaries
        
        Returns:
            Total portfolio risk as percentage
        """
        total_risk = 0.0
        
        for position in active_positions:
            position_risk = position.get('risk_percent', 0)
            total_risk += position_risk
        
        return total_risk
    
    def check_correlation_risk(self, new_symbol: str, active_symbols: List[str]) -> bool:
        """
        Check for correlation risk between symbols
        
        Args:
            new_symbol: Symbol for new position
            active_symbols: List of currently active symbols
        
        Returns:
            True if correlation risk is acceptable
        """
        # Simple correlation check - avoid multiple positions in same base asset
        new_base = new_symbol.split('/')[0]
        
        for symbol in active_symbols:
            existing_base = symbol.split('/')[0]
            if new_base == existing_base:
                return False  # Same base asset
        
        # Check for highly correlated pairs
        correlated_pairs = {
            'BTC': ['BCH', 'BSV'],
            'ETH': ['ETC', 'LTC'],
            'ADA': ['DOT', 'ALGO'],
        }
        
        for base, correlated in correlated_pairs.items():
            if new_base == base:
                for symbol in active_symbols:
                    if symbol.split('/')[0] in correlated:
                        return False
        
        return True
    
    def generate_risk_report(self, risk_params: RiskParameters, symbol: str) -> Dict:
        """
        Generate comprehensive risk report for a trade
        
        Returns:
            Dictionary with risk analysis
        """
        rr_ratio = self.validate_risk_reward_ratio(
            risk_params.entry_price, 
            risk_params.stop_loss, 
            risk_params.take_profit_2
        )
        
        allocation = self.get_trade_allocation()
        
        return {
            'symbol': symbol,
            'risk_reward_ratio': rr_ratio,
            'position_size_percent': risk_params.position_size * 100,
            'max_loss_percent': risk_params.max_risk_percent * 100,
            'leverage': risk_params.leverage,
            'allocation': {
                'tp1_percent': allocation.tp1_allocation * 100,
                'tp2_percent': allocation.tp2_allocation * 100,
                'trail_percent': allocation.trail_allocation * 100
            },
            'levels': {
                'entry': risk_params.entry_price,
                'stop_loss': risk_params.stop_loss,
                'take_profit_1': risk_params.take_profit_1,
                'take_profit_2': risk_params.take_profit_2,
                'trail_start': risk_params.trail_start
            },
            'fees_impact': self.calculate_total_fees(10000) * 100,  # Fees on $10k position
            'slippage_impact': Config.SLIPPAGE_PERCENT * 100
        }
