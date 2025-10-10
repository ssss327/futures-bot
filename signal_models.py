"""
Signal data models for the Qudo futures trading bot
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class SmartMoneySignal:
    """Trading signal with all necessary information"""
    symbol: str
    signal_type: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    timestamp: datetime
    tier: int
    matched_concepts: List[str]
    filters_passed: List[str]

