"""
Phase 4: Trading Signal Generator
Generates actionable trading signals with risk management.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    predicted_change: float
    confidence: float
    strength: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[int] = None
    reasoning: str = ""


class SignalGenerator:
    """Generates trading signals with risk management."""
    
    def __init__(
        self,
        entry_threshold: float = 0.02,  # 0.02% minimum change
        stop_loss_pct: float = 0.03,  # 3% stop loss
        take_profit_pct: float = 0.05,  # 5% take profit
        min_confidence: float = 0.5,  # Minimum 50% confidence
        max_position_size_pct: float = 0.1  # Max 10% of capital
    ):
        """
        Initialize signal generator.
        
        Args:
            entry_threshold: Minimum predicted change to generate signal
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            min_confidence: Minimum confidence to generate signal
            max_position_size_pct: Maximum position size as % of capital
        """
        self.entry_threshold = entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
        self.max_position_size_pct = max_position_size_pct
    
    def generate_signal(
        self,
        prediction: float,
        confidence: float,
        current_price: float,
        capital: float = 100000.0,
        timestamp: Optional[datetime] = None
    ) -> TradingSignal:
        """
        Generate a trading signal.
        
        Args:
            prediction: Predicted price change (percentage)
            confidence: Confidence score (0-1)
            current_price: Current market price
            capital: Available capital
            timestamp: Signal timestamp
        
        Returns:
            TradingSignal object
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        signal_type = "HOLD"
        strength = 0.0
        reasoning = ""
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return TradingSignal(
                timestamp=timestamp,
                signal_type="HOLD",
                predicted_change=prediction,
                confidence=confidence,
                strength=0.0,
                reasoning=f"Low confidence ({confidence:.2f} < {self.min_confidence:.2f})"
            )
        
        # Generate BUY signal
        if prediction > self.entry_threshold:
            signal_type = "BUY"
            strength = min(1.0, (prediction / self.entry_threshold) * confidence)
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            
            # Calculate position size
            position_value = capital * self.max_position_size_pct * confidence
            position_size = int(position_value / current_price)
            
            reasoning = f"Predicted increase of {prediction:.4f}% with {confidence:.2f} confidence"
        
        # Generate SELL signal
        elif prediction < -self.entry_threshold:
            signal_type = "SELL"
            strength = min(1.0, (abs(prediction) / self.entry_threshold) * confidence)
            
            # Calculate stop loss and take profit (inverted for short)
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
            
            # Calculate position size
            position_value = capital * self.max_position_size_pct * confidence
            position_size = int(position_value / current_price)
            
            reasoning = f"Predicted decrease of {abs(prediction):.4f}% with {confidence:.2f} confidence"
        
        else:
            return TradingSignal(
                timestamp=timestamp,
                signal_type="HOLD",
                predicted_change=prediction,
                confidence=confidence,
                strength=0.0,
                reasoning=f"Predicted change ({prediction:.4f}%) below threshold ({self.entry_threshold:.4f}%)"
            )
        
        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            predicted_change=prediction,
            confidence=confidence,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=reasoning
        )
    
    def filter_signals(
        self,
        signals: List[TradingSignal],
        min_strength: float = 0.6,
        max_signals_per_day: int = 5
    ) -> List[TradingSignal]:
        """
        Filter signals based on quality criteria.
        
        Args:
            signals: List of signals to filter
            min_strength: Minimum signal strength
            max_signals_per_day: Maximum signals per day
        
        Returns:
            Filtered list of signals
        """
        # Filter by strength
        filtered = [s for s in signals if s.strength >= min_strength]
        
        # Limit per day
        signals_by_day = {}
        for signal in filtered:
            day_key = signal.timestamp.date()
            if day_key not in signals_by_day:
                signals_by_day[day_key] = []
            signals_by_day[day_key].append(signal)
        
        # Keep only strongest signals per day
        final_signals = []
        for day, day_signals in signals_by_day.items():
            sorted_signals = sorted(day_signals, key=lambda x: x.strength, reverse=True)
            final_signals.extend(sorted_signals[:max_signals_per_day])
        
        return sorted(final_signals, key=lambda x: x.timestamp)
    
    def print_signal(self, signal: TradingSignal):
        """Print signal in a formatted way."""
        print("\n" + "=" * 60)
        print(f"TRADING SIGNAL - {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Signal Type:      {signal.signal_type}")
        print(f"Predicted Change: {signal.predicted_change:.4f}%")
        print(f"Confidence:       {signal.confidence:.2%}")
        print(f"Strength:         {signal.strength:.2%}")
        
        if signal.signal_type != "HOLD":
            print(f"\nEntry Price:      ${signal.entry_price:.2f}")
            print(f"Stop Loss:        ${signal.stop_loss:.2f} ({self.stop_loss_pct:.1%})")
            print(f"Take Profit:      ${signal.take_profit:.2f} ({self.take_profit_pct:.1%})")
            print(f"Position Size:    {signal.position_size} units")
        
        print(f"\nReasoning:        {signal.reasoning}")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    generator = SignalGenerator()
    
    # Example signals
    signal1 = generator.generate_signal(
        prediction=0.05,  # 0.05% predicted increase
        confidence=0.75,
        current_price=100.0,
        capital=100000.0
    )
    
    generator.print_signal(signal1)
    
    signal2 = generator.generate_signal(
        prediction=-0.03,  # 0.03% predicted decrease
        confidence=0.60,
        current_price=100.0,
        capital=100000.0
    )
    
    generator.print_signal(signal2)

