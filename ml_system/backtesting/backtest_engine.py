"""
Phase 4: Backtesting Engine
Simulates trading strategies on historical data to evaluate model performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade type enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    trade_type: TradeType
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_return_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,  # 0.03% per trade
        slippage: float = 0.0001,  # 0.01% slippage
        position_size_pct: float = 0.1  # 10% of capital per trade
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate per trade (as decimal)
            slippage: Slippage rate (as decimal)
            position_size_pct: Position size as percentage of capital
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_size_pct = position_size_pct
    
    def calculate_position_size(
        self,
        current_capital: float,
        entry_price: float,
        confidence: float = 1.0
    ) -> int:
        """
        Calculate position size based on capital and confidence.
        
        Args:
            current_capital: Current available capital
            entry_price: Entry price
            confidence: Confidence score (0-1), adjusts position size
        
        Returns:
            Number of shares/contracts
        """
        # Base position size
        position_value = current_capital * self.position_size_pct * confidence
        
        # Adjust for confidence (higher confidence = larger position)
        adjusted_value = position_value * confidence
        
        # Calculate quantity (assuming lot size of 1 for options)
        quantity = int(adjusted_value / entry_price)
        
        return max(1, quantity)  # At least 1 unit
    
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Original price
            is_buy: True for buy, False for sell
        
        Returns:
            Price with slippage
        """
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """Calculate commission for a trade."""
        trade_value = price * quantity
        return trade_value * self.commission_rate
    
    def backtest_strategy(
        self,
        predictions: pd.Series,
        actual_prices: pd.Series,
        timestamps: pd.Series,
        entry_threshold: float = 0.02,  # 0.02% predicted change to enter
        exit_threshold: float = 0.05,  # 0.05% profit to exit
        stop_loss_pct: float = 0.03,  # 3% stop loss
        max_holding_periods: int = 30,  # Max periods to hold
        confidence_scores: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Backtest a trading strategy based on model predictions.
        
        Args:
            predictions: Predicted price changes (percentage)
            actual_prices: Actual prices at each timestamp
            timestamps: Timestamps for each data point
            entry_threshold: Minimum predicted change to enter trade
            exit_threshold: Profit target (percentage)
            stop_loss_pct: Stop loss percentage
            max_holding_periods: Maximum periods to hold a position
            confidence_scores: Optional confidence scores for predictions
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info("Starting backtest...")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Entry threshold: {entry_threshold:.4f}%")
        logger.info(f"Exit threshold: {exit_threshold:.4f}%")
        logger.info(f"Stop loss: {stop_loss_pct:.2f}%")
        
        capital = self.initial_capital
        trades = []
        current_position: Optional[Trade] = None
        equity_curve = []
        
        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]
            predicted_change = predictions.iloc[i]
            timestamp = timestamps.iloc[i]
            
            # Skip if prediction is NaN
            if pd.isna(predicted_change):
                equity_curve.append(capital)
                continue
            
            # Get confidence (default to 1.0 if not provided)
            confidence = confidence_scores.iloc[i] if confidence_scores is not None else 1.0
            if pd.isna(confidence):
                confidence = 1.0
            
            # Check if we have an open position
            if current_position is not None:
                # Calculate current P&L
                if current_position.trade_type == TradeType.LONG:
                    current_pnl_pct = ((current_price - current_position.entry_price) / current_position.entry_price) * 100
                else:  # SHORT
                    current_pnl_pct = ((current_position.entry_price - current_price) / current_position.entry_price) * 100
                
                # Check exit conditions
                should_exit = False
                exit_reason = None
                
                # Profit target
                if current_pnl_pct >= exit_threshold:
                    should_exit = True
                    exit_reason = f"Profit target ({exit_threshold:.2f}%)"
                
                # Stop loss
                elif current_pnl_pct <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = f"Stop loss ({stop_loss_pct:.2f}%)"
                
                # Max holding period
                elif i - len([t for t in trades if t.exit_time is None]) >= max_holding_periods:
                    should_exit = True
                    exit_reason = "Max holding period"
                
                # Exit if conditions met
                if should_exit:
                    exit_price = self.apply_slippage(current_price, is_buy=False)
                    commission = self.calculate_commission(exit_price, current_position.quantity)
                    
                    # Calculate P&L
                    if current_position.trade_type == TradeType.LONG:
                        pnl = (exit_price - current_position.entry_price) * current_position.quantity - commission * 2
                    else:
                        pnl = (current_position.entry_price - exit_price) * current_position.quantity - commission * 2
                    
                    pnl_pct = (pnl / (current_position.entry_price * current_position.quantity)) * 100
                    
                    # Update trade
                    current_position.exit_time = timestamp
                    current_position.exit_price = exit_price
                    current_position.pnl = pnl
                    current_position.pnl_pct = pnl_pct
                    current_position.exit_reason = exit_reason
                    
                    trades.append(current_position)
                    
                    # Update capital
                    capital += pnl
                    current_position = None
            
            # Check entry conditions (only if no open position)
            if current_position is None:
                # Enter long if predicted increase is significant
                if predicted_change > entry_threshold:
                    entry_price = self.apply_slippage(current_price, is_buy=True)
                    quantity = self.calculate_position_size(capital, entry_price, confidence)
                    commission = self.calculate_commission(entry_price, quantity)
                    
                    # Check if we have enough capital
                    trade_cost = entry_price * quantity + commission
                    if trade_cost <= capital:
                        current_position = Trade(
                            entry_time=timestamp,
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            quantity=quantity,
                            trade_type=TradeType.LONG,
                            exit_reason=None
                        )
                        capital -= trade_cost
                
                # Enter short if predicted decrease is significant
                elif predicted_change < -entry_threshold:
                    entry_price = self.apply_slippage(current_price, is_buy=False)
                    quantity = self.calculate_position_size(capital, entry_price, confidence)
                    commission = self.calculate_commission(entry_price, quantity)
                    
                    # Check if we have enough capital
                    trade_cost = entry_price * quantity + commission
                    if trade_cost <= capital:
                        current_position = Trade(
                            entry_time=timestamp,
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            quantity=quantity,
                            trade_type=TradeType.SHORT,
                            exit_reason=None
                        )
                        capital -= trade_cost
            
            # Record equity
            if current_position is not None:
                # Calculate unrealized P&L
                if current_position.trade_type == TradeType.LONG:
                    unrealized_pnl = (current_price - current_position.entry_price) * current_position.quantity
                else:
                    unrealized_pnl = (current_position.entry_price - current_price) * current_position.quantity
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)
        
        # Close any remaining open positions
        if current_position is not None:
            final_price = actual_prices.iloc[-1]
            exit_price = self.apply_slippage(final_price, is_buy=False)
            commission = self.calculate_commission(exit_price, current_position.quantity)
            
            if current_position.trade_type == TradeType.LONG:
                pnl = (exit_price - current_position.entry_price) * current_position.quantity - commission * 2
            else:
                pnl = (current_position.entry_price - exit_price) * current_position.quantity - commission * 2
            
            pnl_pct = (pnl / (current_position.entry_price * current_position.quantity)) * 100
            
            current_position.exit_time = timestamps.iloc[-1]
            current_position.exit_price = exit_price
            current_position.pnl = pnl
            current_position.pnl_pct = pnl_pct
            current_position.exit_reason = "End of backtest"
            
            trades.append(current_position)
            capital += pnl
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, timestamps)
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        timestamps: pd.Series
    ) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not trades:
            logger.warning("No trades executed in backtest")
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                total_pnl=0, total_return_pct=0, win_rate=0,
                avg_win=0, avg_loss=0, profit_factor=0,
                max_drawdown=0, max_drawdown_pct=0,
                sharpe_ratio=0, sortino_ratio=0,
                trades=[], equity_curve=pd.Series()
            )
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        total_return_pct = ((equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown calculation
        equity_series = pd.Series(equity_curve, index=timestamps[:len(equity_curve)])
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.iloc[drawdown.idxmin()]) * 100 if drawdown.idxmin() is not None else 0
        
        # Sharpe ratio (annualized, assuming daily returns)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            trades=trades,
            equity_curve=equity_series
        )
    
    def print_results(self, result: BacktestResult):
        """Print backtest results in a formatted way."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"Final Capital:       ${self.initial_capital + result.total_pnl:,.2f}")
        print(f"Total Return:        {result.total_return_pct:.2f}%")
        print(f"Total P&L:           ${result.total_pnl:,.2f}")
        print()
        print(f"Total Trades:        {result.total_trades}")
        print(f"Winning Trades:      {result.winning_trades}")
        print(f"Losing Trades:       {result.losing_trades}")
        print(f"Win Rate:            {result.win_rate * 100:.2f}%")
        print()
        print(f"Average Win:          ${result.avg_win:,.2f}")
        print(f"Average Loss:         ${result.avg_loss:,.2f}")
        print(f"Profit Factor:       {result.profit_factor:.2f}")
        print()
        print(f"Max Drawdown:         ${result.max_drawdown:,.2f}")
        print(f"Max Drawdown %:       {result.max_drawdown_pct:.2f}%")
        print()
        print(f"Sharpe Ratio:        {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {result.sortino_ratio:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    from ml_system.training.train_baseline import BaselineTrainer
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    trainer = BaselineTrainer()
    backtester = BacktestEngine(initial_capital=100000.0)
    
    try:
        # Get data and train model
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        features_df = engineer.engineer_all_features(raw_data)
        
        # Train model
        results = trainer.train_all_baselines(features_df, target_col='price_change_pct', task='regression')
        
        # Get best model predictions
        best_name, best_result = trainer.get_best_model('test_r2')
        
        # Run backtest (simplified example)
        print("Backtest example would run here...")
    
    finally:
        extractor.close()

