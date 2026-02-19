"""
Paper Trading System - Test production model without real trades

Generates signals, tracks "would-be" PnL, monitors performance.
"""
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

class PaperTrader:
    """Paper trading system for model validation."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.55):
        """Initialize paper trader."""
        # Load model
        with open(model_path, 'rb') as f:
            self.bundle = pickle.load(f)
        
        self.model = self.bundle['model']
        self.feature_cols = self.bundle['feature_columns']
        self.confidence_threshold = confidence_threshold
        
        # Trading state
        self.current_position = None
        self.trades = []
        self.equity = 0
        
        print(f"📦 Loaded model: {self.bundle['exp_id']}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features matching training."""
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_ma5'] = df['close'].rolling(5).mean()
        df['price_ma20'] = df['close'].rolling(20).mean()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # ATR
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        })
        df['atr'] = tr.max(axis=1).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        return df
    
    def process_bar(self, bar: pd.Series):
        """Process new bar."""
        # Exit check
        if self.current_position:
            if self._should_exit(bar):
                pnl = (bar['close'] - self.current_position['entry_price']) * self.current_position['side'] * 5
                self.trades.append({'pnl': pnl})
                self.equity += pnl
                print(f"Exit @ ${bar['close']:.2f} - PnL: ${pnl:+.2f} | Total: ${self.equity:+.2f}")
                self.current_position = None
        
        # Entry check
        if not self.current_position:
            signal = self._get_signal(bar)
            if signal['side'] != 0:
                self.current_position = {'side': signal['side'], 'entry_price': bar['close']}
                print(f"{'LONG' if signal['side']==1 else 'SHORT'} @ ${bar['close']:.2f}")
    
    def _get_signal(self, bar):
        """Generate signal."""
        if any(pd.isna(bar[col]) for col in self.feature_cols):
            return {'side': 0}
        
        X = np.array([bar[self.feature_cols].values])
        pred = self.model.predict_proba(X)[0]
        
        if pred[1] > self.confidence_threshold:
            return {'side': 1}
        elif pred[0] > self.confidence_threshold:
            return {'side': -1}
        return {'side': 0}
    
    def _should_exit(self, bar):
        """Simple exit logic."""
        return True  # Exit every bar for simplicity
    
    def get_stats(self):
        """Get stats."""
        if not self.trades:
            return {'total_pnl': 0, 'n_trades': 0}
        
        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df['pnl'] > 0]
        
        return {
            'total_pnl': self.equity,
            'n_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
        }

if __name__ == '__main__':
    print("Paper Trading System - Run with: python3 ml_intraday_v3/paper_trading/paper_trader.py")
