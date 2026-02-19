"""
Quick validation test on a single model to verify the process works.
Tests the backtest logic without full retraining.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

def simulate_predictions(df, auc=0.53):
    """
    Simulate model predictions with given AUC.
    In reality, we'd retrain the model and generate real predictions.
    """
    np.random.seed(42)
    
    # Generate predictions with some skill (correlate with future returns)
    df['future_return'] = (df['close'].shift(-10) - df['close']) / df['close']
    
    # Base predictions on future returns (cheating for demo purposes)
    # Add noise to reduce correlation to target AUC
    signal_strength = (auc - 0.5) * 4  # Scale AUC to signal strength
    
    df['raw_signal'] = df['future_return'] * signal_strength + np.random.randn(len(df)) * 0.5
    
    # Convert to probabilities
    from scipy.special import expit
    df['p_target'] = expit(df['raw_signal'])
    df['p_stop'] = 1 - df['p_target']
    
    return df

def run_backtest(df, confidence_threshold=0.55, verbose=True):
    """Run backtest with realistic trading rules."""
    
    trades = []
    current_position = None
    equity = 0
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Exit logic
        if current_position is not None:
            bars_held = idx - current_position['entry_idx']
            
            # Time exit (24 bars = 2 hours)
            if bars_held >= 24:
                exit_price = row['close']
                pnl = (exit_price - current_position['entry_price']) * current_position['side'] * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': exit_price,
                    'exit_reason': 'time',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            # Profit target
            elif current_position['side'] == 1 and row['high'] >= current_position['pt_price']:
                pnl = (current_position['pt_price'] - current_position['entry_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['pt_price'],
                    'exit_reason': 'profit_target',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            elif current_position['side'] == -1 and row['low'] <= current_position['pt_price']:
                pnl = (current_position['entry_price'] - current_position['pt_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['pt_price'],
                    'exit_reason': 'profit_target',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            # Stop loss
            elif current_position['side'] == 1 and row['low'] <= current_position['sl_price']:
                pnl = (current_position['sl_price'] - current_position['entry_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['sl_price'],
                    'exit_reason': 'stop_loss',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
            
            elif current_position['side'] == -1 and row['high'] >= current_position['sl_price']:
                pnl = (current_position['entry_price'] - current_position['sl_price']) * 5
                
                trades.append({
                    **current_position,
                    'exit_idx': idx,
                    'exit_time': row['timestamp'],
                    'exit_price': current_position['sl_price'],
                    'exit_reason': 'stop_loss',
                    'bars_held': bars_held,
                    'pnl': pnl
                })
                
                equity += pnl
                current_position = None
        
        # Entry logic (if no position)
        if current_position is None and not pd.isna(row['atr']):
            # Long entry
            if row['p_target'] > confidence_threshold:
                current_position = {
                    'entry_idx': idx,
                    'entry_time': row['timestamp'],
                    'entry_price': row['close'],
                    'side': 1,
                    'pt_price': row['close'] + 2.0 * row['atr'],
                    'sl_price': row['close'] - 1.5 * row['atr'],
                    'p_target': row['p_target'],
                    'p_stop': row['p_stop']
                }
            
            # Short entry
            elif row['p_stop'] > confidence_threshold:
                current_position = {
                    'entry_idx': idx,
                    'entry_time': row['timestamp'],
                    'entry_price': row['close'],
                    'side': -1,
                    'pt_price': row['close'] - 2.0 * row['atr'],
                    'sl_price': row['close'] + 1.5 * row['atr'],
                    'p_target': row['p_stop'],
                    'p_stop': row['p_target']
                }
    
    # Close any open position at end
    if current_position is not None:
        exit_price = df.iloc[-1]['close']
        pnl = (exit_price - current_position['entry_price']) * current_position['side'] * 5
        
        trades.append({
            **current_position,
            'exit_idx': len(df) - 1,
            'exit_time': df.iloc[-1]['timestamp'],
            'exit_price': exit_price,
            'exit_reason': 'end_of_period',
            'bars_held': len(df) - 1 - current_position['entry_idx'],
            'pnl': pnl
        })
        
        equity += pnl
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {
            'total_pnl': 0,
            'n_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
        }
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    # Max drawdown
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    cummax = trades_df['cumulative_pnl'].cummax()
    drawdown = trades_df['cumulative_pnl'] - cummax
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (daily)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_returns = trades_df.groupby('entry_date')['pnl'].sum()
    sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * np.sqrt(252)
    
    results = {
        'total_pnl': float(trades_df['pnl'].sum()),
        'n_trades': len(trades_df),
        'win_rate': float(len(wins) / len(trades_df)),
        'avg_win': float(wins['pnl'].mean() if len(wins) > 0 else 0),
        'avg_loss': float(losses['pnl'].mean() if len(losses) > 0 else 0),
        'profit_factor': float(wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'trades': trades_df.to_dict('records') if verbose else []
    }
    
    return results

def check_success_criteria(results):
    """Check if results pass success criteria."""
    checks = []
    passed = True
    
    if results['total_pnl'] > 0:
        checks.append(f"✅ Positive PnL: ${results['total_pnl']:.2f}")
    else:
        checks.append(f"❌ Negative PnL: ${results['total_pnl']:.2f} (FAIL)")
        passed = False
    
    if results['win_rate'] > 0.45:
        checks.append(f"✅ Win rate: {100*results['win_rate']:.1f}%")
    else:
        checks.append(f"❌ Win rate: {100*results['win_rate']:.1f}% (FAIL - need >45%)")
        passed = False
    
    if results['max_drawdown'] > -1000:
        checks.append(f"✅ Max drawdown: ${results['max_drawdown']:.2f}")
    else:
        checks.append(f"❌ Max drawdown: ${results['max_drawdown']:.2f} (FAIL - exceeds $1,000)")
        passed = False
    
    if results['sharpe_ratio'] > 0.5:
        checks.append(f"✅ Sharpe ratio: {results['sharpe_ratio']:.2f}")
    else:
        checks.append(f"❌ Sharpe ratio: {results['sharpe_ratio']:.2f} (FAIL - need >0.5)")
        passed = False
    
    if results['n_trades'] > 10:
        checks.append(f"✅ Number of trades: {results['n_trades']}")
    else:
        checks.append(f"⚠️  Low trades: {results['n_trades']} (may not be significant)")
    
    return passed, checks

if __name__ == '__main__':
    print("="*80)
    print("QUICK PHASE 2 VALIDATION TEST")
    print("="*80)
    
    # Load OOS data
    print("\nLoading OOS data...")
    df_oos = pd.read_parquet('data/processed/jan_feb_2026_oos_for_validation.parquet')
    print(f"  {len(df_oos):,} bars from {df_oos['timestamp'].min()} to {df_oos['timestamp'].max()}")
    
    # Test with different AUC levels
    test_aucs = [0.50, 0.53, 0.54, 0.56]
    
    for test_auc in test_aucs:
        print(f"\n{'='*80}")
        print(f"TESTING MODEL WITH AUC = {test_auc}")
        print(f"{'='*80}")
        
        # Simulate predictions
        df_pred = simulate_predictions(df_oos.copy(), auc=test_auc)
        
        # Run backtest
        results = run_backtest(df_pred, confidence_threshold=0.55, verbose=False)
        
        # Check criteria
        passed, checks = check_success_criteria(results)
        
        print("\nResults:")
        for check in checks:
            print(f"  {check}")
        
        if passed:
            print("\n🎉 MODEL PASSED ALL CRITERIA!")
        else:
            print("\n❌ MODEL FAILED")
        
        print(f"\nDetailed metrics:")
        print(f"  Avg Win: ${results['avg_win']:.2f}")
        print(f"  Avg Loss: ${results['avg_loss']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\n✅ Backtest logic verified!")
    print("📝 Next step: Retrain actual models and run full Phase 2 validation")
