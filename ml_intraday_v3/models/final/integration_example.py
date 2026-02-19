# Example: Load and use production model

import pickle
import pandas as pd
import numpy as np

# Load model
with open('ml_intraday_v3/models/final/batch1_exp_00159_production_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
feature_cols = bundle['feature_columns']

# Prepare live data (same features as training)
def prepare_features(df):
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

# Generate predictions
df_live = prepare_features(df_live)
X_live = df_live[feature_cols].values

predictions = model.predict_proba(X_live)
df_live['p_stop'] = predictions[:, 0]
df_live['p_target'] = predictions[:, 1]

# Trading signals
df_live['signal'] = df_live['p_target'] > 0.55

print(f"Signals generated: {df_live['signal'].sum()} / {len(df_live)}")
