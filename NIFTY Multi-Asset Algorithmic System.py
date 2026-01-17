import pandas as pd
import numpy as np
import mibian
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# PART 1 & 2: DATA ENGINEERING & GREEKS
# ==========================================

def calculate_greeks(row):
    """
    Calculates Black-Scholes Greeks for ATM Options.
    Risk-free rate: 6.5%. T: Days to expiry (assumed 5 for intra-week).
    """
    # mibian.BS([Price, Strike, Rate, Days], callPrice=P)
    c = mibian.BS([row['spot_close'], row['strike'], 6.5, 5], callPrice=row['call_ltp'])
    p = mibian.BS([row['spot_close'], row['strike'], 6.5, 5], putPrice=row['put_ltp'])
    
    return pd.Series({
        'call_iv': c.impliedVolatility,
        'put_iv': p.impliedVolatility,
        'call_delta': c.callDelta,
        'put_delta': p.putDelta,
        'gamma': c.gamma,
        'theta': c.callTheta,
        'vega': c.vega
    })

def engineer_features(df):
    """Task 2.1 - 2.4: Feature Construction"""
    # EMA Indicators
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    
    # Greeks (Apply BS Model)
    greeks = df.apply(calculate_greeks, axis=1)
    df = pd.concat([df, greeks], axis=1)
    
    # Derived Features
    df['avg_iv'] = (df['call_iv'] + df['put_iv']) / 2
    df['iv_spread'] = df['call_iv'] - df['put_iv']
    df['pcr_oi'] = df['put_oi'] / df['call_oi']
    df['pcr_vol'] = df['put_vol'] / df['call_vol']
    df['futures_basis'] = (df['fut_close'] - df['close']) / df['close']
    df['spot_returns'] = df['close'].pct_change()
    df['delta_neutral_ratio'] = abs(df['call_delta']) / abs(df['put_delta'])
    df['gamma_exposure'] = df['close'] * df['gamma'] * (df['call_oi'] + df['put_oi'])
    
    return df.dropna()

# ==========================================
# PART 3: REGIME DETECTION (HMM)
# ==========================================

def implement_hmm(df):
    features = ['avg_iv', 'iv_spread', 'pcr_oi', 'call_delta', 'gamma', 'vega', 'futures_basis', 'spot_returns']
    train_size = int(len(df) * 0.7)
    train_data = df[features].iloc[:train_size]
    
    # Standardize for HMM
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_data)
    
    # 3 States: +1 (Up), -1 (Down), 0 (Sideways)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    model.fit(scaled_data)
    
    full_scaled = scaler.transform(df[features])
    df['regime_raw'] = model.predict(full_scaled)
    
    # Map raw HMM states to logical regimes based on returns
    regime_map = df.groupby('regime_raw')['spot_returns'].mean().sort_values().index
    mapping = {regime_map[0]: -1, regime_map[1]: 0, regime_map[2]: 1}
    df['regime'] = df['regime_raw'].map(mapping)
    
    return df, model

# ==========================================
# PART 4 & 5: TRADING & ML ENHANCEMENT
# ==========================================

def backtest_with_ml(df):
    # Base Strategy: 5/15 EMA + Regime Filter
    df['signal'] = 0
    # Long
    df.loc[(df['ema_5'] > df['ema_15']) & (df['regime'] == 1), 'signal'] = 1
    # Short
    df.loc[(df['ema_5'] < df['ema_15']) & (df['regime'] == -1), 'signal'] = -1
    
    # ML Target: Profitability (1 if trade makes money, else 0)
    df['trade_pnl'] = df['signal'].shift(1) * df['spot_returns']
    df['is_profitable'] = (df['trade_pnl'] > 0).astype(int)
    
    # Train XGBoost Filter
    features = ['avg_iv', 'pcr_oi', 'delta_neutral_ratio', 'gamma_exposure', 'futures_basis']
    X = df[features]
    y = df['is_profitable']
    
    split = int(len(df) * 0.7)
    xgb = XGBClassifier(n_estimators=100)
    xgb.fit(X.iloc[:split], y.iloc[:split])
    
    df['ml_filter'] = xgb.predict(X)
    df['enhanced_signal'] = df['signal'] * df['ml_filter']
    
    return df

# ==========================================
# PART 6: OUTLIER ANALYSIS
# ==========================================

def analyze_outliers(df):
    trades = df[df['enhanced_signal'] != 0].copy()
    z_scores = (trades['trade_pnl'] - trades['trade_pnl'].mean()) / trades['trade_pnl'].std()
    outliers = trades[z_scores > 3]
    
    print(f"--- High Performance Trade Summary ---")
    print(f"Total Outliers: {len(outliers)}")
    print(f"Dominant Regime for Outliers: {outliers['regime'].mode()[0]}")
    
    # Correlation of Outliers
    plt.figure(figsize=(10,6))
    sns.heatmap(outliers[['trade_pnl', 'avg_iv', 'gamma_exposure', 'pcr_oi']].corr(), annot=True)
    plt.title("Feature Correlation in Outlier Trades")
    plt.show()

