import pandas as pd
import numpy as np
import mibian
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def merge_and_clean():
    print("Merging datasets and cleaning...")
    spot = pd.read_csv("nifty_spot_5min.csv", parse_dates=['timestamp'])
    fut = pd.read_csv("nifty_futures_5min.csv", parse_dates=['timestamp'])
    opt = pd.read_csv("nifty_options_5min.csv", parse_dates=['timestamp'])
    
    df = pd.merge(spot, fut, on='timestamp', suffixes=('', '_fut'))
    df = pd.merge(df, opt, on='timestamp')
    
    df.rename(columns={'close': 'spot_close', 'close_fut': 'fut_close', 'oi': 'fut_oi'}, inplace=True)
    df.dropna(inplace=True)
    return df

def calculate_greeks_vectorized(df):
    print("Calculating Options Greeks (BS Model)...")
    def get_bs(row):
        c = mibian.BS([row['spot_close'], row['strike'], 6.5, 5], callPrice=row['call_ltp'])
        return pd.Series([c.impliedVolatility, c.callDelta, c.gamma, c.vega])

    df[['iv', 'delta', 'gamma', 'vega']] = df.apply(get_bs, axis=1)
    return df

def add_derived_features(df):
    df['ema_5'] = df['spot_close'].ewm(span=5, adjust=False).mean()
    df['ema_15'] = df['spot_close'].ewm(span=15, adjust=False).mean()
    df['pcr_oi'] = df['put_oi'] / df['call_oi']
    df['futures_basis'] = (df['fut_close'] - df['spot_close']) / df['spot_close']
    df['returns'] = df['spot_close'].pct_change()
    return df.dropna()

def apply_regime_detection(df):
    print("Training HMM for Regime Detection...")
    features = ['iv', 'pcr_oi', 'gamma', 'futures_basis', 'returns']
    train_size = int(len(df) * 0.7)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    hmm.fit(scaled_features[:train_size])
    
    df['regime_raw'] = hmm.predict(scaled_features)
    
    order = df.groupby('regime_raw')['returns'].mean().sort_values().index
    mapping = {order[0]: -1, order[1]: 0, order[2]: 1}
    df['regime'] = df['regime_raw'].map(mapping)
    return df

def backtest_system(df):
    print("Executing Strategy & ML Enhancement...")
    df['signal'] = 0
    df.loc[(df['ema_5'] > df['ema_15']) & (df['regime'] == 1), 'signal'] = 1 # Long
    df.loc[(df['ema_5'] < df['ema_15']) & (df['regime'] == -1), 'signal'] = -1 # Short
    
    df['target'] = (df['returns'].shift(-3).rolling(3).sum() > 0).astype(int)
    
    ml_features = ['iv', 'pcr_oi', 'delta', 'gamma', 'futures_basis', 'regime']
    X = df[ml_features]
    y = df['target']
    
    split = int(len(df) * 0.7)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X[:split], y[:split])
    
    df['ml_prob'] = model.predict_proba(X)[:, 1]
    df['final_signal'] = np.where(df['ml_prob'] > 0.5, df['signal'], 0)
    
    df['strategy_ret'] = df['final_signal'].shift(1) * df['returns']
    df['cum_ret'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    return df

data = merge_and_clean()
data = calculate_greeks_vectorized(data)
data = add_derived_features(data)
data = apply_regime_detection(data)
data = backtest_system(data)

plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['cum_ret'], label='ML Enhanced Strategy')
plt.title("Equity Curve: HMM + EMA + XGBoost Filter")
plt.legend()
plt.show()

print("Project Completed. Final features saved to nifty_features_5min.csv")
data.to_csv("nifty_features_5min.csv", index=False)