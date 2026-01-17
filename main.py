import pandas as pd
import numpy as np
import mibian
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_ema(df, span):
    return df['close'].ewm(span=span, adjust=False).mean()

def calculate_greeks(row):
    """Calculates Greeks using Black-Scholes via mibian"""
    c = mibian.BS([row['spot_close'], row['strike'], 6.5, 5], callPrice=row['call_ltp'])
    p = mibian.BS([row['spot_close'], row['strike'], 6.5, 5], putPrice=row['put_ltp'])
    return pd.Series({
        'call_iv': c.impliedVolatility, 'put_iv': p.impliedVolatility,
        'call_delta': c.callDelta, 'put_delta': p.putDelta,
        'gamma': c.gamma, 'theta': c.callTheta, 'vega': c.vega
    })

def engineer_features(df):
    df['ema_5'] = calculate_ema(df, 5)
    df['ema_15'] = calculate_ema(df, 15)
    df['avg_iv'] = (df['call_iv'] + df['put_iv']) / 2
    df['iv_spread'] = df['call_iv'] - df['put_iv']
    df['pcr_oi'] = df['put_oi'] / df['call_oi']
    df['pcr_vol'] = df['put_vol'] / df['call_vol']
    df['futures_basis'] = (df['fut_close'] - df['close']) / df['close']
    df['spot_returns'] = df['close'].pct_change()
    df['delta_neutral_ratio'] = abs(df['call_delta']) / abs(df['put_delta'])
    df['gamma_exposure'] = df['close'] * df['gamma'] * (df['call_oi'] + df['put_oi'])
    return df.dropna()