import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_nifty_data(filename_prefix="nifty", periods=25000):
    """
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=365)
    timestamps = [start_date + timedelta(minutes=5 * i) for i in range(periods)]
    
    returns = np.random.normal(0.0001, 0.01, periods)
    spot_price = 22000 * np.exp(np.cumsum(returns) / 100)
    
    basis = np.random.normal(50, 10, periods) 
    fut_price = spot_price + basis
    fut_oi = np.random.randint(100000, 500000, periods)
    
    atm_strike = (np.round(spot_price / 50) * 50).astype(int)
    iv_base = np.random.uniform(12, 18, periods)
    
    call_ltp = np.random.uniform(100, 300, periods)
    put_ltp = np.random.uniform(100, 300, periods)
    
    spot_df = pd.DataFrame({
        'timestamp': timestamps,
        'open': spot_price * 0.999,
        'high': spot_price * 1.001,
        'low': spot_price * 0.998,
        'close': spot_price,
        'volume': np.random.randint(1000, 5000, periods)
    })
    
    fut_df = pd.DataFrame({
        'timestamp': timestamps,
        'open': fut_price * 0.999,
        'high': fut_price * 1.001,
        'low': fut_price * 0.998,
        'close': fut_price,
        'oi': fut_oi,
        'volume': np.random.randint(5000, 10000, periods)
    })
    
    opt_df = pd.DataFrame({
        'timestamp': timestamps,
        'strike': atm_strike,
        'call_ltp': call_ltp,
        'put_ltp': put_ltp,
        'call_oi': np.random.randint(50000, 200000, periods),
        'put_oi': np.random.randint(50000, 200000, periods),
        'call_vol': np.random.randint(10000, 50000, periods),
        'put_vol': np.random.randint(10000, 50000, periods)
    })
    
    spot_df.to_csv(f"{filename_prefix}_spot_5min.csv", index=False)
    fut_df.to_csv(f"{filename_prefix}_futures_5min.csv", index=False)
    opt_df.to_csv(f"{filename_prefix}_options_5min.csv", index=False)
    
    print("Files generated: nifty_spot_5min.csv, nifty_futures_5min.csv, nifty_options_5min.csv")
    return spot_df, fut_df, opt_df

spot, fut, opt = generate_synthetic_nifty_data()