def backtest_strategy(df):
    df['signal'] = 0
    df.loc[(df['ema_5'] > df['ema_15']) & (df['regime'] == 1), 'signal'] = 1
    df.loc[(df['ema_5'] < df['ema_15']) & (df['regime'] == 2), 'signal'] = -1
    
    df['strategy_returns'] = df['signal'].shift(1) * df['spot_returns']
    
    sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252 * 75) # 75 5-min candles/day
    cum_returns = (1 + df['strategy_returns']).cumprod()
    return cum_returns, sharpe