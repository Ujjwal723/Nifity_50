def analyze_outliers(df):
    trades = df[df['signal'] != 0].copy()
    trades['pnl'] = trades['strategy_returns']
    z_score = (trades['pnl'] - trades['pnl'].mean()) / trades['pnl'].std()
    outliers = trades[z_score > 3]
    
    print(f"Percentage of Outlier Trades: {len(outliers)/len(trades)*100:.2f}%")
    
    sns.boxplot(x='regime', y='avg_iv', data=outliers)
    plt.title("IV Distribution during High-Performance Trades")
    plt.show()