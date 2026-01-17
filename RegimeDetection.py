def train_hmm(df):
    features = ['avg_iv', 'iv_spread', 'pcr_oi', 'call_delta', 'gamma', 'futures_basis', 'spot_returns']
    train_size = int(len(df) * 0.7)
    train_data = df[features].iloc[:train_size]
    
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
    model.fit(train_data)
    
    df['regime'] = model.predict(df[features])
    return df, model

def plot_regimes(df):
    plt.figure(figsize=(15, 5))
    colors = {0: 'gray', 1: 'green', 2: 'red'} 
    for regime, color in colors.items():
        mask = df['regime'] == regime
        plt.scatter(df.index[mask], df['close'][mask], color=color, s=1, label=f'Regime {regime}')
    plt.title("NIFTY Spot with HMM Regimes")
    plt.legend()
    plt.show()