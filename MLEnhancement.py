from xgboost import XGBClassifier

def train_ml_filter(df):

    df['target'] = (df['spot_returns'].shift(-5).rolling(5).sum() > 0).astype(int)
    
    features = ['avg_iv', 'pcr_oi', 'delta_neutral_ratio', 'gamma_exposure', 'regime']
    X = df[features]
    y = df['target']
    
    split = int(len(df) * 0.7)
    model_xgb = XGBClassifier()
    model_xgb.fit(X.iloc[:split], y.iloc[:split])
    
    df['ml_confidence'] = model_xgb.predict_proba(X)[:, 1]
    return df