import pandas as pd
from types import SimpleNamespace

def load_boston(return_X_y=False):
    """Replacement function for loading in Boston House Prices"""
    df = pd.read_csv('boston/boston_house_prices.csv')
    X = df.drop(columns=['MEDV']).to_numpy()
    y = df['MEDV'].to_numpy()

    if return_X_y:
        return X, y 
    
    dataset  = SimpleNamespace(data=X, target=y)
    
    return dataset