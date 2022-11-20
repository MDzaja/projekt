import pandas as pd
import scipy.stats as stats
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_features(stock_series, market_series, N):
    features = pd.DataFrame(columns = ['Min', 'Max', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Chi-Square', 'p', 'Beta'])

    for i in range(N, len(stock_series)):
        history_data = stock_series[i-N:i]
        chi, p = stats.jarque_bera(history_data)
        new_row = {
            'Min': min(history_data),
            'Max': max(history_data),
            'Mean': history_data.mean(),
            'Std': history_data.std(),
            'Skewness': history_data.skew(), 
            'Kurtosis': history_data.kurtosis(), 
            'Chi-Square': chi, 
            'p': p, 
            'Beta': calculate_beta(history_data, market_series[i-N:i])
        }
        features = features.append(new_row, ignore_index=True)

    features = features.set_index(stock_series[N:].index)
    return features

def calculate_beta(stock_series, market_series):
    data = pd.DataFrame({'stock': stock_series, 'market': market_series})
    # Convert historical stock prices to daily percent change
    price_change = data.pct_change()
    # Deletes row one containing the NaN
    df = price_change.drop(price_change.index[0])
    # Create arrays for x and y variables in the regression model
    # Set up the model and define the type of regression
    x = np.array(df['stock']).reshape((-1,1))
    y = np.array(df['market'])
    model = LinearRegression().fit(x, y)
    return model.coef_[0]