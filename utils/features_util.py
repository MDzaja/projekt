import pandas as pd
import scipy.stats as stats
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

features = ['Min', #minimum value of the stock price in the last N days
    'Max', #maximum value of the stock price in the last N days
    'Mean', #mean value of the stock price in the last N days
    'Std', #standard deviation of the stock price in the last N days
    'Skewness', #skewness of the stock price in the last N days, a measure of the asymmetry of the 
                #probability distribution of a real-valued random variable about its mean,
                #negative avlues indicating these stocks are downward most of the time and experience negative returns
    'Kurtosis', #kurtosis of the stock price in the last N days, a measure of the "tailedness" of the probability,
                #shows the degree of presence of outliers in the underlying distribution
    'Chi-Square', #chi-square test of the stock price in the last N days, a measure of the skewness and kurtosis,
                  #the large values of Chi-Square confrm that the null hypothesis for all series to be normally 
                  #distributed is rejected
    'Beta', #beta of the stock price in the last N days, a measure of the volatility of the stock relative to the market
           #beta = 1 means the stock is as volatile as the market, beta > 1 means the stock is more volatile than the market
    'Mean_Volume' #volume of the stock in the last N days
    ]

def get_available_features():
    return features

def compute_features(stock_symbol, stock_series, market_series, observe_prev_N_days):
    volumes = yf.download(stock_symbol, stock_series.index[0], stock_series.index[-1], progress=False)['Volume']
    features_df = pd.DataFrame(columns = features)

    for i in range(observe_prev_N_days, len(stock_series)):
        history_data = stock_series[i-observe_prev_N_days:i]
        chi, _p = stats.jarque_bera(history_data)
        new_row = {
            'Min': min(history_data),
            'Max': max(history_data),
            'Mean': history_data.mean(),
            'Std': history_data.std(),
            'Skewness': history_data.skew(), 
            'Kurtosis': history_data.kurtosis(), 
            'Chi-Square': chi, 
            'Beta': calculate_beta(history_data, market_series[i-observe_prev_N_days:i]),
            'Mean_Volume': volumes[i-observe_prev_N_days:i].mean()
        }
        features_df = features_df.append(new_row, ignore_index=True)

    features_df = features_df.set_index(stock_series[observe_prev_N_days:].index)
    return features_df

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