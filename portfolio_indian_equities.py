import yfinance as yf
import pickle
from pypfopt.expected_returns import mean_historical_return
from pypfopt import EfficientSemivariance, EfficientCDaR
from pypfopt import (
    EfficientFrontier,
    expected_returns,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from download_data import load_data_from_files

# List of Indian equities to include in the portfolio
tickers = ['RELIANCE.NS', 'INFY.NS',
           'AXISBANK.NS', 'RELIANCE.NS', 'ONGC.NS',
           'TCS.NS', 'LUPIN.NS', 'JINDALSTEL.NS',
           'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS',
           'INDUSINDBK.NS', 'BHARTIARTL.NS',
           'MARUTI.NS', 'ONGC.NS', 'SBILIFE.NS', 'SBIN.NS',
           'M&M.NS', 'SANSERA.NS', 'PAGEIND.NS', 'BOSCHLTD.NS',
           'SUNPHARMA.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS']


# Load the data from the pickle file
# data = load_data_from_files(tickers, '2021-01-01', '2024-06-30')

try:
    data = pickle.load(open('data/indian_equities_data_2022.pkl', 'rb'))
except:
    print('Data not found in the pickle file. Downloading data now...')
    data = yf.download(tickers, start='2022-01-01', end='2024-07-07')
    pickle.dump(data, open('data/indian_equities_data_2022.pkl', 'wb'))

data_from_file = load_data_from_files(tickers, '2022-01-01', '2024-07-11')
# Calculate daily returns
returns = data_from_file['Adj Close']

# Calculate expected returns and covariance matrix
mu = mean_historical_return(returns)

historical_returns = expected_returns.returns_from_prices(returns).dropna(axis=0, how="all")

es = EfficientSemivariance(mu, historical_returns)
es.efficient_return(0.20)

# We can use the same helper methods as before
weights = es.clean_weights()
print(weights)
es.portfolio_performance(verbose=True)

# allocate 10 lacs based on the weights and latest prices
allocation = pd.Series(weights, index=returns.columns) * 1000000
allocation.name = 'Allocation'
print(allocation)

beta = 0.90
df = returns
df = df.resample("W").first()
mu = expected_returns.mean_historical_return(df, frequency=52)
historical_rets = expected_returns.returns_from_prices(df).dropna(axis=0, how="any")
cd = EfficientCDaR(mu, historical_rets, beta=beta)
print(cd.efficient_return(0.20))
cd.portfolio_performance(verbose=True)

cd_weights = cd.clean_weights()
# allocate 10 lacs based on the weights and latest prices
allocation = pd.Series(cd_weights, index=returns.columns) * 1000000
allocation.name = 'Allocation'
print(allocation)

# print weights and cd_weights side by side so that we can comprare
weights_comparison = pd.concat(
    [pd.Series(weights, index=returns.columns) * 1000000, pd.Series(cd_weights, index=returns.columns) * 1000000],
    axis=1)
weights_comparison.columns = ['ES Weights', 'CDaR Weights']
print(weights_comparison)
