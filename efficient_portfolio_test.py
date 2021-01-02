# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

symbols = ['TSLA', 'FB', 'GOOG', 'MSFT', 'AMZN']
# Read Data
test = data.DataReader(symbols, 'yahoo', start='2018/01/01', end='2020/12/31')

test = test['Adj Close']
print (test.head())

# Log of percentage change
tesla = test['TSLA'].pct_change().apply(lambda x: np.log(1+x))

tesla.head()

var_tesla = tesla.var()
print(var_tesla)

fb = test['FB'].pct_change().apply(lambda x: np.log(1+x))
fb.head()

var_fb = fb.var()
print (var_fb)

# Volatility
tesla_vol = np.sqrt(var_tesla * 250)
fb_vol = np.sqrt(var_fb * 250)
print(tesla_vol, fb_vol)

#plot
#test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250)).plot(kind='bar')

# Log of Percentage change
test1 = test.pct_change().apply(lambda x: np.log(1+x))
# Covariance
cov = test1['TSLA'].cov(test1['FB'])
corr = test1['TSLA'].corr(test1['FB'])
print (cov, corr)


returns = test.pct_change().apply(lambda x: np.log(1+x))

print(returns)

weights = np.random.random(len(symbols))
weights /= np.sum(weights)

print (weights)