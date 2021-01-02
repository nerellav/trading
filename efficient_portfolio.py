# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

import numpy as np
import pandas as pd
from pandas_datareader import data
import scipy.optimize as sco
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
np.random.seed(777)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    # multiplying with 252 to make daily returns as annual
    returns = np.sum(mean_returns * weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

#taking negative as we have to maximize portfolio with high sharpe ratio using "minimize"optimization
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def portfolio_volatility(weights, mean_returns, cov_matrix):
    # return standard deviation
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    #num_assets = len(mean_returns)
    num_assets = len(symbols)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # sum of weights == 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # asset weight range
    bounds = tuple( (0.0, 1.0) for asset in range(num_assets))
    
    #SLSQP = Sequential Least Squares Programming
    #L-BFGS-B = Limited-memory BFGS with bounds
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def min_variance(mean_returns, cov_matrix):
    num_assets = len(symbols)
    args = (mean_returns, cov_matrix)
    # sum of weights == 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # asset weight range
    bounds = tuple( (0.0, 1.0) for asset in range(num_assets))
    
    #SLSQP = Sequential Least Squares Programming
    #L-BFGS-B = Limited-memory BFGS with bounds
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, target):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=df.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=df.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    print ("-"*80)
    print ("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(df.columns):
        print( txt,":","annuaised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2))  
    print ("-"*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(df.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    # find efficient portfolios based on expected returns
    
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)



symbols = ['AAPL', 'FB', 'GOOG', 'MSFT', 'AMZN', 'NKE', 'TSLA']
# Read Data
df = data.DataReader(symbols, 'yahoo', start='2018/01/01', end='2020/12/31')

df = df['Adj Close']

#plot
#df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250)).plot(kind='bar')

# Covariance correlation
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
mean_returns = df.pct_change().apply(lambda x: np.log(1+x)).mean()

print (mean_returns)

#UST yield
risk_free_rate = 0.02

returns = df.pct_change().apply(lambda x: np.log(1+x))

print(corr_matrix)

weights = np.random.random(len(symbols))
weights /= np.sum(weights)

print (weights)

random_port_var = np.dot(np.dot(weights, cov_matrix), weights.transpose())

print(random_port_var)

target = np.linspace(0.30, 0.50, 50)

display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, target)

