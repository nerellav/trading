import yfinance as yf
import pickle
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def download_data_by_ticker(tickers, start_date, end_date):
    ticker_data = yf.download(tickers, start=start_date, end=end_date)
    for ticker in tickers:
        # loop through by month from start_date to end_date
        for i in range(int((ticker_data.index[-1] - ticker_data.index[0]).days / 30) + 1):
            start_month = ticker_data.index[0] + pd.DateOffset(months=i*1)
            end_month = start_month + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            # extract data of a ticker for each month from ticker_data

            # ticker_data_month = ticker_data.xs(ticker, level='Ticker', axis=1).loc[start_month:end_month]
            ticker_data_month = ticker_data.loc[:, (slice(None), ticker)].loc[start_month:end_month]

            # exract year and month so that we can use it as the filename
            year = start_month.year
            month = start_month.month
            pickle.dump(ticker_data_month, open(f'data/indian_equities_data_{ticker}_{year}_{month}.pkl', 'wb'))
            print(f'Data downloaded and saved for {ticker}_{year}_{month}')


def load_data_from_files(tickers, start_date, end_date):
    ticker_data_concat = pd.DataFrame()
    for ticker in tickers:
        # loop through all months between start_date and end_date
        ticker_df = pd.DataFrame()
        for beg in pd.date_range(start_date, end_date, freq='MS'):
            month = beg.strftime("%Y_%-m")
            # print(beg.strftime("%Y-%-m-%-d"), (beg + MonthEnd(1)).strftime("%Y-%-m-%-d"))
            ticker_data = pickle.load(
                open(f'data/indian_equities_data_{ticker}_{month}.pkl', 'rb'))
            ticker_df = pd.concat([ticker_df, ticker_data])

        ticker_data_concat = pd.concat([ticker_data_concat, ticker_df], axis=1)

    return ticker_data_concat


if __name__ == '__main__':
    # List of Indian equities to include in the portfolio
    tickers = ['RELIANCE.NS', 'INFY.NS',
               'AXISBANK.NS', 'RELIANCE.NS', 'ONGC.NS',
               'TCS.NS', 'LUPIN.NS', 'JINDALSTEL.NS',
               'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS',
               'INDUSINDBK.NS', 'BHARTIARTL.NS',
               'MARUTI.NS', 'ONGC.NS', 'SBILIFE.NS', 'SBIN.NS',
               'M&M.NS', 'SANSERA.NS', 'PAGEIND.NS', 'BOSCHLTD.NS',
               'SUNPHARMA.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS']

    tickers = ['TCS.NS', 'INFY.NS', 'JINDALSTEL.NS']

    tcs_data_2024_1 = pickle.load(
        open(f'data/indian_equities_data_TCS.NS_2024_1.pkl', 'rb'))
    tcs_data_2024_2 = pickle.load(
        open(f'data/indian_equities_data_TCS.NS_2024_2.pkl', 'rb'))
    tcs_data_2024_3 = pickle.load(
        open(f'data/indian_equities_data_TCS.NS_2024_3.pkl', 'rb'))
    infy_data_2024_1 = pickle.load(
        open(f'data/indian_equities_data_INFY.NS_2024_1.pkl', 'rb'))
    infy_data_2024_2 = pickle.load(
        open(f'data/indian_equities_data_INFY.NS_2024_2.pkl', 'rb'))
    infy_data_2024_3 = pickle.load(
        open(f'data/indian_equities_data_INFY.NS_2024_3.pkl', 'rb'))

    # download_data_by_ticker(tickers, '2021-01-01', '2024-07-10')
    ticker_data = load_data_from_files(tickers, '2024-01-01', '2024-02-28')
    print(ticker_data)

