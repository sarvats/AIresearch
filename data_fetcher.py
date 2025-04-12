import yfinance as yf

def fetch_data(ticker='AAPL', start='2020-01-01', end='2023-01-01'):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()
    return df
