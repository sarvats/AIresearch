import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker='AAPL', start='2020-01-01', end='2023-01-01'):
    """
    Fetch stock data and add technical indicators
    """
    # Set auto_adjust to False to get both Close and Adj Close
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    
    # Select only the columns we need
    if 'Adj Close' in df.columns:
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    else:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # If Adj Close is missing, just use Close
        df['Adj Close'] = df['Close']
    
    # Make sure we're working with a clean DataFrame
    # Add technical indicators manually instead of using the ta library
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Low'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Drop rows with NaN values (mostly at the beginning due to indicators)
    df = df.dropna()
    
    # Reset index to get Date as a column
    df = df.reset_index()
    
    return df