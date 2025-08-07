import yfinance as yf
import pandas as pd 

def get_latest_prices_from_yahoo(start_date) -> pd.DataFrame:
    """
    Get the latest prices from Yahoo Finance for a given start date
    """
    btc = yf.Ticker('BTC-USD')
    df = btc.history(start=start_date, period=None)
    df["Date"] = df.index
    df.reset_index(drop=True, inplace=True)
    df = df[["Date", "Open"]]
    df.rename(columns={"Date": "timestamp", "Open": "price"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"], errors='coerce')
    df.sort_values(by="timestamp", inplace=True)
    return df