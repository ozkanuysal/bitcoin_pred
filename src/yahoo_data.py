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
    # Drop rows with invalid timestamps
    num_invalid = df["timestamp"].isna().sum()
    if num_invalid > 0:
        print(f"Dropping {num_invalid} rows with invalid timestamps from Yahoo data.")
        df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype('datetime64[ns]')
    df["price"] = pd.to_numeric(df["price"], errors='coerce')
    df.sort_values(by="timestamp", inplace=True)
    print(f"Yahoo timestamp dtype: {df['timestamp'].dtype}")
    print(df['timestamp'].head())
    return df