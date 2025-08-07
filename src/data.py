import pandas as pd
import streamlit as st
from .yahoo_data import get_latest_prices_from_yahoo

@st.cache_data
def load_data(): 
    print("Loading data...")
    url = ""
    df = pd.read_parquet(url)
    df = df.sort_values('timestamp') # Ensure data is sorted chronologically
    df = try_adding_latest_prices(df)
    return df

def try_adding_latest_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo Finance API is not always available, so we try to add the latest prices and if it fails, we return the original dataframe"""
    try:
        return add_latest_prices(df)
    except Exception as e:
        print(f"Error adding latest prices: {e}")
        return df

def add_latest_prices(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df['timestamp'].max()
    latest_date_plus_1 = latest_date + pd.Timedelta(days=1)
    latest_prices = get_latest_prices_from_yahoo(latest_date_plus_1)
    print(f"Number of prices added: {len(latest_prices)}")
    df = pd.concat([df, latest_prices])
    df.reset_index(drop=True, inplace=True)
    return df