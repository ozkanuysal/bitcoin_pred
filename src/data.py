import pandas as pd
import streamlit as st

def load_csv_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    if df.empty:
        return df
    
    df.rename(columns={'Date': 'timestamp', 'Price': 'price'}, inplace=True)
    df['price'] = df['price'].str.replace(',', '').astype(float)
    return df

def handle_timestamps(df):
    if df.empty or 'timestamp' not in df.columns:
        return df

    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%Y", errors='coerce', utc=True)
    
    num_invalid = df['timestamp'].isna().sum()
    if num_invalid > 0:
        st.warning(f"Dropping {num_invalid} rows with invalid timestamps.")
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
            return df

    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    return df

@st.cache_data
def load_data():
    st.info("Loading Bitcoin historical data...")
    file_path = '/home/ozkan/Desktop/bitcoin_prediction/bitcoin_pred/Bitcoin_Historical_Data.csv'
    
    df = load_csv_data(file_path)
    df = preprocess_data(df)
    df = handle_timestamps(df)

    if df.empty:
        st.error("No valid data after cleaning. Please check your CSV file.")
        return df

    df = df.sort_values('timestamp')
    st.success(f"Loaded {len(df)} rows of Bitcoin data.")
    return df