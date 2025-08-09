import pandas as pd
import streamlit as st

@st.cache_data
def load_data(for_display=False):
    st.info("Loading Bitcoin historical data...")
    file_path = '/home/ozkan/Desktop/bitcoin_prediction/bitcoin_pred/Bitcoin_Historical_Data.csv'
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

    # Rename columns and clean price
    df.rename(columns={'Date': 'timestamp', 'Price': 'price'}, inplace=True)
    df['price'] = df['price'].str.replace(',', '').astype(float)

    # Robust datetime conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%Y", errors='coerce', utc=True)
    # Remove timezone info
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    # Truncate excessive precision (microseconds)
    df['timestamp'] = df['timestamp'].dt.floor('ns')
    # Drop rows with invalid timestamps
    num_invalid = df['timestamp'].isna().sum()
    if num_invalid > 0:
        st.warning(f"Dropping {num_invalid} rows with invalid timestamps.")
        df = df.dropna(subset=['timestamp'])
    # Ensure Arrow compatibility: always convert to string for display
    if for_display or not pd.api.types.is_datetime64_ns_dtype(df['timestamp']):
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Final check for Arrow compatibility
    if df['timestamp'].dtype == 'O':
        st.warning("Timestamp column is object dtype. Displaying as string. Some features may be limited.")
    if df.empty:
        st.error("No valid data after cleaning. Please check your CSV file.")
        return df
    df = df.sort_values('timestamp')
    st.success(f"Loaded {len(df)} rows of Bitcoin data.")
    return df