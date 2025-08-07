import pandas as pd
import streamlit as st

@st.cache_data
def load_data(): 
    print("Loading data...")
    file_path = '/home/ozkan/Desktop/bitcoin_prediction/bitcoin_pred/Bitcoin_Historical_Data.csv'
    df = pd.read_csv(file_path)
    df.rename(columns={'Date': 'timestamp', 'Price': 'price'}, inplace=True)
    df['price'] = df['price'].str.replace(',', '').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%Y")
    df = df.sort_values('timestamp')
    return df