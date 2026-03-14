import pandas as pd
import streamlit as st
import os
import json
import glob

# Proje kök dizini (src/ klasörünün bir üstü)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_coingecko_data():
    """data/ klasöründeki en güncel CoinGecko CSV'sini yükle"""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    metadata_path = os.path.join(data_dir, "metadata.json")

    # metadata.json'dan dosya adını al
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        csv_path = os.path.join(data_dir, metadata["filename"])
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]

            # Fiyat kolonu: close veya open
            price_col = 'close' if 'close' in df.columns else 'open'
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].dt.tz_localize(None).astype('datetime64[ns]')
            df = df[['timestamp', price_col]].rename(columns={price_col: 'price'})
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

    # Fallback: dosya adındaki gün numarasına göre en büyüğü seç
    csv_files = sorted(glob.glob(os.path.join(data_dir, "BTC_USD_*.csv")))
    if csv_files:
        csv_path = csv_files[-1]  # sort ile en büyük numara sonda
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        price_col = 'close' if 'close' in df.columns else 'open'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.tz_localize(None).astype('datetime64[ns]')
        df = df[['timestamp', price_col]].rename(columns={price_col: 'price'})
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    return None


def _load_legacy_csv():
    """Eski Bitcoin_Historical_Data.csv dosyasını yükle (investing.com formatı)"""
    file_path = os.path.join(PROJECT_ROOT, "Bitcoin_Historical_Data.csv")
    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    if df.empty:
        return None

    df.rename(columns={'Date': 'timestamp', 'Price': 'price'}, inplace=True)
    df['price'] = df['price'].str.replace(',', '').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%Y", errors='coerce', utc=True)

    num_invalid = df['timestamp'].isna().sum()
    if num_invalid > 0:
        df.dropna(subset=['timestamp'], inplace=True)

    if df.empty:
        return None

    df['timestamp'] = df['timestamp'].dt.tz_localize(None).astype('datetime64[ns]')
    df = df[['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)
    return df


@st.cache_data
def load_data():
    st.info("Loading Bitcoin historical data...")

    # Önce güncel CoinGecko verisini dene
    df = _load_coingecko_data()
    if df is not None and not df.empty:
        st.success(f"Loaded {len(df)} rows of Bitcoin data (CoinGecko).")
        return df

    # Fallback: eski manual CSV
    df = _load_legacy_csv()
    if df is not None and not df.empty:
        st.success(f"Loaded {len(df)} rows of Bitcoin data (legacy CSV).")
        return df

    st.error("No valid data found. Please check your data files.")
    return pd.DataFrame()
