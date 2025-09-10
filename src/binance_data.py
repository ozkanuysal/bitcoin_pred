import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import time
import streamlit as st

class BinanceDataFetcher:
    def __init__(self, api_key=None, api_secret=None):
        """Binance API client başlatıcı"""
        self.client = Client(api_key, api_secret)
    
    @st.cache_data(ttl=300)  # 5 dakika cache süresi
    def get_historical_klines(_self, symbol="BTCUSDT", interval="1d", days_ago=30):
        """
        Belirli bir zaman aralığı için tarihsel mum verilerini çeker
        
        Args:
            symbol: İşlem çifti (örn. "BTCUSDT")
            interval: Zaman aralığı ("1d", "4h", "1h", "15m" vb)
            days_ago: Bugünden geriye doğru kaç gün
            
        Returns:
            pandas.DataFrame: İşlenen tarihsel veri
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_ago)
        
        # Unix timestamp (ms) formatına dönüştür
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        try:
            klines = _self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_timestamp,
                end_str=end_timestamp
            )
            
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Veri tiplerini düzenle
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Gereksiz sütunları kaldır
            df = df[['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            st.error(f"Veri çekerken hata oluştu: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol="BTCUSDT"):
        """Güncel fiyatı döndürür"""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])