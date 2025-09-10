import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import os
import sys
import time
import json

def get_historical_klines(client, symbol="BTCUSDT", interval="1d", days_ago=30):
    """Belirli bir zaman aralığı için tarihsel mum verilerini çeker"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_ago)
    
    # Unix timestamp (ms) formatına dönüştür
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    try:
        klines = client.get_historical_klines(
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
        print(f"Veri çekerken hata oluştu: {e}")
        return pd.DataFrame()

def main():
    # Komut satırı argümanlarını al
    if len(sys.argv) > 1:
        days_ago = int(sys.argv[1])
    else:
        days_ago = 30  # Default değer
    
    symbol = "BTCUSDT"
    interval = "1d"
    
    # Data klasörünü oluştur (yoksa)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Binance client oluştur
    # Not: API anahtarları olmadan da public verilere erişilebilir
    client = Client()
    
    print(f"Son {days_ago} günün verilerini çekiyorum...")
    
    # Veriyi çek
    df = get_historical_klines(client, symbol=symbol, interval=interval, days_ago=days_ago)
    
    if df.empty:
        print("Veri çekilemedi!")
        sys.exit(1)
    
    # CSV dosyasını kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{interval}_{days_ago}days_{timestamp}.csv"
    csv_path = os.path.join(data_dir, filename)
    
    df.to_csv(csv_path, index=False)
    
    # Son çekilen verileri metadata olarak kaydet
    metadata_file = os.path.join(data_dir, "metadata.json")
    metadata = {
        "last_update": datetime.now().isoformat(),
        "symbol": symbol,
        "interval": interval,
        "days": days_ago,
        "records": len(df),
        "filename": filename
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Veri başarıyla çekildi ve kaydedildi: {csv_path}")
    print(f"Toplam kayıt: {len(df)}")

if __name__ == "__main__":
    main()