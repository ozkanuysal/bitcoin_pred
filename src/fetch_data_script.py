import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import sys
import time
import json

def get_bitcoin_historical_data(days_ago=30):
    """CoinGecko API'den Bitcoin fiyat verilerini çeker"""
    # CoinGecko API endpoint
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    yesterday_end = datetime.now() - timedelta(days=1)
    yesterday_end = yesterday_end.replace(hour=23, minute=59, second=59 , microsecond=0)
    
    start_time = int((yesterday_end - timedelta(days=days_ago)).timestamp())
    end_time = int(yesterday_end.timestamp())

    # API parametreleri
    params = {
        'vs_currency': 'usd',
        'from': start_time,  # saniye cinsinden
        'to': end_time,      # saniye cinsinden
    }
    
    try:
        print(f"CoinGecko API'den son {days_ago} günün verilerini çekiyorum...")
        print(" Veri aralığı:", {yesterday_end - timedelta(days=days_ago), " - ", yesterday_end})
        response = requests.get(url, params=params)
        response.raise_for_status()  # Hata durumunda exception fırlat
        
        data = response.json()
        
        # Fiyat verileri [timestamp, price] formatında
        prices = data.get('prices', [])
        # Hacim verileri [timestamp, volume] formatında 
        volumes = data.get('total_volumes', [])
        
        # Timestamp ve fiyat verilerini DataFrame'e dönüştür
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        # DataFrame'leri birleştir
        df = pd.merge(df_prices, df_volumes, on='timestamp', how='left')

        # Timestamp'i datetime'a dönüştür
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f" Toplam {len(df)} kayıt çekildi.")

        # Günlük veriye dönüştür (CoinGecko 90+ günde zaten günlük döndürür,
        # kısa aralıklarda saatlik döndürür — her iki durumu da doğru handle et)
        df = df.set_index('timestamp')
        df_daily = df.resample("1D").agg({'close': 'last', 'volume': 'sum'})
        df_daily = df_daily.dropna().reset_index()
        print(f"Günlük resample sonrası {len(df_daily)} kayıt mevcut.")

        df_daily['date'] = df_daily['timestamp'].dt.date
        df_daily['open'] = df_daily['close']
        df_daily = df_daily.sort_values(by='timestamp', ascending=True)
        df_daily['change'] = df_daily['close'].pct_change() * 100
        df_daily['change'] = df_daily['change'].fillna(0)
        df_daily = df_daily.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
        df_final = df_daily[['timestamp', 'date', 'open', 'close', 'volume', 'change']]
        return df_final

            
    except Exception as e:
        print(f"Veri çekerken hata oluştu: {e}")
        return pd.DataFrame()

def main():
    # Komut satırı argümanlarını al
    if len(sys.argv) > 1:
        days_ago = int(sys.argv[1])
    else:
        days_ago = 30  # Default değer
    
    # Data klasörünü oluştur (yoksa)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Veriyi çek
    df = get_bitcoin_historical_data(days_ago=days_ago)
    
    if df.empty:
        print("Veri çekilemedi!")
        sys.exit(1)
    
    # CSV dosyasını kaydet
    filename = f"BTC_USD_{days_ago}days.csv"
    csv_path = os.path.join(data_dir, filename)
    
    df.to_csv(csv_path, index=False)
    
    # Son çekilen verileri metadata olarak kaydet
    metadata_file = os.path.join(data_dir, "metadata.json")
    yesterday_end = datetime.now() - timedelta(days=1)
    yesterday_end = yesterday_end.replace(hour=23, minute=59, second=59 , microsecond=0)

    metadata = {
        "last_update": datetime.now().isoformat(),
        "symbol": "BTC/USD",
        "days": days_ago,
        "records": len(df),
        "filename": filename,
        "source": "CoinGecko",
        "data_end_date": yesterday_end.isoformat(),
        "data_start_date": (yesterday_end - timedelta(days=days_ago)).isoformat(),
        "note": "Data include complete days up to yesterday."

    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Veri başarıyla çekildi ve kaydedildi: {csv_path}")
    print(f"Toplam kayıt: {len(df)}")

if __name__ == "__main__":
    main()