import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import sys
import time
import json

def get_bitcoin_historical_data(days_ago=30, interval="hourly"):
    """CoinGecko API'den Bitcoin fiyat verilerini çeker"""
    # CoinGecko API endpoint
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    # Şimdiki zaman (Unix timestamp, ms)
    end_time = int(datetime.now().timestamp() * 1000)
    # Belirtilen gün sayısı kadar önceki zaman
    start_time = int((datetime.now() - timedelta(days=days_ago)).timestamp() * 1000)
    
    # API parametreleri - interval can be daily, hourly, or minutely
    params = {
        'vs_currency': 'usd',
        'from': start_time // 1000,  # saniye cinsinden 
        'to': end_time // 1000,      # saniye cinsinden
        'days': days_ago,
        'interval': interval  # 'daily', 'hourly', or CoinGecko might accept other intervals
    }
    
    try:
        print(f"CoinGecko API'den son {days_ago} günün {interval} verilerini çekiyorum...")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Hata durumunda exception fırlat
        
        data = response.json()
        
        # Fiyat verileri [timestamp, price] formatında
        prices = data.get('prices', [])
        # Hacim verileri [timestamp, volume] formatında 
        volumes = data.get('total_volumes', [])
        
        # Timestamp ve fiyat verilerini DataFrame'e dönüştür
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'open'])  # renamed to 'open'
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        # DataFrame'leri birleştir
        df = pd.merge(df_prices, df_volumes, on='timestamp', how='left')
        
        # Timestamp'i datetime'a dönüştür
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date'] = df['timestamp'].dt.date
        
        # Verileri ters çevir (yeni tarihten eskiye doğru sırala)
        df = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
        
        # Fiyat değişimi hesapla - bu değer kesinlikle open değerinden farklı olacak
        df['change'] = df['open'].diff(-1)  # Bir önceki döneme göre fiyat değişimi
        df['change_pct'] = (df['open'] / df['open'].shift(-1) - 1) * 100  # Yüzdelik değişim
        
        # Sadece istenen sütunları seç
        df = df[['timestamp', 'date', 'open', 'volume', 'change', 'change_pct']]
        
        return df
            
    except Exception as e:
        print(f"Veri çekerken hata oluştu: {e}")
        return pd.DataFrame()

def main():
    # Komut satırı argümanlarını al
    if len(sys.argv) > 2:
        days_ago = int(sys.argv[1])
        interval = sys.argv[2]  # 'daily', 'hourly' veya diğer desteklenen değerler
    elif len(sys.argv) > 1:
        days_ago = int(sys.argv[1])
        interval = "hourly"  # Default to hourly
    else:
        days_ago = 30  # Default değer
        interval = "hourly"  # Default to hourly
    
    # Data klasörünü oluştur (yoksa)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Veriyi çek
    df = get_bitcoin_historical_data(days_ago=days_ago, interval=interval)
    
    if df.empty:
        print("Veri çekilemedi!")
        sys.exit(1)
    
    # CSV dosyasını kaydet
    filename = f"BTC_USD_{days_ago}days_{interval}.csv"
    csv_path = os.path.join(data_dir, filename)
    
    df.to_csv(csv_path, index=False)
    
    # Son çekilen verileri metadata olarak kaydet
    metadata_file = os.path.join(data_dir, "metadata.json")
    metadata = {
        "last_update": datetime.now().isoformat(),
        "symbol": "BTC/USD",
        "days": days_ago,
        "interval": interval,
        "records": len(df),
        "filename": filename,
        "source": "CoinGecko"
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Veri başarıyla çekildi ve kaydedildi: {csv_path}")
    print(f"Toplam kayıt: {len(df)}")
    
    # İlk birkaç satırı göster
    print("\nİlk 5 veri satırı:")
    print(df.head(5))
    
    # Değişim istatistiklerini ekrana yazdır
    print("\nDeğişim İstatistikleri:")
    print(f"En yüksek artış: +${df['change'].max():.2f} ({df['change_pct'].max():.2f}%)")
    print(f"En büyük düşüş: ${df['change'].min():.2f} ({df['change_pct'].min():.2f}%)")
    print(f"Ortalama değişim: ${df['change'].mean():.2f} ({df['change_pct'].mean():.2f}%)")
    print(f"Son {days_ago} gün toplam değişim: ${df['open'].iloc[0] - df['open'].iloc[-1]:.2f} ({((df['open'].iloc[0] - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100):.2f}%)")

if __name__ == "__main__":
    main()