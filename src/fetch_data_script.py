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
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    # Şimdiki zaman (Unix timestamp, ms)
    end_time = int(datetime.now().timestamp() * 1000)
    # Belirtilen gün sayısı kadar önceki zaman
    start_time = int((datetime.now() - timedelta(days=days_ago)).timestamp() * 1000)
    
    # API parametreleri
    params = {
        'vs_currency': 'usd',
        'from': start_time // 1000,  # saniye cinsinden 
        'to': end_time // 1000,      # saniye cinsinden
        'days': days_ago
    }
    
    try:
        print(f"CoinGecko API'den son {days_ago} günün verilerini çekiyorum...")
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
        df['date'] = df['timestamp'].dt.date
        
        # OHLC verisi için - CoinGecko sadece kapanış fiyatı veriyor, 
        # basit bir yaklaşımla aynı değeri diğer alanlara kopyalayalım
        # Not: Bu gerçek OHLC verisi değil, sadece veri yapısını korumak için
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        
        # Sütunları yeniden düzenle
        df = df[['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Verileri ters çevir (yeni tarihten eskiye doğru sırala)
        df = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
        
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
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"BTC_USD_{days_ago}days_{timestamp}.csv"
    # csv_path = os.path.join(data_dir, filename)
    filename = f"BTC_USD_{days_ago}days.csv"
    csv_path = os.path.join(data_dir, filename)
    
    df.to_csv(csv_path, index=False)
    
    # Son çekilen verileri metadata olarak kaydet
    metadata_file = os.path.join(data_dir, "metadata.json")
    metadata = {
        "last_update": datetime.now().isoformat(),
        "symbol": "BTC/USD",
        "days": days_ago,
        "records": len(df),
        "filename": filename,
        "source": "CoinGecko"
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Veri başarıyla çekildi ve kaydedildi: {csv_path}")
    print(f"Toplam kayıt: {len(df)}")

if __name__ == "__main__":
    main()