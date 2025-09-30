import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
import sys
import os

def load_csv_flexible(csv_path):
    df = pd.read_csv(csv_path)
    # Kolon isimlerini normalize et
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # Fiyat kolonu belirle
    if 'price' in cols:
        price_col = 'price'
    elif 'open' in cols:
        price_col = 'open'
    elif 'close' in cols:
        price_col = 'close'
    else:
        raise ValueError("Fiyat kolonu bulunamadı!")
    # Tarih/timestamp kolonu belirle
    if 'timestamp' in cols:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_col = 'timestamp'
    elif 'date' in cols:
        df['timestamp'] = pd.to_datetime(df['date'])
        date_col = 'timestamp'
    else:
        raise ValueError("Tarih kolonu bulunamadı!")
    # Prophet için uygun dataframe
    prophet_df = df[[date_col, price_col]].rename(columns={date_col: 'ds', price_col: 'y'})
    return prophet_df, df.set_index('timestamp')[price_col]

def predict_next_day_prophet(csv_path):
    prophet_df, _ = load_csv_flexible(csv_path)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    # Son tarihi bul ve bir gün ekle
    last_date = prophet_df['ds'].max()
    next_day = last_date + pd.Timedelta(days=1)
    future = pd.DataFrame({'ds': [next_day]})
    forecast = model.predict(future)
    yhat = forecast.iloc[0]['yhat']
    print(f"Prophet ile {next_day.date()} için tahmin: {yhat:.2f} USD")

def predict_next_day_arima(csv_path):
    _, price_series = load_csv_flexible(csv_path)
    # Indexi sıralı ve unique yap
    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    # Günlük kapanış fiyatı için yeniden örnekle
    daily_series = price_series.resample('D').last()
    # Eksik günleri doldur (forward fill)
    daily_series = daily_series.ffill()
    # Son 365 gün ile sınırla (varsa)
    if len(daily_series) > 365:
        daily_series = daily_series.iloc[-365:]
    # ARIMA modeli
    model = sm.tsa.ARIMA(daily_series, order=(1, 1, 1))
    arima_result = model.fit()
    # Son tarihi bul ve bir gün ekle
    last_date = daily_series.index.max()
    next_day = last_date + pd.Timedelta(days=1)
    forecast = arima_result.get_forecast(steps=1)
    predicted = forecast.predicted_mean.iloc[0]
    print(f"ARIMA ile {next_day.date()} için tahmin: {predicted:.2f} USD")

if __name__ == "__main__":
    # Komut satırından dosya adı al
    if len(sys.argv) < 2:
        print("Kullanım: python quick_predict.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print("Dosya bulunamadı:", csv_path)
        sys.exit(1)
    print("Prophet tahmini:")
    predict_next_day_prophet(csv_path)
    print("ARIMA tahmini:")
    predict_next_day_arima(csv_path)