import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
import sys
import os
import warnings
import json
import argparse

warnings.filterwarnings("ignore")

WINDOWS = [30, 60, 90, 120]
WEIGHTS = {30: 0.4, 60: 0.3, 90: 0.2, 120: 0.1}


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

    prophet_df["date_only"] = prophet_df['ds'].dt.date
    daily_df = prophet_df.groupby("date_only").agg({'y': 'last'}).reset_index()
    daily_df.columns = ['ds', 'y']
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])

    last_date = daily_df['ds'].max()
    hours_in_last_day = prophet_df[prophet_df['ds'].dt.date == last_date.date()].shape[0]
    if hours_in_last_day < 20:
        print(f"Uyarı: Son gün ({last_date.date()}) eksik veri içeriyor ({hours_in_last_day} saat). Prophet tahmini yanıltıcı olabilir.")
        daily_df = daily_df[daily_df['ds'] < last_date]

        if daily_df.empty:
            print("Yeterli veri yok, tahmin yapılamıyor.")
            return None  # ← Hata durumunda None dön
        last_date = daily_df['ds'].max()
        print(f"Son tam gün olarak {last_date.date()} kullanılıyor.")

    print(f"Prophet modeli için veri aralığı: {daily_df['ds'].min().date()} - {daily_df['ds'].max().date()} ({len(daily_df)} gün)")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(daily_df)
    
    # Bir sonraki gün için tahmin
    next_day = last_date + pd.Timedelta(days=1)
    future = pd.DataFrame({'ds': [next_day]})
    forecast = model.predict(future)
    yhat = forecast.iloc[0]['yhat']
    print(f"Prophet ile {next_day.date()} için tahmin: {yhat:.2f} USD")
    
    # ← DEĞERİ DÖNDÜR
    return {
        'date': str(next_day.date()),
        'prediction': float(yhat)
    }


def predict_next_day_arima(csv_path):
    _, price_series = load_csv_flexible(csv_path)
    
    # Indexi sıralı ve unique yap
    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    
    # Günlük kapanış fiyatı için yeniden örnekle
    daily_series = price_series.resample('D').last()
    # Eksik günleri doldur (forward fill)
    daily_series = daily_series.ffill()

    last_date_idx = daily_series.index[-1]
    hourly_in_last_day = price_series[price_series.index.date == last_date_idx.date()].shape[0]

    if hourly_in_last_day < 20:
        print(f"Uyarı: Son gün ({last_date_idx.date()}) eksik veri içeriyor ({hourly_in_last_day} saat). ARIMA tahmini yanıltıcı olabilir.")
        daily_series = daily_series[:-1]

        if daily_series.empty:
            print("Yeterli veri yok, tahmin yapılamıyor.")
            return None  # ← Hata durumunda None dön
        print(f"Son tam gün olarak {daily_series.index[-1].date()} kullanılıyor.")

    # Son 365 gün ile sınırla (varsa)
    if len(daily_series) > 365:
        daily_series = daily_series.iloc[-365:]
        print(f"ARIMA modeli için son 365 gün kullanılıyor: {daily_series.index.min().date()} - {daily_series.index.max().date()}")
    
    # ARIMA modeli
    model = sm.tsa.ARIMA(daily_series, order=(1, 1, 1))
    arima_result = model.fit()
    
    # Son tarihi bul ve bir gün ekle
    last_date = daily_series.index.max()
    next_day = last_date + pd.Timedelta(days=1)
    forecast = arima_result.get_forecast(steps=1)
    predicted = forecast.predicted_mean.iloc[0]
    print(f"ARIMA ile {next_day.date()} için tahmin: {predicted:.2f} USD")
    
    # ← DEĞERİ DÖNDÜR
    return {
        'date': str(next_day.date()),
        'prediction': float(predicted)
    }

def get_daily_data(csv_path):
    """ CSV dosyasından gunluk verı hazırlama. """
    prophet_df , price_series = load_csv_flexible(csv_path)

    prophet_df["date_only"] = prophet_df['ds'].dt.date
    daily_df = prophet_df.groupby("date_only").agg({'y': 'last'}).reset_index()
    
    daily_df.columns = ['ds', 'y']
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])
    daily_df = daily_df.sort_values("ds").reset_index(drop=True)

    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    daily_price_series = price_series.resample('D').last().ffill()

    return daily_df, daily_price_series

def predict_prophet_for_window(daily_df, window_days):
    """
    Belirli bir pencere için prophet tahmini yapar.
    """

    if len(daily_df) < window_days:
        return None
    
    windowed_df = daily_df.iloc[-window_days:].copy()
    last_date = windowed_df['ds'].max()

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(windowed_df)

    next_day = last_date + pd.Timedelta(days=1)
    future = pd.DataFrame({'ds': [next_day]})
    forecast = model.predict(future)

    return {
        "prediction": round(float(forecast.iloc[0]['yhat']), 2),
        "data_range": f"{windowed_df['ds'].min().date()} to {windowed_df['ds'].max().date()}"
    }

def predict_arima_for_window(daily_price_series, window_days):
    """
    Belirli bir pencere için ARIMA tahmini yapar.
    """

    if len(daily_price_series) < window_days:
        return None
    
    windowed_series = daily_price_series.iloc[-window_days:].copy()

    model = sm.tsa.ARIMA(windowed_series, order=(1, 1, 1))
    arima_result = model.fit()

    forecast = arima_result.get_forecast(steps=1)
    predicted = forecast.predicted_mean.iloc[0]

    return {
        "prediction": round(float(predicted), 2),
        "data_range": f"{windowed_series.index.min().date()} to {windowed_series.index.max().date()}"
    }

def calculate_ensemble(window_predictions, weights=WEIGHTS):
    """
    Pencere tahminlerinden ağırlıklı ortalama hesaplar.
    """

    total_weight = 0
    weighted_sum = 0

    for window, data in window_predictions.items():
        if data is not None:
            days = int(window.replace("_days", ""))
            weight = weights.get(days, 0)
            weighted_sum += data["prediction"] * weight
            total_weight += weight

    if total_weight == 0:
        return None

    return {
        "prediction": round(weighted_sum / total_weight, 2),
        "weights": {k: v for k, v in weights.items() if f"{k}_days" in window_predictions}
    }


def predict_multi_window(csv_path):
    """
    Tüm pencereler için tahmin  yapar.
    """

    daily_df, daily_series = get_daily_data(csv_path)
    total_days = len(daily_df)

    latest_price = round(float(daily_df['y'].iloc[-1]), 2)
    last_date = daily_df['ds'].max()
    next_day = last_date + pd.Timedelta(days=1)

    prophet_windows = {}
    for window in WINDOWS:
        if total_days >= window:
            result = predict_prophet_for_window(daily_df, window)
        if result:
            prophet_windows[f"{window}_days"] = result
    
    arima_windows = {}
    for window in WINDOWS:
        if total_days >= window:
            result = predict_arima_for_window(daily_series, window)
        if result:
            arima_windows[f"{window}_days"] = result


    prophet_emsemble = calculate_ensemble(prophet_windows)
    arima_ensemble = calculate_ensemble(arima_windows)

    combined = None
    if prophet_emsemble and arima_ensemble:
        combined = {
            "prediction": round((prophet_emsemble["prediction"] + arima_ensemble["prediction"]) / 2, 2),
            "methods": 'average of Prophet and arima emsembles'
        }

    prophet_main =  predict_next_day_prophet(csv_path)
    arima_main = predict_next_day_arima(csv_path)

    return {
        "data_info": {
            "total_days_available": total_days,
            "data_source": "CoinGecko",
            "latest_price": latest_price
        },
        "prophet": {
            "date": str(next_day.date()),
            "prediction": prophet_main['prediction'] if prophet_main else None,
            "windows": prophet_windows,
            "ensemble": prophet_emsemble
        },
        "arima": {
            "date": str(next_day.date()),
            "prediction": arima_main['prediction'] if arima_main else None,
            "windows": arima_windows,
            "ensemble": arima_ensemble
        },
        "combined_ensemble": combined
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin price prediction.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing price data.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Dosya bulunamadi: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.json:

        result = predict_multi_window(args.csv_path)
        print(json.dumps(result, indent=2))

    else:
        print("Prophet tahmini:")
        predict_next_day_prophet(args.csv_path)
        print("ARIMA tahmini:")
        predict_next_day_arima(args.csv_path)

    # # Komut satırından dosya adı al
    # if len(sys.argv) < 2:
    #     print("Kullanım: python quick_predict.py <csv_path>")
    #     sys.exit(1)
    # csv_path = sys.argv[1]
    # if not os.path.exists(csv_path):
    #     print("Dosya bulunamadı:", csv_path)
    #     sys.exit(1)
    # print("Prophet tahmini:")
    # predict_next_day_prophet(csv_path)
    # print("ARIMA tahmini:")
    # predict_next_day_arima(csv_path)