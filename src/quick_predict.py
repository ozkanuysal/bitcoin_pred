import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
import sys
import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

# Multi-window configuration
WINDOWS = [30, 60, 90, 120]
WEIGHTS = {30: 0.4, 60: 0.3, 90: 0.2, 120: 0.1}

# Global flag for JSON mode (suppress prints to stdout)
_JSON_MODE = [False]  # Use list to allow modification in nested scope

def log_message(msg):
    """Print message to stderr in JSON mode, stdout otherwise"""
    if _JSON_MODE[0]:
        print(msg, file=sys.stderr)
    else:
        print(msg)

def set_json_mode(enabled):
    """Enable or disable JSON mode"""
    _JSON_MODE[0] = enabled

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

def _is_daily_data(df, date_col='ds'):
    """Verinin günlük mü yoksa saatlik mi olduğunu tespit et"""
    unique_dates = df[date_col].dt.date.nunique()
    rows_per_day = len(df) / max(unique_dates, 1)
    return rows_per_day < 2  # Gün başına ortalama < 2 satır ise günlük veri

def predict_next_day_prophet(csv_path):
    prophet_df, _ = load_csv_flexible(csv_path)

    prophet_df["date_only"] = prophet_df['ds'].dt.date
    daily_df = prophet_df.groupby("date_only").agg({'y': 'last'}).reset_index()
    daily_df.columns = ['ds', 'y']
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])

    last_date = daily_df['ds'].max()

    # Eksik gün kontrolü: sadece saatlik veri için uygula
    if not _is_daily_data(prophet_df):
        hours_in_last_day = prophet_df[prophet_df['ds'].dt.date == last_date.date()].shape[0]
        if hours_in_last_day < 20:
            log_message(f"Uyarı: Son gün ({last_date.date()}) eksik veri içeriyor ({hours_in_last_day} saat). Prophet tahmini yanıltıcı olabilir.")
            daily_df = daily_df[daily_df['ds'] < last_date]

            if daily_df.empty:
                log_message("Yeterli veri yok, tahmin yapılamıyor.")
                return None
            last_date = daily_df['ds'].max()
            log_message(f"Son tam gün olarak {last_date.date()} kullanılıyor.")

    log_message(f"Prophet modeli için veri aralığı: {daily_df['ds'].min().date()} - {daily_df['ds'].max().date()} ({len(daily_df)} gün)")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(daily_df)

    # Bir sonraki gün için tahmin
    next_day = last_date + pd.Timedelta(days=1)
    future = pd.DataFrame({'ds': [next_day]})
    forecast = model.predict(future)
    yhat = forecast.iloc[0]['yhat']
    log_message(f"Prophet ile {next_day.date()} için tahmin: {yhat:.2f} USD")

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

    # Eksik gün kontrolü: sadece saatlik veri için uygula
    is_daily = price_series.index.date.nunique() >= len(price_series) * 0.5
    if not is_daily:
        hourly_in_last_day = price_series[price_series.index.date == last_date_idx.date()].shape[0]
        if hourly_in_last_day < 20:
            log_message(f"Uyarı: Son gün ({last_date_idx.date()}) eksik veri içeriyor ({hourly_in_last_day} saat). ARIMA tahmini yanıltıcı olabilir.")
            daily_series = daily_series[:-1]

            if daily_series.empty:
                log_message("Yeterli veri yok, tahmin yapılamıyor.")
                return None
            log_message(f"Son tam gün olarak {daily_series.index[-1].date()} kullanılıyor.")

    # Son 365 gün ile sınırla (varsa)
    if len(daily_series) > 365:
        daily_series = daily_series.iloc[-365:]
        log_message(f"ARIMA modeli için son 365 gün kullanılıyor: {daily_series.index.min().date()} - {daily_series.index.max().date()}")

    # ARIMA modeli
    model = sm.tsa.ARIMA(daily_series, order=(1, 1, 1))
    arima_result = model.fit()

    # Son tarihi bul ve bir gün ekle
    last_date = daily_series.index.max()
    next_day = last_date + pd.Timedelta(days=1)
    forecast = arima_result.get_forecast(steps=1)
    predicted = forecast.predicted_mean.iloc[0]
    log_message(f"ARIMA ile {next_day.date()} için tahmin: {predicted:.2f} USD")

    # ← DEĞERİ DÖNDÜR
    return {
        'date': str(next_day.date()),
        'prediction': float(predicted)
    }

def get_daily_data(csv_path):
    """CSV'den günlük veri hazırla"""
    prophet_df, price_series = load_csv_flexible(csv_path)

    # Prophet için günlük data
    prophet_df["date_only"] = prophet_df['ds'].dt.date
    daily_df = prophet_df.groupby("date_only").agg({'y': 'last'}).reset_index()
    daily_df.columns = ['ds', 'y']
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])
    daily_df = daily_df.sort_values('ds').reset_index(drop=True)

    # ARIMA için günlük series
    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    daily_series = price_series.resample('D').last().ffill()

    return daily_df, daily_series


def predict_prophet_for_window(daily_df, window_days):
    """Belirli bir pencere için Prophet tahmini"""
    if len(daily_df) < window_days:
        return None

    # Son N günü al
    windowed_df = daily_df.iloc[-window_days:].copy()
    last_date = windowed_df['ds'].max()

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(windowed_df)

    next_day = last_date + pd.Timedelta(days=1)
    future = pd.DataFrame({'ds': [next_day]})
    forecast = model.predict(future)

    return {
        'prediction': round(float(forecast.iloc[0]['yhat']), 2),
        'data_range': f"{windowed_df['ds'].min().date()} to {windowed_df['ds'].max().date()}"
    }


def predict_arima_for_window(daily_series, window_days):
    """Belirli bir pencere için ARIMA tahmini"""
    if len(daily_series) < window_days:
        return None

    # Son N günü al
    windowed_series = daily_series.iloc[-window_days:]

    model = sm.tsa.ARIMA(windowed_series, order=(1, 1, 1))
    arima_result = model.fit()

    forecast = arima_result.get_forecast(steps=1)
    predicted = forecast.predicted_mean.iloc[0]

    return {
        'prediction': round(float(predicted), 2),
        'data_range': f"{windowed_series.index.min().date()} to {windowed_series.index.max().date()}"
    }


def calculate_ensemble(window_predictions, weights=WEIGHTS):
    """Ağırlıklı ensemble hesapla"""
    total_weight = 0
    weighted_sum = 0

    for window, data in window_predictions.items():
        if data is not None:
            days = int(window.replace('_days', ''))
            weight = weights.get(days, 0)
            weighted_sum += data['prediction'] * weight
            total_weight += weight

    if total_weight == 0:
        return None

    return {
        'prediction': round(weighted_sum / total_weight, 2),
        'weights': {f"{k}_days": v for k, v in weights.items() if f"{k}_days" in window_predictions}
    }


def predict_multi_window(csv_path):
    """Tüm pencereler için tahminler üret"""
    daily_df, daily_series = get_daily_data(csv_path)
    total_days = len(daily_df)

    # Son fiyat bilgisi
    latest_price = round(float(daily_df['y'].iloc[-1]), 2)
    last_date = daily_df['ds'].max()
    next_day = last_date + pd.Timedelta(days=1)

    log_message(f"Multi-window prediction başlatılıyor...")
    log_message(f"Toplam veri: {total_days} gün, Son tarih: {last_date.date()}")

    # Prophet tahminleri
    prophet_windows = {}
    for window in WINDOWS:
        if total_days >= window:
            log_message(f"Prophet {window} gün penceresi hesaplanıyor...")
            result = predict_prophet_for_window(daily_df, window)
            if result:
                prophet_windows[f"{window}_days"] = result

    # ARIMA tahminleri
    arima_windows = {}
    for window in WINDOWS:
        if total_days >= window:
            log_message(f"ARIMA {window} gün penceresi hesaplanıyor...")
            result = predict_arima_for_window(daily_series, window)
            if result:
                arima_windows[f"{window}_days"] = result

    # Ensemble hesapla
    prophet_ensemble = calculate_ensemble(prophet_windows)
    arima_ensemble = calculate_ensemble(arima_windows)

    # Combined ensemble (Prophet + ARIMA ortalaması)
    combined = None
    if prophet_ensemble and arima_ensemble:
        combined = {
            'prediction': round((prophet_ensemble['prediction'] + arima_ensemble['prediction']) / 2, 2),
            'method': 'average of prophet and arima ensembles'
        }

    # Ana tahminler (tüm veri ile)
    log_message("Ana tahminler (tüm veri ile) hesaplanıyor...")
    prophet_main = predict_next_day_prophet(csv_path)
    arima_main = predict_next_day_arima(csv_path)

    return {
        'data_info': {
            'total_days_available': total_days,
            'data_source': 'CoinGecko',
            'latest_price': latest_price
        },
        'prophet': {
            'date': str(next_day.date()),
            'prediction': round(prophet_main['prediction'], 2) if prophet_main else None,
            'windows': prophet_windows,
            'ensemble': prophet_ensemble
        },
        'arima': {
            'date': str(next_day.date()),
            'prediction': round(arima_main['prediction'], 2) if arima_main else None,
            'windows': arima_windows,
            'ensemble': arima_ensemble
        },
        'combined_ensemble': combined
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitcoin price prediction')
    parser.add_argument('csv_path', help='Path to CSV file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Dosya bulunamadı: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Enable JSON mode - all log messages go to stderr
        set_json_mode(True)
        # JSON output mode - only JSON goes to stdout
        result = predict_multi_window(args.csv_path)
        print(json.dumps(result, indent=2))
    else:
        # Legacy mode - mevcut çıktı formatı (backward compatibility)
        print("Prophet tahmini:")
        predict_next_day_prophet(args.csv_path)
        print("ARIMA tahmini:")
        predict_next_day_arima(args.csv_path)
