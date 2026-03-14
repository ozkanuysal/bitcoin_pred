# Bitcoin Prediction Projesi - Kapsamli Analiz Raporu

**Tarih:** 2026-03-14
**Analiz eden:** Claude Opus 4.6

---

## 1. PROJE GENEL BAKIS

### 1.1 Amac
Bitcoin (BTC/USD) fiyatini gunluk olarak tahmin eden, otomatik veri toplama ve prediction pipeline'i olan bir proje. Iki ana kullanim alani var:
1. **Otomatik pipeline** (GitHub Actions): Her gun CoinGecko'dan veri ceker, Prophet + ARIMA ile yarin icin tahmin uretir
2. **Streamlit UI** (yerel): Interaktif veri gorsellestirme, model fitting ve hyperparameter tuning

### 1.2 Teknoloji Stack
- **Dil:** Python 3.12 (pyproject), 3.10 (GitHub Actions)
- **Paket yonetimi:** uv (pyproject.toml + uv.lock)
- **ML modelleri:** Facebook Prophet, ARIMA (statsmodels)
- **Veri kaynaklari:** CoinGecko API (otomatik), Binance API (UI), Yahoo Finance (yardimci), Investing.com (eski manual CSV)
- **UI:** Streamlit
- **CI/CD:** GitHub Actions (2 workflow)
- **Gorsellestirme:** Plotly
- **Deploy (planli):** Docker + Fly.io (env.sh'de alias var ama Dockerfile yok)

---

## 2. DOSYA YAPISI VE AKIS

```
bitcoin_pred/
├── .github/workflows/
│   ├── bitcoin.yml          # Veri cekme workflow (00:00 UTC)
│   └── predict.yml          # Tahmin workflow (03:00 UTC)
├── data/
│   ├── BTC_USD_{N}days.csv  # CoinGecko'dan cekilmis gunluk veriler (son 10 dosya tutulur)
│   └── metadata.json        # Son veri cekme bilgisi
├── prediction/
│   └── YYYY_MM_DD_predictions.json  # Gunluk tahminler (162 dosya, Ekim 2025'ten beri)
├── src/
│   ├── main_ui.py           # Streamlit uygulama giris noktasi
│   ├── data.py              # Eski CSV yukleyici (Bitcoin_Historical_Data.csv icin)
│   ├── bitcoin_data.py      # Binance canli veri sayfasi (Streamlit)
│   ├── binance_data.py      # Binance API wrapper
│   ├── yahoo_data.py        # Yahoo Finance veri cekici
│   ├── fetch_data_script.py # CoinGecko veri cekme script'i (GitHub Actions kullanir)
│   ├── quick_predict.py     # Ana tahmin motoru (Prophet + ARIMA, multi-window)
│   ├── baseline.py          # Baseline Prophet + ARIMA Streamlit sayfalari
│   ├── log_forecast.py      # Log-transformed Prophet modeli (Streamlit)
│   ├── general_view.py      # Genel istatistik/gorsel sayfasi (Streamlit)
│   └── hypertune.py         # Prophet hyperparameter tuning (Streamlit)
├── Bitcoin_Historical_Data.csv      # Manuel veri (investing.com, 31 satir, Ekim-Kasim 2025)
├── Bitcoin_Historical_Data_old.csv  # Eski manuel veri (365 satir, Ekim 2024 - Agustos 2025)
├── main.py                  # Streamlit entry point
├── env.sh                   # Shell alias'lari
├── pyproject.toml           # Proje konfigurasyonu
├── requirements.txt         # Bagimliliklar
└── test.py                  # Basit test (sadece data yukler ve head() yazdirir)
```

---

## 3. OTOMATIK PIPELINE AKISI (GitHub Actions)

### 3.1 Workflow 1: Veri Cekme (`bitcoin.yml`)
- **Zamanlama:** Her gun 00:00 UTC + push to main + manual trigger
- **Islem:**
  1. `metadata.json`'dan son `days` degerini okur
  2. `days + 1` ile `fetch_data_script.py` calistirir
  3. CoinGecko API'den N gunluk BTC/USD verisi ceker
  4. `data/BTC_USD_{N}days.csv` olarak kaydeder
  5. Eski CSV'leri temizler (son 10 dosya kalir)
  6. Git commit + push

### 3.2 Workflow 2: Tahmin (`predict.yml`)
- **Zamanlama:** Her gun 03:00 UTC + push to main + manual trigger
- **Islem:**
  1. En son CSV dosyasini bulur (`ls -t`)  ← **KRiTiK BUG (bkz. Bolum 4.1)**
  2. `quick_predict.py` ile multi-window tahmin uretir
  3. JSON dosyasina yarin tarihi ve creation date ekler (jq ile)
  4. `prediction/` klasorune kaydeder
  5. Git commit + push

### 3.3 Tahmin Motoru (`quick_predict.py`) - Detayli Calisma Sekli
1. **CSV Okuma:** `load_csv_flexible()` — timestamp + fiyat kolonu otomatik tespit eder
2. **Gunluk Veri Hazirlama:** `get_daily_data()` — saatlik/gunluk veriyi gunluk'e cevirir
3. **Multi-Window Tahmin:** 4 farkli pencere (30, 60, 90, 120 gun) ile ayri ayri tahmin:
   - Her pencere icin Prophet modeli fit edilir
   - Her pencere icin ARIMA(1,1,1) modeli fit edilir
4. **Ensemble:** Pencere tahminleri agirlikli ortalama ile birlestirilir
   - Agirliklar: 30 gun = 0.4, 60 gun = 0.3, 90 gun = 0.2, 120 gun = 0.1
5. **Ana Tahmin:** Tum veri ile tek Prophet + tek ARIMA tahmini
6. **Combined Ensemble:** Prophet ve ARIMA ensemble ortalamalari birlestirilir

---

## 4. TESPiT EDiLEN SORUNLAR VE BUGLAR

### 4.1 ★★★ KRiTiK BUG: Yanlis CSV Dosyasi Secimi (predict.yml)

**Sorun:**
```yaml
# predict.yml, satir 39
latest_csv=$(ls -t data/BTC_USD_*.csv | head -n 1)
```
`ls -t` dosyalari **modification time**'a gore siralar. Ancak `actions/checkout@v4` tum dosyalarin mtime'ini **checkout anina** ayarlar. Bu nedenle 10 CSV dosyasindan hangisinin secilecegi **tamamen rastgele/tahmin edilemez**dir.

**Kanit:** 162 tahmin dosyasi analiz edildiginde:
- Tahminlerin **buyuk cogunlugu 10 gun eski** veriyle yapilmis
- `total_days_available` degeri asla guncel CSV ile uyusmuyor
- Ornek: 2026-03-15 icin tahmin yapilirken 240 gunluk CSV yerine **231 gunluk** (9 gun onceki) CSV kullanilmis

**Detayli istatistikler:**
| Tarih Araligi | Tipik Staleness |
|---|---|
| Aralik 2025 | 3-10 gun |
| Ocak 2026 | 9-10 gun |
| Subat 2026 | 9-10 gun |
| Mart 2026 | 2-10 gun |

**Etki:** Tum multi-window tahminler (Aralik 2025'ten beri) yanlis veriye dayanmaktadir. Tahminler gercek "yarin" icin degil, **10 gun oncenin yarini** icin yapilmaktadir.

**Cozum onerisi:**
```yaml
# Secimlik 1: Dosya adindaki gun numarasina gore en buyugu sec
latest_csv=$(ls data/BTC_USD_*.csv | sort -V | tail -n 1)

# Secimlik 2: metadata.json'dan oku
latest_csv="data/$(jq -r '.filename' data/metadata.json)"
```

---

### 4.2 ★★★ KRiTiK BUG: Tarih Uyumsuzlugu (prediction_date vs model date)

**Sorun:** JSON ciktidaki iki farkli tarih FARKLI kaynaklardan geliyor:
- `prediction_date`: Workflow'un shell'inde `date -d "tomorrow"` ile hesaplanan yarin tarihi
- `prophet.date` / `arima.date`: Python'da CSV verisinin son tarihine +1 gun eklenmesi

**Sonuc:** Ikisi ASLA uyusmuyor cunku:
1. CSV zaten eski (Bug 4.1)
2. Ayrica Python'daki eksik gun kontrolu (`hours_in_last_day < 20`) son gunu DE siliyor

**Ornekler:**
```
2026_03_15_predictions.json:
  prediction_date = "2026-03-15"  (workflow: yarin)
  prophet.date    = "2026-03-05"  (python: CSV'deki son tarih + 1)
  → 10 gun fark!
```

**Etki:** Kullanicinin "2026-03-15 tahmini" olarak gordugu sey aslinda "2026-03-05 tahmini"dir.

---

### 4.3 ★★ ONEMLI BUG: Eksik Gun Kontrolu Gunluk Veride Yanlis Calisiyor

**Sorun:** `quick_predict.py` satirlari 66-69:
```python
hours_in_last_day = prophet_df[prophet_df['ds'].dt.date == last_date.date()].shape[0]
if hours_in_last_day < 20:
    daily_df = daily_df[daily_df['ds'] < last_date]
```

Bu kontrol **saatlik veri** icin tasarlanmis (20 saatten az veri varsa gunu eksik say). Ancak CoinGecko 90+ gun icin **gunluk** veri donduruyor. Gunluk veride her tarih icin **1 satir** var, yani `hours_in_last_day = 1 < 20` HER ZAMAN dogru oluyor. Bu nedenle **son gun HER ZAMAN siliniyor**.

**Etki:** Model, var olan verinin son gununu kullanmiyor. Ek 1 gunluk veri kaybi.

**Cozum:** Veri granularitesini kontrol et; gunluk veride bu kontrolu devre disi birak veya sadece saatlik veride uygula.

---

### 4.4 ★★ ONEMLI: Streamlit UI Bozuk (Hardcoded Yol)

**Sorun:** `src/data.py` satir 40:
```python
file_path = '/home/ozkan/Desktop/bitcoin/bitcoin_pred/Bitcoin_Historical_Data.csv'
```
Bu yol **Linux** icin ve **eski bir dizin yapisi** icin gecerli. Kullanicinin Mac'inde yol:
`/Users/ozkanuysal/Desktop/bitcoin_pred/Bitcoin_Historical_Data.csv`

Bu fonksiyon `general_view()`, `baseline()`, `arima_forecast()` ve `hypertune_app()` tarafindan kullaniliyor. **Streamlit UI'nin bu sayfalari calismayacak.**

---

### 4.5 ★★ ONEMLI: Kullanilmayan Importlar ve Olü Kod (main_ui.py)

**Sorun:** `main_ui.py` su fonksiyonlari import ediyor ama kullanmiyor:
```python
from .log_forecast import log_forecast      # ← KULLANILMIYOR
from .hypertune import hypertune_app         # ← KULLANILMIYOR
```
Sidebar'daki selectbox'ta sadece 4 sayfa var: "Bitcoin Verileri", "General View", "Baseline Model", "ARIMA Forecast". `log_forecast` ve `hypertune_app` erisileemiyor.

---

### 4.6 ★ KUCUK: Eksik Bagimlilklar

- `binance_data.py` `python-binance` paketini kullaniyor (`from binance.client import Client`) ama bu paket ne `requirements.txt`'de ne de `pyproject.toml`'da listelenmis. `bitcoin_data.py` (Streamlit Binance sayfasi) calismayacak.
- `matplotlib` `bitcoin_data.py`'de import edilmis ama kullanilmiyor
- GitHub Actions'ta farkli versiyonlar: `numpy<2.0` ve `prophet==1.1.5` (pyproject: `numpy>=2.3.1`, `prophet>=1.1.7`)

---

### 4.7 ★ KUCUK: CoinGecko Veri Resample Kafa Karistirici

**Sorun:** `fetch_data_script.py` satir 51:
```python
df_hourly = df.resample("1H").agg({'close': 'last', 'volume': 'sum'})
```
90+ gun sorgulanduginda CoinGecko zaten **gunluk** veri donduruyor. Bu resample islemsiz kalir (no-op), hata vermez ama kodun amaci ile gercekligi uyusmuyor.

Ayrica `df_hourly['open'] = df_hourly['close']` ile open=close yapiliyor — teknik olarak yanlis ama tahminleri etkilemiyor (ikisi de ayni deger).

---

### 4.8 ★ KUCUK: Bitcoin_Historical_Data.csv Guncel Degil

Root'taki `Bitcoin_Historical_Data.csv` sadece 31 satir (Ekim-Kasim 2025, investing.com'dan). Bu dosya artik Streamlit UI disinda kullanilmiyor ve `data.py`'deki path de yanlis. `_old.csv` versiyonu 365 satir iceriyor (Ekim 2024 - Agustos 2025). Her iki dosya da otomatik pipeline tarafindan KULLANILMIYOR — bunlar eski, manual veri.

---

### 4.9 ★ KUCUK: test.py Gereksiz

`test.py` sadece `load_data()` cagirip `head()` yazdiriyor. Gercek bir test framework'u yok (pytest vs.), herhangi bir assertion yok. `airflow/test.py` tamamen bos.

---

### 4.10 ★ KUCUK: predict.yml Push-Trigger Race Condition

`predict.yml` hem `schedule` hem `push` ile tetikleniyor. `bitcoin.yml` default GITHUB_TOKEN ile push yaptiginda normalde `predict.yml`'i tetiklemez (GitHub'un loop korunmasi). Ancak bir insan push yaparsa, her iki workflow da ayni anda tetiklenir — predict.yml, bitcoin.yml'den onceki verileri kullanabilir.

---

## 5. TAHMiN YONTEMLERi DEGERLENDIRMESI

### 5.1 Prophet Modeli
- **Konfigürasyon:** `yearly_seasonality=True`, `weekly_seasonality=True`
- **Sorun:** 30-240 gun veri ile yillik mevsimsellik ogrenilMEZ (en az 2 yillik veri gerekir)
- **Iyi yani:** Haftalik mevsimsellik kripto icin anlamli olabilir (hafta sonu/hafta ici fiyat farki)

### 5.2 ARIMA Modeli
- **Konfigürasyon:** Sabit ARIMA(1,1,1), parametre optimize edilMEMIS
- **Sorun:** (p,d,q) = (1,1,1) en basit konfigürasyon. AIC/BIC bazli model secimi yapilmiyor.
- **Iyi yani:** Basit ve hizli, overfitting riski dusuk

### 5.3 Ensemble Yontemi
- **Agirliklar:** 30d=0.4, 60d=0.3, 90d=0.2, 120d=0.1 (kisa vadeye agirlik)
- **Sorun:** Agirliklar sabit, tarihsel performansa gore ayarlanmiyor
- **Combined:** Prophet ensemble + ARIMA ensemble basit aritmetik ortala

### 5.4 Tahmin Dogrlugu
Veritabaninda dogruluk olcumu yapilmiyor. Gercek fiyatla tahmin karsilastirilmiyor. Projenin tahminlerinin ne kadar dogru oldugu bilinmiyor.

---

## 6. VERI AKIS DIYAGRAMI

```
CoinGecko API
     │
     ▼ (bitcoin.yml, 00:00 UTC)
fetch_data_script.py
     │
     ▼
data/BTC_USD_{N}days.csv + metadata.json
     │
     ▼ (predict.yml, 03:00 UTC)
quick_predict.py ← CSV secimi BURADA BOZUK (ls -t)
     │
     ├── Prophet (4 pencere + ana tahmin)
     ├── ARIMA (4 pencere + ana tahmin)
     └── Ensemble (agirlikli + birlesik)
     │
     ▼
prediction/YYYY_MM_DD_predictions.json
```

---

## 7. PROJE GELiSiM KRONOLOJISI (Git commit ve prediction formatindan cikarim)

| Donem | Gelisme |
|---|---|
| Ekim 2025 (baslangic) | Basit Prophet + ARIMA, eski format (sadece tarih ve tahmin) |
| ~24 Ekim 2025 | `prediction_date` ve `created_at` eklendi |
| ~7 Aralik 2025 | Multi-window sistem, `data_info`, ensemble eklendi |
| ~Aralik 2025 - Mart 2026 | Pipeline calisir ama **yanlis CSV kullanir** (Bug 4.1) |

---

## 8. STREAMLIT UI YAPISI

```
Ana Sayfa (main.py → main_ui.py)
├── "Bitcoin Verileri" → bitcoin_data() [Binance canli veri, mum grafigi]
├── "General View"     → general_view() [Istatistik, EMA, volatilite - BOZUK: yanlis yol]
├── "Baseline Model"   → baseline() [Prophet fit + 30 gun tahmin - BOZUK: yanlis yol]
└── "ARIMA Forecast"   → arima_forecast() [ARIMA fit + 30 gun tahmin - BOZUK: yanlis yol]

ERISILEMEYEN SAYFALAR:
× "Log Forecast"       → log_forecast() [Log-transformed Prophet - import edilmis ama menu'de yok]
× "Hypertune"           → hypertune_app() [Prophet CV tuning - import edilmis ama menu'de yok]
```

---

## 9. OZET

### Calisan Kisimlar
- CoinGecko veri cekme pipeline'i (bitcoin.yml) ✓
- Prophet + ARIMA tahmin motoru (quick_predict.py) ✓ (kod mantigi dogru, ama yanlis veriye uygulanmakta)
- Git otomatik commit/push ✓
- CSV temizlik (son 10 dosya) ✓

### Bozuk / Sorunlu Kisimlar
- **CSV dosya secimi** — tahminler ~10 gun eski veriye dayaniyor (KRiTiK)
- **Tarih uyumsuzlugu** — prediction_date ile model date farkli (KRiTiK)
- **Eksik gun kontrolu** — gunluk veride gereksiz veri kaybi (ONEMLI)
- **Streamlit UI** — General View, Baseline, ARIMA sayfalari calismayacak (ONEMLI)
- **Eksik bagimliliklar** — python-binance paketi listede yok (ORTA)
- **Olu kod** — log_forecast ve hypertune menude yok (DUSUK)

---

## 10. NOT

Bu analiz projenin 2026-03-14 tarihli git snapshot'ina dayanmaktadir. Tum kaynak dosyalar, veri dosyalari, workflow'lar ve 162 prediction JSON dosyasi incelenmistir. Sorunlar dogrudan koddan ve veri analizinden kanit gosterilerek belirlenmistir.
