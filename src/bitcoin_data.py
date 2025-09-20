import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .binance_data import BinanceDataFetcher
import matplotlib.pyplot as plt

def bitcoin_data():
    st.title("Bitcoin Fiyat Verileri")
    
    # Yan panel seçenekleri
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Veri Parametreleri")
        
        days = st.slider("Gün Sayısı", min_value=7, max_value=365, value=30, step=1)
        symbol = st.selectbox("Sembol", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"], index=0)
        interval = st.selectbox("Zaman Aralığı", 
                               ["1d", "4h", "1h", "15m", "5m"],
                               index=0,
                               format_func=lambda x: {
                                   "1d": "Günlük", 
                                   "4h": "4 Saatlik",
                                   "1h": "Saatlik",
                                   "15m": "15 Dakikalık",
                                   "5m": "5 Dakikalık"
                               }.get(x, x))
        
        fetch_button = st.button("Verileri Getir")
    
    # Binance client oluştur
    binance = BinanceDataFetcher()
    
    if fetch_button or 'df_btc' not in st.session_state:
        with st.spinner('Veriler yükleniyor...'):
            df = binance.get_historical_klines(symbol=symbol, interval=interval, days_ago=days)
            
            if not df.empty:
                # Verileri ters çevir (yeni tarihten eskiye doğru sırala)
                df = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
                
                # Fiyat değişimi hesapla
                df['price_change'] = df['close'].diff(-1)  # Bir önceki güne göre değişim
                df['price_change_pct'] = (df['close'] / df['close'].shift(-1) - 1) * 100  # Yüzdelik değişim
                
                st.session_state['df_btc'] = df
                st.session_state['symbol'] = symbol
                st.session_state['interval'] = interval
                st.session_state['days'] = days
            else:
                st.error("Veri çekilemedi. Lütfen tekrar deneyin.")
    
    with col2:
        if 'df_btc' in st.session_state:
            df = st.session_state['df_btc']
            current_symbol = st.session_state['symbol']
            current_interval = st.session_state['interval']
            
            # Güncel fiyat
            try:
                current_price = binance.get_current_price(symbol=current_symbol)
                st.metric(
                    label=f"Güncel {current_symbol} Fiyatı", 
                    value=f"${current_price:.2f}",
                    delta=f"{((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100):.2f}%"
                )
            except:
                st.warning("Güncel fiyat alınamadı.")
            
            # Alt alta iki grafik oluştur (mum grafiği ve değişim grafiği)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Mum grafiği ekle
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=current_symbol
            ), row=1, col=1)
            
            # Değişim grafiği ekle (bar chart)
            colors = ['red' if x < 0 else 'green' for x in df['price_change_pct']]
            fig.add_trace(go.Bar(
                x=df['timestamp'], 
                y=df['price_change_pct'],
                name='Günlük Değişim %',
                marker_color=colors
            ), row=2, col=1)
            
            # Grafiği özelleştir
            fig.update_layout(
                title=f'{current_symbol} Son {st.session_state["days"]} Gün ({current_interval})',
                xaxis_title='Tarih',
                yaxis_title='Fiyat (USDT)',
                xaxis_rangeslider_visible=False,
                height=700
            )
            
            # İkinci grafik için y ekseni başlığı
            fig.update_yaxes(title_text="Değişim %", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # İstatistikler
            st.subheader("İstatistikler")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("En Yüksek", f"${df['high'].max():.2f}")
                st.metric("Ortalama Hacim", f"{df['volume'].mean():.2f} BTC")
                st.metric("En Yüksek Günlük Artış", f"+{df['price_change_pct'].max():.2f}%")
            
            with stats_col2:
                st.metric("En Düşük", f"${df['low'].min():.2f}")
                st.metric("Standart Sapma", f"${df['close'].std():.2f}")
                st.metric("En Büyük Günlük Düşüş", f"{df['price_change_pct'].min():.2f}%")
            
            with stats_col3:
                st.metric("Ortalama", f"${df['close'].mean():.2f}")
                st.metric("Değişim %", f"{((df['close'].iloc[0] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100):.2f}%")
                st.metric("Ort. Günlük Değişim", f"{df['price_change_pct'].mean():.2f}%")
            
            # Ham veriyi göster
            with st.expander("Ham Veriyi Göster"):
                # Price change sütunlarını güzelleştir
                display_df = df.copy()
                display_df['price_change'] = display_df['price_change'].map('${:,.2f}'.format)
                display_df['price_change_pct'] = display_df['price_change_pct'].map('{:,.2f}%'.format)
                
                st.dataframe(display_df)
                
                # CSV indirme düğmesi
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="CSV İndir",
                    data=csv,
                    file_name=f'{current_symbol}_{current_interval}_{st.session_state["days"]}days.csv',
                    mime='text/csv',
                )