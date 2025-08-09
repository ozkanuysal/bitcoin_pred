import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from .data import load_data
import statsmodels.api as sm

def general_view(): 
    df = load_data()
    if df.empty:
        st.error("No valid data available for display.")
        return
    st.subheader('Descriptive Statistics')
    st.write(df.describe())

    st.subheader('First 5 rows of the dataset')
    st.write(df.head())

    # Logarithmic Growth Curve
    df['log_price'] = np.log(df['price'])
    fig_log_growth = px.line(df, x='timestamp', y='log_price', title='Logarithmic Growth Curve of BTC Price')
    st.plotly_chart(fig_log_growth)

    # Exponential Moving Averages (EMA)
    df['EMA20'] = df['price'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['price'].ewm(span=50, adjust=False).mean()
    fig_ema = px.line(df, x='timestamp', y='price', title='BTC Price with 20-day and 50-day EMAs')
    fig_ema.add_scatter(x=df['timestamp'], y=df['EMA20'], mode='lines', name='EMA 20')
    fig_ema.add_scatter(x=df['timestamp'], y=df['EMA50'], mode='lines', name='EMA 50')
    st.plotly_chart(fig_ema)
    # Calculate 200-day moving average
    df['MA200'] = df['price'].rolling(window=200).mean()
    # Price and 200-day MA
    fig = px.line(df, x='timestamp', y='price', title='BTC Price Over Time')
    fig.add_scatter(x=df['timestamp'], y=df['MA200'], mode='lines', name='200-Day MA')
    st.plotly_chart(fig)

    # Daily returns
    df['returns'] = df['price'].pct_change()
    fig_returns = px.line(df, x='timestamp', y='returns', title='Daily Returns (%)')
    st.plotly_chart(fig_returns)

    # Rolling volatility (30-day std of returns)
    df['volatility_30d'] = df['returns'].rolling(window=30).std()
    fig_vol = px.line(df, x='timestamp', y='volatility_30d', title='30-Day Rolling Volatility')
    st.plotly_chart(fig_vol)

    # Histogram of daily returns
    fig_hist = px.histogram(df, x='returns', nbins=50, title='Histogram of Daily Returns')
    st.plotly_chart(fig_hist)

    # Scatter plot: price vs. 200-day MA
    fig_scatter = px.scatter(df, x='MA200', y='price', title='Price vs. 200-Day Moving Average')
    st.plotly_chart(fig_scatter)