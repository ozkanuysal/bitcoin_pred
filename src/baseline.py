import streamlit as st
from prophet import Prophet
import pandas as pd
import statsmodels.api as sm
from prophet.plot import plot_components_plotly, plot_plotly
import plotly.express as px
from .data import load_data
import warnings

warnings.filterwarnings("ignore")


def fit_prophet_model(df):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    return model

def baseline():
    st.title('Baseline Model')
    st.info("We fit directly on the price data, using the default parameters for the Prophet model.")
    
    # Load data at the beginning
    df = load_data()
    if df.empty:
        st.warning("No data available to fit the model.")
        return

    df = df.rename(columns={'price': 'y', 'timestamp': 'ds'})

    # Fit the model
    with st.spinner('Training the forecasting model...'):
        model = fit_prophet_model(df)

    # Create future dates for forecasting (30 days)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    main_plot = plot_plotly(model, forecast)
    fig_components = plot_components_plotly(model, forecast)
    
    st.subheader('Forecast and Components Plots')
    st.plotly_chart(main_plot, use_container_width=True)
    st.plotly_chart(fig_components, use_container_width=True)

    # Show next 30 days prediction summary
    st.subheader("Next 30 Days Bitcoin Price Forecast")
    forecast_next_30 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
    st.dataframe(forecast_next_30, use_container_width=True)

def arima_forecast():
    st.title('ARIMA Forecast')
    st.info("This page uses the ARIMA model from Statsmodels to forecast Bitcoin prices.")

    # Load data
    df = load_data()
    if df.empty:
        st.warning("No data available for ARIMA forecast.")
        return

    # Use last 365 days for ARIMA fitting for speed and relevance
    df_arima = df.set_index('timestamp').copy()
    
    # Ensure the index is a DatetimeIndex
    if not isinstance(df_arima.index, pd.DatetimeIndex):
        df_arima.index = pd.to_datetime(df_arima.index)
    
    price_series = df_arima['price'].dropna()
    
    if len(price_series) > 365:
        price_series = price_series.iloc[-365:]
        
    try:
        # Simple ARIMA(1,1,1) model
        model = sm.tsa.ARIMA(price_series, order=(1, 1, 1))
        arima_result = model.fit()
        
        forecast_steps = 30
        forecast = arima_result.get_forecast(steps=forecast_steps)
        
        # Correctly generate forecast index
        forecast_index = pd.date_range(start=price_series.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        
        conf_int = forecast.conf_int()
        
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': forecast.predicted_mean,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        })
        
        fig_arima = px.line(forecast_df, x='date', y='forecast', title='ARIMA Forecast (Next 30 Days)')
        fig_arima.add_scatter(x=forecast_df['date'], y=forecast_df['lower'], mode='lines', name='Lower CI', line=dict(dash='dash'))
        fig_arima.add_scatter(x=forecast_df['date'], y=forecast_df['upper'], mode='lines', name='Upper CI', line=dict(dash='dash'))
        
        st.plotly_chart(fig_arima)
        st.dataframe(forecast_df, use_container_width=True)
        
    except Exception as e:
        st.warning(f'ARIMA model failed: {e}')