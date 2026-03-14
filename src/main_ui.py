import streamlit as st
from .log_forecast import log_forecast
from .general_view import general_view
from .baseline import baseline, arima_forecast
from .hypertune import hypertune_app
from .bitcoin_data import bitcoin_data

def main_ui():
    st.set_page_config(page_title="BTC Price Forecasting", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                               ["Bitcoin Verileri", "General View", "Baseline Model",
                                "ARIMA Forecast", "Log Forecast", "Hyperparameter Tuning"],
                               index=0)

    if page == "General View":
        general_view()
    elif page == "Baseline Model":
        baseline()
    elif page == "ARIMA Forecast":
        arima_forecast()
    elif page == "Bitcoin Verileri":
        bitcoin_data()
    elif page == "Log Forecast":
        log_forecast()
    elif page == "Hyperparameter Tuning":
        hypertune_app()