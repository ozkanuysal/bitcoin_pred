import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
from .data import load_data

def fit_prophet_model(df): 
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    return model

def baseline(): 
    st.title('Baseline Model')
    st.info("We fit directly on the price data, using the default parameters for the Prophet model.")
    fit_model = st.button("Fit Baseline Model")
    if not fit_model:
        return

    df = load_data()
    df = df.copy() 
    df = df.rename(columns={'price': 'y', 'timestamp': 'ds'})

    # Fit the model
    with st.spinner('Training the forecasting model...'):
        model = fit_prophet_model(df)

    # Create future dates for forecasting (365 days)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    main_plot = plot_plotly(model, forecast)
    fig_components = plot_components_plotly(model, forecast)
    st.subheader('Forecast and Components Plots')
    st.plotly_chart(main_plot, use_container_width=True)
    st.plotly_chart(fig_components, use_container_width=True)