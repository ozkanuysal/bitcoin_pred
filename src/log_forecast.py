import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
from .data import load_data
import plotly.graph_objects as go

def fit_prophet_model(df): 
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    return model

def log_forecast(): 
    st.title('Log Forecast Model')
    st.info("We fit on the log of the price data, using the default parameters for the Prophet model. You can adjust the hyperparameters to see how they affect the model.")
    # Add hyperparameter inputs to the sidebar
    st.sidebar.header("Log Model Hyperparameters")
    # Default values based on Prophet's defaults
    cp_prior_scale = st.sidebar.number_input(
        'Changepoint Prior Scale (Trend Flexibility)', 
        min_value=0.001, 
        max_value=0.5, 
        value=0.05, # Prophet default
        step=0.01,
        format="%.3f"
    )
    seasonality_prior_scale = st.sidebar.number_input(
        'Seasonality Prior Scale (Seasonality Strength)', 
        min_value=0.01, 
        max_value=10.0, 
        value=10.0, # Prophet default
        step=0.1,
        format="%.2f"
    )

    df = load_data()
    df = df.copy() 
    df = df.rename(columns={'price': 'y', 'timestamp': 'ds'})
    # Store original y before log transform for the final plot
    df['y_orig'] = df['y']
    df['y'] = np.log(df['y']) 
    
    fit_model = st.button("Fit Model on Log Prices")
    if not fit_model:
        return

    # Fit the model using selected hyperparameters
    with st.spinner(f'Training the forecasting model with C:{cp_prior_scale}, S:{seasonality_prior_scale}...'):
        model = Prophet(
            changepoint_prior_scale=cp_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        ).fit(df[['ds', 'y']])

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)


        main_plot = plot_plotly(model, forecast)
        fig_components = plot_components_plotly(model, forecast)

    st.subheader('Forecast and Components Plots')
    st.plotly_chart(main_plot, use_container_width=True)
    st.plotly_chart(fig_components, use_container_width=True)

    # Transform predictions back to original scale FOR DISPLAY, not for plotting func
    st.subheader('Forecast on Original Scale')
    forecast = forecast.copy()
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    df['y'] = np.exp(df['y'])
    # Create a new plot for the original scale
    fig_original_scale = go.Figure()

    # Add actual prices (original scale)
    fig_original_scale.add_trace(go.Scatter(
        x=df['ds'], 
        y=df['y'], 
        name='Actual', 
        mode='markers',
        marker=dict(color='black', size=4)
    ))

    # Add forecast (original scale)
    fig_original_scale.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        name='Forecast', 
        mode='lines', 
        line=dict(color='blue')
    ))

    # Add confidence interval (original scale)
    fig_original_scale.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,255,0.1)'),
        name='Upper Confidence Interval',
    ))
    # Add lower confidence interval (original scale)
    fig_original_scale.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill=None, 
        mode='lines',   
        line=dict(color='rgba(0,0,255,0.1)'),
        name='Lower Confidence Interval',
    ))
    
    fig_original_scale.update_layout(yaxis_type="log")
    
    st.plotly_chart(fig_original_scale, use_container_width=True)