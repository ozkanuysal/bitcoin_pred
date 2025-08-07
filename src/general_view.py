import pandas as pd
import streamlit as st
import plotly.express as px
from .data import load_data

def general_view(): 
    df = load_data()
    st.title('General View of the BTC Price Data')
    
    st.subheader('Descriptive Statistics (df.describe())')
    st.write(df.describe())

    st.subheader('First 5 rows of the dataset')
    st.write(df.head())

    # Calculate 20-day moving average
    df['MA200'] = df['price'].rolling(window=200).mean()

    fig = px.line(df, x='timestamp', y='price', title='BTC Price Over Time')
    # Add the moving average trace
    fig.add_scatter(x=df['timestamp'], y=df['MA200'], mode='lines', name='200-Day MA')
    st.plotly_chart(fig)

    fig_log = px.line(df, x='timestamp', y='price', title='BTC Price Over Time (Log Scale)', log_y=True)
    # Add the moving average trace to the log chart
    fig_log.add_scatter(x=df['timestamp'], y=df['MA200'], mode='lines', name='200-Day MA')
    st.plotly_chart(fig_log)

    