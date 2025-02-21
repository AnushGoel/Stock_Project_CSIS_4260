import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import time

# Load Parquet File
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_parquet(file_path)
    df.rename(columns={'date': 'Date', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.set_index('Date', inplace=True)
    return df

df = load_stock_data('scaled_dataset_1x_snappy.parquet')

# ✅ Use 'name' column (Correct case-sensitive name)
st.sidebar.header("Stock Analysis Options")
company_list = df['name'].unique()
company = st.sidebar.selectbox("Select Company", company_list)

# Filter Data for Selected Company
company_data = df[df['name'] == company]

# Display Stock Data
st.write(f"### Stock Data for {company}")
st.dataframe(company_data.tail(10))

# Candlestick Chart
def candlestick_chart(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.2, subplot_titles=('Stock Price', 'Volume'),
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], 
                                 low=data['low'], close=data['Close'], name="Candlesticks"), row=1, col=1)

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume"), row=2, col=1)

    fig.update_layout(title=f"{company} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# LSTM Model
def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(data, forecast_days):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - forecast_days):
        X.append(scaled_data[i:i+forecast_days])
        y.append(scaled_data[i+forecast_days])

    X, y = np.array(X), np.array(y)
    
    model = create_lstm_model(input_shape=(X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    return model, scaler

# User Selects Visualization
plot_option = st.selectbox("Select Plot", ["Candlestick Chart", "LSTM Forecast"])

if plot_option == "Candlestick Chart":
    candlestick_chart(company_data)

elif plot_option == "LSTM Forecast":
    st.write("Training LSTM model, please wait...")
    start_time = time.time()
    model, scaler = train_lstm_model(company_data, forecast_days=30)
    end_time = time.time()
    st.success(f"LSTM Model Trained in {round(end_time - start_time, 2)} seconds!")

    future_predictions = model.predict(scaler.transform(company_data['Close'].values.reshape(-1, 1))[-30:])
    
    # Display Forecast
    st.write(f"### 30-Day Forecast for {company}")
    st.line_chart(future_predictions)
