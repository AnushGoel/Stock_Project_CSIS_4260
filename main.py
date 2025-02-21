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
from datetime import timedelta

# ========================== Load Parquet File with Fix ==========================
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_parquet(file_path)
    df.rename(columns={'date': 'Date', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # ✅ Convert Date to datetime
    df.set_index('Date', inplace=True)  # ✅ Set datetime index
    return df

df = load_stock_data('scaled_dataset_1x_snappy.parquet')

# ========================== Sidebar Options ==========================
st.sidebar.header("📊 Stock Analysis Options")
company_list = df['name'].unique()
company = st.sidebar.selectbox("Select Company", company_list)

# Time Range Selection
time_options = {"1 Year": 252, "6 Months": 126, "3 Months": 63, "1 Month": 21}
selected_time_range = st.sidebar.radio("Select Time Range", list(time_options.keys()))
time_range = time_options[selected_time_range]

forecast_days = st.sidebar.slider("Forecast Days", min_value=10, max_value=60, step=5)

# Filter Data for Selected Company & Time Range
company_data = df[df['name'] == company].iloc[-time_range:]

# ✅ Ensure 'Date' is a datetime index for forecasting
company_data.index = pd.to_datetime(company_data.index)

# ========================== LSTM Model ==========================
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

# ========================== Visualization Selection ==========================
st.subheader(f"📈 {company} - Stock Analysis")
plot_option = st.selectbox("Select Plot", ["Candlestick Chart", "LSTM Forecast"])

if plot_option == "Candlestick Chart":
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)
    fig.add_trace(go.Candlestick(x=company_data.index, open=company_data['open'], high=company_data['high'],
                                 low=company_data['low'], close=company_data['Close'], name="Candlesticks"), row=1, col=1)
    fig.add_trace(go.Bar(x=company_data.index, y=company_data['Volume'], name="Volume"), row=2, col=1)
    fig.update_layout(title=f"{company} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

elif plot_option == "LSTM Forecast":
    st.write("Training LSTM model, please wait...")
    model, scaler = train_lstm_model(company_data, forecast_days)
    
    # ✅ Generate future dates based on last available date
    last_date = company_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # ✅ Get forecasted closing prices
    future_predictions = model.predict(scaler.transform(company_data['Close'].values.reshape(-1, 1))[-forecast_days:])
    
    # ✅ Create forecast dataframe with Dates & Predicted Prices
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions.flatten()})
    
    st.write("### 📈 Forecasted Stock Prices")
    st.dataframe(forecast_df)  # ✅ Show table with forecasted prices
    st.line_chart(forecast_df.set_index("Date"))  # ✅ Show forecast plot
