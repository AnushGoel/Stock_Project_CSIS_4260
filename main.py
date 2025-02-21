import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ========================== Filter Companies with at Least 6 Months of Data ==========================
min_required_days = 126  # Approx. 6 months of trading days (21 days per month)
valid_companies = [company for company in df['name'].unique() if len(df[df['name'] == company]) >= min_required_days]
df = df[df['name'].isin(valid_companies)]  # Remove companies with less than 6 months of data

# ========================== Sidebar Options ==========================
st.sidebar.header("📊 Stock Analysis Options")
company_list = valid_companies
company = st.sidebar.selectbox("Select Company", company_list)

# Forecasting Range (Limited to 10 days - 6 months)
forecast_days = st.sidebar.slider("Forecast Days", min_value=10, max_value=126, step=5)

# Filter Data for Selected Company
company_data = df[df['name'] == company]

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

# ========================== UI with Two Tabs: Company Analysis & Forecasted Results ==========================
tab1, tab2 = st.tabs(["📈 Company Analysis", "🔮 Forecasted Results"])

with tab1:
    st.subheader(f"📊 {company} - Stock Analysis")
    
    # Candlestick Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)
    fig.add_trace(go.Candlestick(x=company_data.index, open=company_data['open'], high=company_data['high'],
                                 low=company_data['low'], close=company_data['Close'], name="Candlesticks"), row=1, col=1)
    fig.add_trace(go.Bar(x=company_data.index, y=company_data['Volume'], name="Volume"), row=2, col=1)
    fig.update_layout(title=f"{company} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Moving Averages
    company_data['SMA_20'] = company_data['Close'].rolling(window=20).mean()
    company_data['SMA_50'] = company_data['Close'].rolling(window=50).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=company_data.index, y=company_data['Close'], mode='lines', name='Close Price'))
    fig_ma.add_trace(go.Scatter(x=company_data.index, y=company_data['SMA_20'], mode='lines', name='SMA 20'))
    fig_ma.add_trace(go.Scatter(x=company_data.index, y=company_data['SMA_50'], mode='lines', name='SMA 50'))
    fig_ma.update_layout(title=f"{company} Moving Averages")
    st.plotly_chart(fig_ma)

with tab2:
    st.subheader("🔮 Stock Price Forecast")
    st.write(f"📅 Forecasting **{forecast_days} days** ahead for **{company}**.")

    # Train LSTM Model
    st.write("🔄 Training LSTM model, please wait...")
    model, scaler = train_lstm_model(company_data, forecast_days)
    
    # ✅ Generate future dates based on last available date
    last_date = company_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # ✅ Get forecasted closing prices (corrected method)
    last_closing_prices = company_data['Close'].values[-forecast_days:].reshape(-1, 1)
    scaled_last_prices = scaler.transform(last_closing_prices)  # Scale the last known prices
    future_predictions = model.predict(scaled_last_prices)  # Predict future values
    future_predictions = scaler.inverse_transform(future_predictions)  # Convert back to real prices

    # ✅ Create forecast dataframe with Dates & Predicted Prices
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions.flatten()})
    
    # ✅ Show forecasted results
    st.write("### 📈 Forecasted Stock Prices")
    st.dataframe(forecast_df)  # ✅ Show table with forecasted prices
    st.line_chart(forecast_df.set_index("Date"))  # ✅ Show forecast plot
