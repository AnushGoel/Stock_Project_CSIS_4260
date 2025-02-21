import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

# ========================== Load Parquet File ==========================
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_parquet(file_path)
    df.rename(columns={'date': 'Date', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # ✅ Convert Date to date only (No time)
    df.set_index('Date', inplace=True)
    return df

df = load_stock_data('scaled_dataset_1x_snappy.parquet')

# ========================== Filter Companies with at Least 6 Months of Data ==========================
min_required_days = 126  # 6 months (21 trading days per month)
valid_companies = [company for company in df['name'].unique() if len(df[df['name'] == company]) >= min_required_days]
df = df[df['name'].isin(valid_companies)]  # Keep only valid companies

# ========================== Sidebar Options ==========================
st.sidebar.header("📊 Stock Analysis Options")
company_list = valid_companies
company = st.sidebar.selectbox("Select Company", company_list)

# Forecasting Range (10 days - 6 months)
forecast_days = st.sidebar.slider("Forecast Days", min_value=10, max_value=126, step=5)

# Filter Data for Selected Company
company_data = df[df['name'] == company]

# ========================== Technical Indicators ==========================
def add_technical_indicators(data):
    # RSI Calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)

    return data

company_data = add_technical_indicators(company_data)

# ========================== Faster XGBoost Model for Forecasting ==========================
def train_xgboost_model(data, forecast_days):
    data = data['Close'].values
    X, y = [], []
    for i in range(len(data) - forecast_days):
        X.append(data[i:i+forecast_days])
        y.append(data[i+forecast_days])

    X, y = np.array(X), np.array(y)

    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    return model

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

    # RSI, MACD, and Bollinger Bands
    st.subheader("📉 Technical Indicators")

    # RSI
    fig_rsi = px.line(company_data, x=company_data.index, y="RSI", title="RSI (Relative Strength Index)")
    st.plotly_chart(fig_rsi)

    # MACD
    fig_macd = px.line(company_data, x=company_data.index, y=["MACD", "Signal"], title="MACD Indicator")
    st.plotly_chart(fig_macd)

    # Bollinger Bands
    fig_bbands = px.line(company_data, x=company_data.index, y=["Close", "Upper_Band", "Lower_Band"], title="Bollinger Bands")
    st.plotly_chart(fig_bbands)

    # Seasonal Decomposition Plot (Fixed Size)
    st.subheader("📊 Seasonal Trend Analysis")
    decomposition = seasonal_decompose(company_data['Close'], model='multiplicative', period=30)
    fig, axs = plt.subplots(3, figsize=(12, 6))
    axs[0].plot(decomposition.trend, label="Trend")
    axs[0].set_title("Stock Price Trend")
    axs[1].plot(decomposition.seasonal, label="Seasonality")
    axs[1].set_title("Stock Seasonality")
    axs[2].plot(decomposition.resid, label="Residual")
    axs[2].set_title("Stock Residual")
    plt.tight_layout()  # ✅ Fix overlaying issue
    st.pyplot(fig)

with tab2:
    st.subheader("🔮 Stock Price Forecast")
    st.write(f"📅 Forecasting **{forecast_days} days** ahead for **{company}**.")

    # Train XGBoost Model
    st.write("🔄 Training XGBoost model, please wait...")
    model = train_xgboost_model(company_data, forecast_days)

    # ✅ Generate future predictions properly
    future_predictions = []
    input_data = company_data['Close'].values[-forecast_days:].reshape(1, -1)

    for _ in range(forecast_days):
        pred = model.predict(input_data)[0]  # Predict next value
        future_predictions.append(pred)

        # Shift input window: remove first value, add new prediction
        input_data = np.roll(input_data, -1)
        input_data[0, -1] = pred

    # ✅ Ensure `future_dates` has only DATE (no time)
    future_dates = [(company_data.index[-1] + timedelta(days=i)) for i in range(1, len(future_predictions) + 1)]
    future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]  # ✅ Convert to string format

    # ✅ Create forecast dataframe
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions})

    # ✅ Show forecasted results
    st.write("### 📈 Forecasted Stock Prices")
    st.dataframe(forecast_df)
    st.line_chart(forecast_df.set_index("Date"))
