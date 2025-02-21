import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots

# ========================== Theme Toggle (Dark/Light Mode) ==========================
theme_mode = st.sidebar.radio("🌙 Theme Mode", ["Light Mode", "Dark Mode"])
theme = "plotly_dark" if theme_mode == "Dark Mode" else "plotly_white"

# ========================== Load Parquet File ==========================
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_parquet(file_path)
    df.rename(columns={'date': 'Date', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date  
    df.set_index('Date', inplace=True)
    return df

df = load_stock_data('scaled_dataset_1x_snappy.parquet')

# ========================== Filter Companies with at Least 6 Months of Data ==========================
min_required_days = 126  
valid_companies = [company for company in df['name'].unique() if len(df[df['name'] == company]) >= min_required_days]
df = df[df['name'].isin(valid_companies)]  

# ========================== Sidebar Options ==========================
st.sidebar.header("📊 Stock Analysis Options")
company_list = valid_companies
company = st.sidebar.selectbox("Select Company", company_list)
forecast_days = st.sidebar.slider("Forecast Days", min_value=10, max_value=126, step=5)
company_data = df[df['name'] == company]

# ========================== Faster XGBoost Model for Forecasting ==========================
def train_xgboost_model(data, forecast_days):
    data = data['Close'].values
    X, y = [], []
    for i in range(len(data) - forecast_days):
        X.append(data[i:i+forecast_days])
        y.append(data[i+forecast_days])
    X, y = np.array(X), np.array(y)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    return model

# ========================== Generate Summary Based on Predicted vs Past Prices ==========================
def generate_price_summary(past_prices, predicted_prices):
    actual_change = past_prices[-1] - past_prices[0]
    predicted_change = predicted_prices[-1] - predicted_prices[0]

    if actual_change > 0:
        past_trend = "an upward trend"
    else:
        past_trend = "a downward trend"

    if predicted_change > 0:
        future_trend = "expected to continue rising"
    else:
        future_trend = "expected to decline further"

    return f"The stock price has shown {past_trend} over the past period. Based on the forecast, the price is {future_trend} in the upcoming days."

# ========================== UI with Tabs ==========================
tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis", "🔮 Forecasted Results", "💰 Portfolio Simulator"])

with tab1:
    st.subheader(f"📊 {company} - Stock Analysis")

    # ✅ Animated Line Chart with Confidence Intervals
    fig_trend = px.line(company_data, x=company_data.index, y="Close", title=f"{company} Stock Price Trend", template=theme)
    fig_trend.update_traces(line=dict(width=2), mode='lines+markers')
    fig_trend.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trend)

    # ✅ Interactive Candlestick Chart with Zoom & Pan
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2, 
                               row_heights=[0.7, 0.3], subplot_titles=("Candlestick Chart", "Volume"))

    fig_candle.add_trace(go.Candlestick(
        x=company_data.index, open=company_data['open'], high=company_data['high'],
        low=company_data['low'], close=company_data['Close'], name="Candlestick",
        hoverinfo="x+open+high+low+close"), row=1, col=1)

    fig_candle.add_trace(go.Bar(
        x=company_data.index, y=company_data['Volume'], name="Volume",
        marker_color='blue', opacity=0.6), row=2, col=1)

    fig_candle.update_layout(
        title=f"{company} Candlestick Chart",
        xaxis_rangeslider_visible=True,
        template=theme,
        hovermode="x unified"
    )
    st.plotly_chart(fig_candle)

    # ✅ Technical Indicators - RSI, MACD, Williams %R, Bollinger Bands
    company_data['RSI'] = 100 - (100 / (1 + (company_data['Close'].diff().where(company_data['Close'].diff() > 0, 0)
                                             .rolling(window=14).mean() /
                                             company_data['Close'].diff().where(company_data['Close'].diff() < 0, 0)
                                             .abs().rolling(window=14).mean())))

    company_data['Williams %R'] = ((company_data['high'].rolling(14).max() - company_data['Close']) /
                                   (company_data['high'].rolling(14).max() - company_data['low'].rolling(14).min())) * -100

    company_data['SMA_20'] = company_data['Close'].rolling(window=20).mean()
    company_data['Upper_Band'] = company_data['SMA_20'] + (company_data['Close'].rolling(window=20).std() * 2)
    company_data['Lower_Band'] = company_data['SMA_20'] - (company_data['Close'].rolling(window=20).std() * 2)

    fig_rsi = px.line(company_data, x=company_data.index, y="RSI", title="RSI (Relative Strength Index)", template=theme)
    fig_williams = px.line(company_data, x=company_data.index, y="Williams %R", title="Williams %R Indicator", template=theme)
    fig_bollinger = px.line(company_data, x=company_data.index, y=["Close", "Upper_Band", "Lower_Band"], title="Bollinger Bands", template=theme)

    st.plotly_chart(fig_rsi)
    st.plotly_chart(fig_williams)
    st.plotly_chart(fig_bollinger)

with tab2:
    st.subheader("🔮 Stock Price Forecast")
    st.write(f"📅 Forecasting **{forecast_days} days** ahead for **{company}**.")
    
    model = train_xgboost_model(company_data, forecast_days)
    future_predictions = []
    input_data = company_data['Close'].values[-forecast_days:].reshape(1, -1)

    for _ in range(forecast_days):
        pred = model.predict(input_data)[0]
        future_predictions.append(round(pred, 2))
        input_data = np.roll(input_data, -1)
        input_data[0, -1] = pred

    summary = generate_price_summary(company_data['Close'].values[-forecast_days:], future_predictions)

    st.write("📄 **Prediction Summary:**")
    st.write(summary)

