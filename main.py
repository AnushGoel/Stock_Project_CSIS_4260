import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from plotly.subplots import make_subplots

# ========================== Theme Toggle (Dark/Light Mode) ==========================
theme_mode = st.sidebar.radio("🌙 Theme Mode", ["Light Mode", "Dark Mode"])
theme = "plotly_dark" if theme_mode == "Dark Mode" else "plotly_white"

# ========================== Load Parquet File ==========================
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_parquet(file_path)
    # Rename columns if necessary; adjust as needed for your dataset
    df.rename(columns={'date': 'Date', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    # Convert Date column to datetime (this keeps time info for Plotly)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_stock_data('scaled_dataset_1x_snappy.parquet')

# ========================== Filter Companies with at Least 6 Months of Data ==========================
min_required_days = 126  # Approximately 6 months of trading days
valid_companies = [
    company for company in df['name'].unique()
    if len(df[df['name'] == company]) >= min_required_days
]
df = df[df['name'].isin(valid_companies)]

# ========================== Sidebar Options ==========================
st.sidebar.header("📊 Stock Analysis Options")
company_list = valid_companies
company = st.sidebar.selectbox("Select Company", company_list)
forecast_days = st.sidebar.slider("Forecast Days", min_value=10, max_value=126, step=5)
company_data = df[df['name'] == company].sort_index()

# ========================== Ensure Numeric Data for OHLC & Volume ==========================
numeric_columns = ['open', 'high', 'low', 'Close', 'Volume']
for col in numeric_columns:
    if col in company_data.columns:
        company_data[col] = pd.to_numeric(company_data[col], errors='coerce')

# ========================== Prepare Data for OHLC Chart ==========================
# Drop any rows with missing OHLC values to prevent plotting errors.
ohlc_data = company_data.dropna(subset=['open', 'high', 'low', 'Close'])

# ========================== Forecasting Functions & Computation ==========================
def generate_price_summary(past_prices, predicted_prices):
    actual_change = past_prices[-1] - past_prices[0]
    predicted_change = predicted_prices[-1] - predicted_prices[0]
    past_trend = "an upward trend" if actual_change > 0 else "a downward trend"
    future_trend = "expected to continue rising" if predicted_change > 0 else "expected to decline further"
    return (
        f"The stock price has shown {past_trend} over the past period. "
        f"Based on the forecast, the price is {future_trend} in the upcoming days."
    )

def train_xgboost_model(data, forecast_days):
    data_values = data['Close'].values
    X, y = [], []
    for i in range(len(data_values) - forecast_days):
        X.append(data_values[i:i+forecast_days])
        y.append(data_values[i+forecast_days])
    X, y = np.array(X), np.array(y)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    return model

# Compute forecast predictions once so they can be used in multiple tabs.
model = train_xgboost_model(company_data, forecast_days)
future_predictions = []
input_data = company_data['Close'].values[-forecast_days:].reshape(1, -1)
for _ in range(forecast_days):
    pred = model.predict(input_data)[0]
    future_predictions.append(round(pred, 2))
    input_data = np.roll(input_data, -1)
    input_data[0, -1] = pred

# Build a forecast DataFrame with a date-only column.
forecast_index = pd.date_range(
    start=company_data.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)
forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Predicted_Close": future_predictions
})
forecast_df['Date'] = forecast_df['Date'].dt.date  # Keep only the date portion

forecast_summary = generate_price_summary(
    company_data['Close'].values[-forecast_days:],
    future_predictions
)

# ========================== UI with Tabs ==========================
tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis", "🔮 Forecasted Results", "💰 Portfolio Simulator"])

# ---------- Tab 1: Stock Analysis ----------
with tab1:
    st.subheader(f"📊 {company} - Stock Analysis")
    
    # Historical Line Chart
    fig_trend = px.line(
        company_data,
        x=company_data.index,
        y="Close",
        title=f"{company} Stock Price Trend",
        template=theme
    )
    fig_trend.update_traces(line=dict(width=2), mode='lines+markers')
    fig_trend.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # OHLC Chart with Volume (alternative to candlestick)
    fig_ohlc = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2,
        row_heights=[0.7, 0.3],
        subplot_titles=("OHLC Chart", "Volume")
    )
    
    fig_ohlc.add_trace(go.Ohlc(
        x=ohlc_data.index,
        open=ohlc_data['open'],
        high=ohlc_data['high'],
        low=ohlc_data['low'],
        close=ohlc_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name="OHLC"
    ), row=1, col=1)
    
    fig_ohlc.add_trace(go.Bar(
        x=ohlc_data.index,
        y=ohlc_data['Volume'],
        marker_color='blue',
        opacity=0.6,
        name="Volume"
    ), row=2, col=1)
    
    fig_ohlc.update_layout(
        title=f"{company} OHLC Chart",
        xaxis_rangeslider_visible=True,
        template=theme,
        hovermode="x unified"
    )
    st.plotly_chart(fig_ohlc, use_container_width=True)
    
    # Technical Indicators
    company_data['RSI'] = 100 - (100 / (1 + (
        company_data['Close'].diff().where(company_data['Close'].diff() > 0, 0)
        .rolling(window=14).mean() /
        company_data['Close'].diff().where(company_data['Close'].diff() < 0, 0)
        .abs().rolling(window=14).mean()
    )))
    company_data['Williams %R'] = (
        (company_data['high'].rolling(14).max() - company_data['Close']) /
        (company_data['high'].rolling(14).max() - company_data['low'].rolling(14).min())
    ) * -100
    company_data['SMA_20'] = company_data['Close'].rolling(window=20).mean()
    company_data['Upper_Band'] = company_data['SMA_20'] + (company_data['Close'].rolling(window=20).std() * 2)
    company_data['Lower_Band'] = company_data['SMA_20'] - (company_data['Close'].rolling(window=20).std() * 2)
    
    fig_rsi = px.line(
        company_data,
        x=company_data.index,
        y="RSI",
        title="RSI (Relative Strength Index)",
        template=theme
    )
    fig_williams = px.line(
        company_data,
        x=company_data.index,
        y="Williams %R",
        title="Williams %R Indicator",
        template=theme
    )
    fig_bollinger = px.line(
        company_data,
        x=company_data.index,
        y=["Close", "Upper_Band", "Lower_Band"],
        title="Bollinger Bands",
        template=theme
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.plotly_chart(fig_williams, use_container_width=True)
    st.plotly_chart(fig_bollinger, use_container_width=True)

# ---------- Tab 2: Forecasted Results ----------
with tab2:
    st.subheader("🔮 Stock Price Forecast")
    st.write(f"📅 Forecasting **{forecast_days} days** ahead for **{company}**.")
    
    # Display Forecast Table (date only)
    st.write("### Forecast Table")
    st.dataframe(forecast_df)
    
    st.write("### Prediction Summary")
    st.write(forecast_summary)
    
    # Forecast Graph: Historical Close + Forecasted Close
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=company_data.index,
        y=company_data['Close'],
        mode='lines',
        name='Historical Close'
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Close'],
        mode='lines+markers',
        name='Forecasted Close'
    ))
    fig_forecast.update_layout(
        title=f"{company} Forecast vs Historical Close",
        template=theme
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ---------- Tab 3: Portfolio Simulator ----------
with tab3:
    st.subheader("💰 Portfolio Simulator")
    st.write("Simulate your portfolio performance based on the forecasted prices.")
    
    initial_investment = st.number_input("Enter your initial investment amount", value=1000.0, min_value=0.0, step=100.0)
    current_price = company_data['Close'].iloc[-1]
    shares_bought = initial_investment / current_price
    st.write(f"At the current price of **{current_price:.2f}**, you can buy approximately **{shares_bought:.2f}** shares.")
    
    # Use the forecast DataFrame to simulate portfolio value over the forecast period.
    portfolio_df = forecast_df.copy()
    portfolio_df["Simulated_Portfolio_Value"] = portfolio_df["Predicted_Close"] * shares_bought
    
    st.write("### Forecasted Portfolio Performance")
    st.dataframe(portfolio_df[['Date', 'Simulated_Portfolio_Value']])
    
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_df["Date"],
        y=portfolio_df["Simulated_Portfolio_Value"],
        mode='lines+markers',
        name='Portfolio Value'
    ))
    fig_portfolio.update_layout(
        title="Simulated Portfolio Value Over Forecast Period",
        template=theme
    )
    st.plotly_chart(fig_portfolio, use_container_width=True)
