import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
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

# ========================== Trading Recommendations ==========================
def get_stock_recommendation(prices):
    latest_change = prices[-1] - prices[-2]
    if latest_change > 0:
        return "📈 **BUY** - The stock is showing an upward trend."
    elif latest_change < 0:
        return "📉 **SELL** - The stock is declining, consider selling."
    else:
        return "⚖️ **HOLD** - No major movement in the stock."

# ========================== UI with Tabs ==========================
tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis", "🔮 Forecasted Results", "💰 Portfolio Simulator"])

with tab1:
    st.subheader(f"📊 {company} - Stock Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(company_data, x=company_data.index, y="Close", title=f"{company} Stock Closing Prices", template=theme)
        st.plotly_chart(fig)

    with col2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)
        fig.add_trace(go.Candlestick(x=company_data.index, open=company_data['open'], high=company_data['high'],
                                     low=company_data['low'], close=company_data['Close'], name="Candlesticks"), row=1, col=1)
        fig.add_trace(go.Bar(x=company_data.index, y=company_data['Volume'], name="Volume"), row=2, col=1)
        fig.update_layout(title=f"{company} Candlestick Chart", xaxis_rangeslider_visible=False, template=theme)
        st.plotly_chart(fig)

    with st.expander("📉 Stock Indicator Correlation Heatmap"):
        # ✅ Fix: Select only numeric columns & handle NaN values
        numeric_data = company_data.select_dtypes(include=[np.number]).dropna()
        correlation = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("🔮 Stock Price Forecast")
    st.write(f"📅 Forecasting **{forecast_days} days** ahead for **{company}**.")
    st.write("🔄 Training XGBoost model, please wait...")
    
    model = train_xgboost_model(company_data, forecast_days)
    future_predictions = []
    input_data = company_data['Close'].values[-forecast_days:].reshape(1, -1)

    for _ in range(forecast_days):
        pred = model.predict(input_data)[0]
        future_predictions.append(round(pred, 2))
        input_data = np.roll(input_data, -1)
        input_data[0, -1] = pred

    future_dates = [(company_data.index[-1] + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, len(future_predictions) + 1)]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions})
    st.dataframe(forecast_df)

    fig_forecast = px.line(forecast_df, x="Date", y="Predicted Close Price", title="Predicted Stock Price Over Time", template=theme)
    st.plotly_chart(fig_forecast)

    st.subheader("📊 Trading Recommendation")
    st.write(get_stock_recommendation(future_predictions))

with tab3:
    st.subheader("💰 Portfolio Growth Simulator")
    investment = st.number_input("Initial Investment Amount ($)", min_value=100, max_value=1000000, step=1000, value=10000)
    growth_rate = (future_predictions[-1] - future_predictions[0]) / future_predictions[0] * 100  
    final_value = investment * (1 + (growth_rate / 100))

    st.write(f"📈 **Expected Investment Value After {forecast_days} Days:** **${round(final_value, 2)}**")
    st.write(f"📊 **Stock Growth Rate:** {round(growth_rate, 2)}%")
