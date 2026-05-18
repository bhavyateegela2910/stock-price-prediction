import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

# Title
st.title("AI Stock Market Prediction System")

# User input
stock_symbol = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

# Download stock data
data = yf.download(
    stock_symbol,
    start="2023-01-01",
    end="2024-01-01"
)

# Show data
st.subheader("Stock Data")

st.dataframe(data.tail())

# Candlestick chart
fig = go.Figure(
    data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )
    ]
)

fig.update_layout(
    title=f"{stock_symbol} Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Load model
model = load_model(
    "models/lstm_model.keras"
)

# Load scaler
scaler = joblib.load(
    "models/scaler.pkl"
)

# Prediction
close_prices = data["Close"].values.reshape(-1, 1)

scaled_data = scaler.transform(
    close_prices
)

last_60_days = scaled_data[-60:]

x_test = []

x_test.append(
    last_60_days[:, 0]
)

x_test = np.array(x_test)

x_test = np.reshape(
    x_test,
    (x_test.shape[0], x_test.shape[1], 1)
)

prediction = model.predict(
    x_test,
    verbose=0
)

prediction = scaler.inverse_transform(
    prediction
)

st.subheader("Next Day Prediction")

st.success(
    f"Predicted Price: ${prediction[0][0]:.2f}"
)

# Live price
ticker = yf.Ticker(stock_symbol)

try:
    live_price = ticker.info["currentPrice"]

    st.subheader("Live Stock Price")

    st.info(f"${live_price}")

except:
    st.warning("Live price unavailable.")