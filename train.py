import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Download stock data
stock_symbol = "AAPL"

data = yf.download(
    stock_symbol,
    start="2015-01-01",
    end="2024-01-01"
)

# Save dataset
data.to_csv("data/stock_data.csv")

print("Dataset Downloaded Successfully!")

# Close price
close_prices = data["Close"].values.reshape(-1, 1)

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(close_prices)

# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Create sequences
x_train = []
y_train = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    x_train.append(
        scaled_data[i-sequence_length:i, 0]
    )

    y_train.append(
        scaled_data[i, 0]
    )

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape
x_train = np.reshape(
    x_train,
    (x_train.shape[0], x_train.shape[1], 1)
)

print("Training Shape:", x_train.shape)

# Build LSTM model
model = Sequential()

model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=64
    )
)

model.add(Dropout(0.2))

model.add(Dense(units=1))

# Compile
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Train
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32
)

# Save model
model.save(
    "models/lstm_model.keras"
)

print("Model Saved Successfully!")