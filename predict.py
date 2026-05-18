import math
import joblib
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# Load model
model = load_model(
    "models/lstm_model.keras"
)

# Load scaler
scaler = joblib.load(
    "models/scaler.pkl"
)

# Download stock data
stock_symbol = "AAPL"

data = yf.download(
    stock_symbol,
    start="2023-01-01",
    end="2024-01-01"
)

# Close prices
dataset = data["Close"].values.reshape(-1, 1)

actual_prices = dataset

# Scale
scaled_data = scaler.transform(dataset)

# Create test data
x_test = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):

    x_test.append(
        scaled_data[i-sequence_length:i, 0]
    )

x_test = np.array(x_test)

# Reshape
x_test = np.reshape(
    x_test,
    (x_test.shape[0], x_test.shape[1], 1)
)

# Predict
predictions = model.predict(x_test)

# Reverse scale
predictions = scaler.inverse_transform(
    predictions
)

# Actual
actual = actual_prices[sequence_length:]

# RMSE
rmse = math.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

accuracy = 100 - rmse

print("Accuracy:", round(accuracy, 2), "%")

# Future prediction
future_days = 7

future_predictions = []

current_batch = scaled_data[-60:]

for i in range(future_days):

    current_batch_reshaped = current_batch.reshape(
        (1, 60, 1)
    )

    predicted = model.predict(
        current_batch_reshaped,
        verbose=0
    )

    future_predictions.append(
        predicted[0][0]
    )

    current_batch = np.append(
        current_batch[1:],
        predicted[0]
    )

future_predictions = np.array(
    future_predictions
)

future_predictions = scaler.inverse_transform(
    future_predictions.reshape(-1, 1)
)

print("\nNext 7 Days Prediction:\n")

for i, price in enumerate(future_predictions):
    print(f"Day {i+1}: ${price[0]:.2f}")

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    actual,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="Predicted Price"
)

plt.title("Stock Price Prediction")

plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()

plt.show()