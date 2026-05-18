from flask import Flask, render_template, request
import joblib
import numpy as np
import yfinance as yf

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model(
    "models/lstm_model.keras"
)

# Load scaler
scaler = joblib.load(
    "models/scaler.pkl"
)

@app.route("/", methods=["GET", "POST"])

def home():

    predicted_price = None

    stock_symbol = "AAPL"

    if request.method == "POST":

        stock_symbol = request.form["stock"]

        data = yf.download(
            stock_symbol,
            start="2023-01-01",
            end="2024-01-01"
        )

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

        predicted_price = round(
            float(prediction[0][0]),
            2
        )

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        stock_symbol=stock_symbol
    )

if __name__ == "__main__":
    app.run(debug=True)