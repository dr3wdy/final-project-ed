import os
import yfinance as yf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# === 1. Load API Key from .env ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

# === 2. Function to fetch data and predict ===
def predict_tomorrow_price(ticker):
    # Get last 60 days of data
    data = yf.download(ticker, period="60d", interval="1d")
    data = data.dropna(subset=["Close"])

    # Create numeric index for regression (like time steps)
    data["t"] = np.arange(len(data))
    X = data[["t"]]
    y = data["Close"]

    # Train a simple linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict next day
    next_t = np.array([[len(data)]])
    predicted_price = model.predict(next_t)[0]

    # Get last actual close for reference
    last_close = data["Close"].iloc[-1]
    diff = predicted_price - last_close
    pct_change = (diff / last_close) * 100

    print(f"\n📈 {ticker} Prediction for Tomorrow:")
    print(f"Last Close: ${last_close:.2f}")
    print(f"Predicted Next Close: ${predicted_price:.2f} ({pct_change:+.2f}%)")

    return predicted_price

# === 3. Example run ===
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, NVDA): ").upper().strip()
    predict_tomorrow_price(ticker)
