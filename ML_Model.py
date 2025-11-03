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
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        print(f"❌ No data found for ticker '{ticker}'. Check the symbol and try again.")
        return None

    # Use Adj Close instead
    data = data.dropna(subset=["Adj Close"])
    data["t"] = np.arange(len(data))
    X = data[["t"]]
    y = data["Adj Close"]

    model = LinearRegression()
    model.fit(X, y)

    next_t = np.array([[len(data)]])
    predicted_price = model.predict(next_t)[0]

    last_close = data["Adj Close"].iloc[-1]
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
