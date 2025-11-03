import os
import yfinance as yf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# === 1. Load API Key from .env ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

# === 2. Function to fetch data and predict ===
def predict_tomorrow_price(ticker):
    # Explicitly fetch without auto_adjust (some versions ignore this flag)
    data = yf.download(ticker, period="60d", interval="1d", progress=False)

    if data.empty:
        print(f"❌ No data found for ticker '{ticker}'. Check the symbol or connection.")
        return None

    # Try to find a valid price column dynamically
    possible_cols = ["Close", "Adj Close", "close", "adjclose"]
    price_col = None
    for col in possible_cols:
        if col in data.columns:
            price_col = col
            break

    if not price_col:
        print(f"❌ No recognizable closing price column found. Columns available: {list(data.columns)}")
        return None

    # Clean and prep data
    data = data.dropna(subset=[price_col])
    data["t"] = np.arange(len(data))
    X = data[["t"]]
    y = data[price_col]

    # Train a simple linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict next day
    next_t = np.array([[len(data)]])
    predicted_price = model.predict(next_t)[0]

    last_close = data[price_col].iloc[-1]
    diff = predicted_price - last_close
    pct_change = (diff / last_close) * 100

    print(f"\n📈 {ticker} Prediction for Tomorrow:")
    print(f"Using column: {price_col}")
    print(f"Last Close: ${last_close:.2f}")
    print(f"Predicted Next Close: ${predicted_price:.2f} ({pct_change:+.2f}%)")

    return predicted_price

# === 3. Example run ===
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, NVDA): ").upper().strip()
    predict_tomorrow_price(ticker)
