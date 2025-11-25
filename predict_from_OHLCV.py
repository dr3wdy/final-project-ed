import os
import re
import json
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

import google.generativeai as genai
from google.api_core.exceptions import PermissionDenied, FailedPrecondition, NotFound

# =========================
# 1) ENV / API SETUP
# =========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

PREFERRED_MODELS = [
    # Common, current IDs (AI Studio / google-generativeai)
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    # Sometimes the SDK returns fully-qualified names via list_models
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-flash-002",
]

def pick_gemini_model_name() -> str:
    """
    Returns a model name string that supports generateContent.
    Tries preferred names first; falls back to the first available.
    """
    try:
        available = {m.name for m in genai.list_models()}
    except Exception:
        available = set()

    # Prefer our list (and confirm presence if we could list)
    for name in PREFERRED_MODELS:
        if not available or name in available:
            return name

    # Fallback: first that supports generateContent
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
    except Exception:
        pass

    raise RuntimeError("No Gemini model available for generateContent. Check your key/project.")

# =========================
# 2) DATA HELPERS
# =========================
def pick_price_column(df: pd.DataFrame) -> str | None:
    # Try common names first
    for col in ["Close", "Adj Close", "close", "adjclose"]:
        if col in df.columns:
            return col
    # Fallback: first numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[0] if num_cols else None

def to_records_for_prompt(df: pd.DataFrame, price_col: str, limit: int = 45):
    use = df.tail(limit).copy()
    out = []
    for idx, row in use.iterrows():
        rec = {
            "date": str(getattr(idx, "date", lambda: idx)()),
            "close": float(row[price_col]),
        }
        if "Open" in df.columns:   rec["open"] = float(row["Open"])
        if "High" in df.columns:   rec["high"] = float(row["High"])
        if "Low"  in df.columns:   rec["low"]  = float(row["Low"])
        if "Volume" in df.columns and pd.notna(row["Volume"]):
            rec["volume"] = int(row["Volume"])
        out.append(rec)
    return out

def extract_number(text: str) -> float | None:
    # Try JSON first (with code-fence cleanup)
    try:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        data = json.loads(cleaned)
        if isinstance(data, dict) and "predicted_close" in data:
            return float(data["predicted_close"])
        if isinstance(data, (int, float)):
            return float(data)
    except Exception:
        pass
    # Fallback: first float
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group(0)) if m else None

# =========================
# 3) GEMINI PREDICTOR
# =========================
def call_gemini_safely(model, parts, **kwargs):
    try:
        return model.generate_content(parts, **kwargs)
    except PermissionDenied as e:
        print("❌ Permission denied calling Gemini.")
        print("• Enable the Generative Language API or use an AI Studio key.")
        print("• Ensure the key is from the same project where the API is enabled and billing is active.")
        print(f"Details: {e.message}")
        return None
    except FailedPrecondition as e:
        print("❌ Precondition failed (often service not active or billing not enabled).")
        print(f"Details: {e.message}")
        return None
    except NotFound as e:
        print("❌ Model not found for this API/version. Try a different model ID.")
        print(f"Details: {e.message}")
        return None
    except Exception as e:
        print("❌ Unexpected error calling Gemini:", e)
        return None

def predict_with_gemini(ticker: str, days: int = 120, temperature: float = 0.2) -> float | None:
    ticker = (ticker or "").upper().strip()
    if not ticker:
        print("❌ No ticker entered.")
        return None

    # Pull recent data (history() is more stable than download() for single tickers)
    df = yf.Ticker(ticker).history(period=f"{days}d", interval="1d",
                                   auto_adjust=True, actions=False)
    if df is None or df.empty:
        print(f"❌ No price data for '{ticker}'.")
        return None

    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if x]).strip() for tup in df.columns]

    price_col = pick_price_column(df)
    if not price_col:
        print(f"❌ No usable price column. Columns: {list(df.columns)}")
        return None

    df = df.dropna(subset=[price_col])
    if len(df) < 10:
        print("❌ Not enough data points.")
        return None

    # Features to help the model reason (still only time-series based)
    px = df[price_col].astype(float)
    feats = {
        "last_close": float(px.iloc[-1]),
        "ret_1d": float(px.pct_change(1).iloc[-1]),
        "ret_5d": float(px.pct_change(5).iloc[-1]) if len(px) >= 6 else None,
        "ma_5": float(px.rolling(5).mean().iloc[-1]) if len(px) >= 5 else None,
        "ma_10": float(px.rolling(10).mean().iloc[-1]) if len(px) >= 10 else None,
        "vol_5": float(px.pct_change().rolling(5).std().iloc[-1]) if len(px) >= 6 else None,
        "vol_10": float(px.pct_change().rolling(10).std().iloc[-1]) if len(px) >= 11 else None,
    }

    series_for_llm = to_records_for_prompt(df, price_col, limit=45)
    last_close = feats["last_close"]

    # Build prompt
    prompt = (
        "You are a numerical forecaster. Using ONLY the historical OHLCV time series provided, "
        "estimate the next trading day's closing price. Do not use external news or assumptions. "
        "Return a single JSON object with this exact schema:\n\n"
        '{\n  "predicted_close": <number>,\n  "rationale": "<1-2 sentence summary of the trend you used>"\n}\n\n'
        "Constraints:\n"
        "- Base the estimate strictly on the supplied series.\n"
        "- Use the same currency/scale as the provided close values.\n"
        "- Output ONLY the JSON object (no code fences, no extra text)."
    )

    # Pick a supported model name and create the model once
    model_name = pick_gemini_model_name()
    print(f"Using Gemini model: {model_name}")
    model = genai.GenerativeModel(model_name)

    response = call_gemini_safely(
        model,
        [
            {"role": "user", "parts": [
                {"text": prompt},
                {"text": f"Ticker: {ticker}"},
                {"text": f"Derived features: {json.dumps(feats)}"},
                {"text": f"Recent data (most recent last): {json.dumps(series_for_llm)}"}
            ]}
        ],
        generation_config={"temperature": temperature},
    )

    if not response or not getattr(response, "text", None):
        print("❌ Empty response from Gemini.")
        return None

    pred = extract_number(response.text)
    if pred is None:
        print("❌ Could not parse a numeric prediction from Gemini response:")
        print(response.text)
        return None

    diff = pred - last_close
    pct = (diff / last_close) * 100 if last_close else 0.0

    # Try to print rationale if present
    rationale = ""
    try:
        js = json.loads(re.sub(r"^```(json)?", "", response.text.strip()).strip("` \n"))
        rationale = js.get("rationale", "")
    except Exception:
        pass

    print(f"\n🤖 {ticker} — Gemini Prediction")
    print(f"Last Close: ${last_close:.2f}")
    print(f"Predicted Next Close: ${pred:.2f} ({pct:+.2f}%)")
    if rationale:
        print(f"Rationale: {rationale}")

    return float(pred)

# =========================
# 4) OPTIONAL: Linear baseline
# =========================
def predict_with_linear(ticker: str) -> float | None:
    df = yf.Ticker(ticker).history(period="60d", interval="1d",
                                   auto_adjust=True, actions=False)
    if df.empty:
        print(f"❌ No data for '{ticker}'.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if x]).strip() for tup in df.columns]

    price_col = pick_price_column(df)
    if not price_col:
        print(f"❌ No price column in: {list(df.columns)}")
        return None

    df = df.dropna(subset=[price_col])
    if len(df) < 2:
        print("❌ Not enough data points for linear regression.")
        return None

    df["t"] = np.arange(len(df), dtype=float)
    X, y = df[["t"]], df[price_col].astype(float)
    model = LinearRegression().fit(X, y)
    pred = float(model.predict([[len(df)]])[0])
    last_close = float(y.iloc[-1])

    print(f"\n📈 {ticker} — Linear Regression Baseline")
    print(f"Last Close: ${last_close:.2f}")
    print(f"Predicted Next Close: ${pred:.2f} ({(pred-last_close)/last_close:+.2%})")
    return pred

# =========================
# 5) MAIN
# =========================
if __name__ == "__main__":
    try:
        ticker_in = input("Enter stock ticker (e.g., AAPL, NVDA): ").upper().strip()
    except EOFError:
        ticker_in = ""

    # Primary: Gemini (uses your GOOGLE_API_KEY)
    predict_with_gemini(ticker_in)

    # Optional: compare with simple linear baseline
    # predict_with_linear(ticker_in)
