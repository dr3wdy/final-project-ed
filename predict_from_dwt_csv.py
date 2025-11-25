import os
import re
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

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
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-flash-002",
]

def pick_gemini_model_name() -> str:
    """Return a model name that supports generateContent, preferring stable IDs."""
    try:
        available = {m.name for m in genai.list_models()}
    except Exception:
        available = set()

    for name in PREFERRED_MODELS:
        if not available or name in available:
            return name

    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
    except Exception:
        pass

    raise RuntimeError("No Gemini model available. Check your API key/project/billing.")

# =========================
# 2) CSV HELPERS (DWT)
# =========================
def load_dwt_csv(path: str) -> pd.DataFrame:
    """
    Loads CSV emitted by your dwt_tool.py:
      columns: Date index (or 'Date' column), 'Close', 'Close_DWT'
    Returns a DataFrame indexed by datetime with one column 'dwt_close' (float).
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    # Find date column or use index
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "time", "datetime"):
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    # Prefer your tool's smoothed column name
    candidates = ["Close_DWT", "close_dwt", "dwt_close", "dwt", "smooth", "smoothed", "close_smooth", "smooth_close"]
    price_col = None
    for c in candidates:
        match = [col for col in df.columns if col.lower() == c.lower()]
        if match:
            price_col = match[0]
            break

    if price_col is None:
        # If your tool changed names, help by printing columns
        raise ValueError(f"Could not find a DWT column in CSV. Columns: {list(df.columns)}. "
                         "Expected 'Close_DWT' from your dwt_tool.py output.")

    out = df[[price_col]].copy()
    out.columns = ["dwt_close"]
    out["dwt_close"] = pd.to_numeric(out["dwt_close"], errors="coerce")
    out = out.dropna()
    if out.empty:
        raise ValueError("DWT series is empty after cleaning.")
    return out

def dwt_to_records_for_prompt(dwt_df: pd.DataFrame, limit: int = 60):
    use = dwt_df.tail(limit).copy()
    out = []
    for idx, row in use.iterrows():
        date_val = getattr(idx, "date", lambda: idx)()
        out.append({
            "date": str(date_val),
            "dwt_close": float(row["dwt_close"]),
        })
    return out

# =========================
# 3) GENERIC HELPERS
# =========================
def extract_number(text: str) -> float | None:
    # Prefer JSON parse (strip fences if present)
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
    # Fallback: first float-ish token
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group(0)) if m else None

def call_gemini_safely(model, parts, **kwargs):
    try:
        return model.generate_content(parts, **kwargs)
    except PermissionDenied as e:
        print("❌ Permission denied calling Gemini.")
        print("• Enable the Generative Language API or use an AI Studio key.")
        print("• Ensure billing is active for the project that key belongs to.")
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

# =========================
# 4) GEMINI PREDICTOR (DWT CSV)
# =========================
def predict_from_dwt_csv_gemini(csv_path: str, temperature: float = 0.2) -> float | None:
    try:
        dwt = load_dwt_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to load DWT CSV: {e}")
        return None

    if len(dwt) < 10:
        print("❌ Need at least 10 rows in DWT CSV.")
        return None

    series_for_llm = dwt_to_records_for_prompt(dwt, limit=60)
    last = float(dwt["dwt_close"].iloc[-1])

    prompt = (
        "You are a numerical forecaster. Using ONLY the provided DWT-smoothed closing price time series, "
        "estimate the next trading day's closing price of the same series. Do not use external information. "
        "Return a single JSON object with this exact schema:\n\n"
        '{\n  "predicted_close": <number>,\n  "rationale": "<1-2 sentence summary of the trend you used>"\n}\n\n'
        "Constraints:\n"
        "- Use only the supplied DWT series.\n"
        "- Output ONLY the JSON (no code fences or extra text)."
    )

    model_name = pick_gemini_model_name()
    print(f"Using Gemini model: {model_name}")
    model = genai.GenerativeModel(model_name)

    response = call_gemini_safely(
        model,
        [
            {"role": "user", "parts": [
                {"text": prompt},
                {"text": f"DWT data (most recent last): {json.dumps(series_for_llm)}"}
            ]}
        ],
        generation_config={"temperature": temperature},
    )

    if not response or not getattr(response, "text", None):
        print("❌ Empty response from Gemini.")
        return None

    pred = extract_number(response.text)
    if pred is None:
        print("❌ Could not parse numeric prediction from Gemini response:")
        print(response.text)
        return None

    diff = pred - last
    pct = (diff / last) * 100 if last else 0.0

    print("\n🤖 DWT Gemini Prediction")
    print(f"Last DWT Close: {last:.4f}")
    print(f"Predicted Next Close: {pred:.4f} ({pct:+.2f}%)")

    # Optional: rationale
    try:
        js = json.loads(re.sub(r"^```(json)?", "", response.text.strip()).strip("` \n"))
        if isinstance(js, dict) and "rationale" in js:
            print(f"Rationale: {js['rationale']}")
    except Exception:
        pass

    return float(pred)

# =========================
# 5) MAIN
# =========================
if __name__ == "__main__":
    try:
        csv_path = input("Path to DWT CSV (e.g., C:\\path\\to\\NVDA_original_and_denoised.csv): ").strip().strip('"')
    except EOFError:
        csv_path = ""
    predict_from_dwt_csv_gemini(csv_path)
