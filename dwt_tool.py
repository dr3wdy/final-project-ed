#!/usr/bin/env python3
"""
dwt_tool.py
-----------
Fetch a stock's price history for a given date range, smooth the Close series
with a Discrete Wavelet Transform (DWT), and save plots for:
  1) Original Close
  2) DWT-smoothed Close
  3) Original vs DWT-smoothed overlay
Also exports a CSV containing both series.

Examples:
    python dwt_tool.py --ticker NVDA --start 2023-01-01 --end 2025-11-01
    python dwt_tool.py --ticker AAPL
    python dwt_tool.py --ticker MSFT --wavelet sym5 --threshold hard
    python dwt_tool.py --ticker SPY --interval 1h --start 2024-01-01

Dependencies:
    pip install yfinance pywavelets pandas numpy matplotlib
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pywt
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # ensure headless savefig works everywhere
import matplotlib.pyplot as plt


# =========================
# Data fetching + normalize
# =========================

def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize any yfinance return shape into single-level OHLCV with 'Close'."""
    if df is None or df.empty:
        raise ValueError("No data returned.")

    # MultiIndex -> try to collapse to field columns
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        # Case A: fields on level 0
        if {"Open", "High", "Low", "Close", "Adj Close", "Volume"}.issubset(set(lvl0)):
            if ticker in set(lvl1):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                df = df.droplevel(1, axis=1)
        # Case B: ticker on level 0
        elif ticker in set(lvl0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            # Fallback: collapse to last level
            df.columns = df.columns.get_level_values(-1)

    else:
        # Single-level but labels may contain the ticker prefix
        rename_map = {}
        for c in df.columns:
            s = str(c).lower()
            if "open" in s and "interest" not in s:
                rename_map[c] = "Open"
            elif "high" in s:
                rename_map[c] = "High"
            elif "low" in s:
                rename_map[c] = "Low"
            elif "adj close" in s or "adjclose" in s:
                rename_map[c] = "Adj Close"
            elif s == "close" or (" close" in s and "adj" not in s):
                rename_map[c] = "Close"
            elif "volume" in s:
                rename_map[c] = "Volume"
        if rename_map:
            df = df.rename(columns=rename_map)

    # Ensure we have Close (use Adj Close if necessary)
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        else:
            raise ValueError(f"Could not normalize columns. Got: {list(df.columns)}")

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].sort_index()
    return df


def fetch_prices(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Robust fetch: try yf.download, else fall back to Ticker.history.
    Returns OHLCV with guaranteed 'Close'.
    """
    # 1) Try download()
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        group_by="column",
        auto_adjust=auto_adjust,
    )
    try:
        return _normalize_ohlcv(df, ticker)
    except Exception:
        # 2) Fallback to history()
        tk = yf.Ticker(ticker)
        df2 = tk.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
        if df2 is None or df2.empty:
            raise ValueError(f"No data returned for {ticker} with the given range.")
        return _normalize_ohlcv(df2, ticker)


# =========================
# DWT denoise
# =========================

def _universal_threshold(coeffs: np.ndarray) -> float:
    """Universal threshold (VisuShrink) using MAD estimate of noise."""
    if coeffs.size == 0:
        return 0.0
    sigma = np.median(np.abs(coeffs - np.median(coeffs))) / 0.6745
    if sigma == 0.0:
        return 0.0
    return sigma * math.sqrt(2.0 * math.log(coeffs.size))


def dwt_denoise(
    series: pd.Series,
    wavelet: str = "db4",
    level: Optional[int] = None,
    threshold_mode: str = "soft",
) -> pd.Series:
    """
    Wavelet denoise a 1D series (e.g., Close).
    - wavelet: 'db4', 'haar', 'sym5', etc.
    - level: None for max allowed
    - threshold_mode: 'soft' or 'hard'
    """
    if series.isna().all():
        raise ValueError("Input series contains only NaNs.")

    x = series.astype(float).interpolate(limit_direction="both").to_numpy()

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    cA, cDs = coeffs[0], coeffs[1:]

    new_coeffs = [cA]
    for cD in cDs:
        thr = _universal_threshold(cD)
        new_coeffs.append(cD if thr == 0.0 else pywt.threshold(cD, thr, mode=threshold_mode))

    x_rec = pywt.waverec(new_coeffs, wavelet=wavelet)[: len(x)]
    return pd.Series(x_rec, index=series.index, name=f"{series.name}_denoised")


# =========================
# Plotting
# =========================

def plot_series(
    original: pd.Series,
    denoised: pd.Series,
    save_dir: Path,
    ticker: str,
) -> Tuple[Path, Path, Path]:
    """Save original, denoised, and overlay plots; return their paths."""
    save_dir.mkdir(parents=True, exist_ok=True)

    orig_path = save_dir / f"{ticker}_original.png"
    den_path = save_dir / f"{ticker}_denoised.png"
    combo_path = save_dir / f"{ticker}_original_vs_denoised.png"

    # Original
    plt.figure()
    plt.plot(original.index, original.values)
    plt.title(f"{ticker} Close (Original)")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.tight_layout(); plt.savefig(orig_path); plt.close()

    # Denoised
    plt.figure()
    plt.plot(denoised.index, denoised.values)
    plt.title(f"{ticker} Close (DWT Denoised)")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.tight_layout(); plt.savefig(den_path); plt.close()

    # Overlay
    plt.figure()
    plt.plot(original.index, original.values, label="Original")
    plt.plot(denoised.index, denoised.values, label="DWT Denoised")
    plt.title(f"{ticker} Close: Original vs DWT Denoised")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
    plt.tight_layout(); plt.savefig(combo_path); plt.close()

    return orig_path, den_path, combo_path


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DWT denoise a stock's Close price and plot results.")
    p.add_argument("--ticker", required=True, help="Ticker symbol, e.g., NVDA")
    p.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    p.add_argument("--end", help="End date YYYY-MM-DD (exclusive)", default=None)
    p.add_argument("--interval", default="1d", help="yfinance interval (e.g., 1d, 1wk, 1mo, 1h)")
    p.add_argument("--wavelet", default="db4", help="Wavelet name (default: db4)")
    p.add_argument("--level", default="auto", help="Decomposition level (int or 'auto')")
    p.add_argument("--threshold", default="soft", choices=["soft", "hard"], help="Threshold mode")
    p.add_argument("--save_dir", default="outputs", help="Directory to save plots and CSV")
    return p.parse_args()


def main():
    args = parse_args()
    level = None if str(args.level).lower() == "auto" else int(args.level)

    print(f"[INFO] Fetching {args.ticker} data {args.start or '(start=max)'} -> {args.end or '(end=now)'} @ {args.interval}")
    df = fetch_prices(args.ticker, start=args.start, end=args.end, interval=args.interval)

    close = df["Close"].astype(float)
    print(f"[INFO] Points fetched: {len(close)}")

    print(f"[INFO] Denoising with wavelet={args.wavelet}, level={'auto' if level is None else level}, mode={args.threshold}")
    close_denoised = dwt_denoise(close, wavelet=args.wavelet, level=level, threshold_mode=args.threshold)

    save_dir = Path(args.save_dir)
    orig_p, den_p, combo_p = plot_series(close, close_denoised, save_dir, args.ticker.upper())

    # Export both series
    out_csv = save_dir / f"{args.ticker}_original_and_denoised.csv"
    pd.concat([close.rename("Close"), close_denoised.rename("Close_DWT")], axis=1).to_csv(out_csv)

    print("[OK] Saved:")
    print(f"  • Original plot : {orig_p}")
    print(f"  • Denoised plot : {den_p}")
    print(f"  • Overlay plot  : {combo_p}")
    print(f"  • Data CSV      : {out_csv}")


if __name__ == "__main__":
    main()
