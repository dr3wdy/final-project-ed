#!/usr/bin/env python3
"""
Ultra-simple, modern GUI for your DWT + Gemini tools.

Usage:
- Users only type a ticker and optionally move a couple of sliders.
- No file paths, no dates, no wavelet jargon exposed by default.
- Now with:
    • Descriptions for each main control
    • DWT PNG overlay preview
    • Gemini rationale shown in the DWT tab
"""

from pathlib import Path
import traceback
import json
import re

import pandas as pd
import customtkinter as ctk

# Optional: Pillow for image display (DWT plots)
try:
    from PIL import Image as PilImage
except ImportError:
    PilImage = None

import google.generativeai as genai

import dwt_tool
import predict_from_OHLCV
import predict_from_dwt_csv


# -----------------------
# Helper: build / reuse DWT CSV + return PNG path
# -----------------------

def ensure_dwt_csv_for_ticker(
    ticker: str,
    save_dir: Path = Path("outputs"),
    wavelet: str = "db4",
    level: int | None = None,
    threshold_mode: str = "soft",
) -> tuple[Path, Path | None]:
    """
    Ensure there is a DWT CSV at:
        save_dir / f"{ticker}_original_and_denoised.csv"

    Returns:
        (csv_path, combo_plot_path_or_None)

    If CSV already exists, we just compute where the overlay PNG
    *would* live and return that Path (even if file is missing).
    """
    ticker = ticker.upper().strip()
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{ticker}_original_and_denoised.csv"

    # Overlay plot name used by dwt_tool.plot_series
    combo_plot_path = save_dir / f"{ticker}_original_vs_denoised.png"

    if csv_path.exists():
        # CSV already built; just hand back paths
        return csv_path, (combo_plot_path if combo_plot_path.exists() else None)

    # Fetch OHLCV
    df = dwt_tool.fetch_prices(ticker, start=None, end=None, interval="1d")
    close = df["Close"].astype(float)

    # DWT denoise
    close_denoised = dwt_tool.dwt_denoise(
        close,
        wavelet=wavelet,
        level=level,
        threshold_mode=threshold_mode,
    )

    # Plots + CSV
    # plot_series returns (orig_path, den_path, combo_path)
    _, _, combo_from_func = dwt_tool.plot_series(
        original=close,
        denoised=close_denoised,
        save_dir=save_dir,
        ticker=ticker,
    )

    combo_plot_path = Path(combo_from_func)

    out_df = pd.concat(
        [close.rename("Close"), close_denoised.rename("Close_DWT")],
        axis=1,
    )
    out_df.to_csv(csv_path)

    return csv_path, combo_plot_path if combo_plot_path.exists() else None


# -----------------------
# Helper: Gemini prediction with explanation (DWT)
# -----------------------

def dwt_predict_with_explanation(csv_path: str, temperature: float = 0.2):
    """
    Use the same logic as predict_from_dwt_csv.py, but also return Gemini's explanation.

    Returns:
        (predicted_close: float | None, explanation: str)
    """
    p = predict_from_dwt_csv

    try:
        dwt = p.load_dwt_csv(csv_path)
    except Exception as e:
        return None, f"Failed to load DWT CSV: {e}"

    if len(dwt) < 10:
        return None, "Need at least 10 rows in the DWT CSV."

    # Convert last ~60 points into JSON-friendly records
    series_for_llm = p.dwt_to_records_for_prompt(dwt, limit=60)
    last = float(dwt["dwt_close"].iloc[-1])

    # Same prompt as in predict_from_dwt_csv.py
    prompt = (
        "You are a numerical forecaster. Using ONLY the provided DWT-smoothed closing price time series, "
        "estimate the next trading day's closing price of the same series. Do not use external information. "
        "Return a single JSON object with this exact schema:\n\n"
        '{\n  "predicted_close": <number>,\n  "rationale": "<1-2 sentence summary of the trend you used>"\n}\n\n'
        "Constraints:\n"
        "- Use only the supplied DWT series.\n"
        "- Output ONLY the JSON (no code fences or extra text)."
    )

    model_name = p.pick_gemini_model_name()
    model = genai.GenerativeModel(model_name)

    response = p.call_gemini_safely(
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
        return None, "Empty response from Gemini."

    text = response.text
    pred = p.extract_number(text)
    if pred is None:
        # Couldn't parse a clean number; still return the raw text as explanation
        return None, text

    # Try to extract "rationale" from the JSON
    rationale = ""
    try:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(json)?", "", cleaned)
        cleaned = cleaned.strip("` \n")
        js = json.loads(cleaned)
        if isinstance(js, dict):
            rationale = js.get("rationale", "")
    except Exception:
        pass

    explanation = rationale if rationale else text
    return float(pred), explanation


# -----------------------
# Modern GUI
# -----------------------

class StockApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Global appearance
        ctk.set_appearance_mode("dark")       # "dark" | "light" | "system"
        ctk.set_default_color_theme("green")  # "blue" | "green" | "dark-blue"

        self.title("WaveLens – Stock Forecaster")
        self.geometry("1050x620")
        self.minsize(900, 540)

        # For holding DWT image reference
        self.dwt_image = None

        # Root layout: sidebar + main content
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_pages()

        self.show_page("quick")

    # ---------- BASIC LAYOUT ----------

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_rowconfigure((3, 4, 5), weight=1)

        title = ctk.CTkLabel(
            sidebar,
            text="WaveLens",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="w")

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Quick forecasts\nwith Gemini + DWT",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=("gray70", "gray70"),
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(4, 20), sticky="w")

        # Navigation buttons
        self.btn_quick = ctk.CTkButton(
            sidebar,
            text="📊 Quick forecast",
            anchor="w",
            command=lambda: self.show_page("quick"),
        )
        self.btn_quick.grid(row=2, column=0, padx=16, pady=4, sticky="ew")

        self.btn_dwt = ctk.CTkButton(
            sidebar,
            text="🌊 DWT-smoothed forecast",
            anchor="w",
            command=lambda: self.show_page("dwt"),
        )
        self.btn_dwt.grid(row=3, column=0, padx=16, pady=4, sticky="ew")

        # Appearance toggle at bottom
        self.appearance_var = ctk.StringVar(value="Dark")

        def on_appearance_change(choice: str):
            mode = choice.lower()
            ctk.set_appearance_mode(mode)

        mode_label = ctk.CTkLabel(
            sidebar,
            text="Theme",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray70"),
        )
        mode_label.grid(row=6, column=0, padx=20, pady=(0, 2), sticky="w")

        mode_switch = ctk.CTkSegmentedButton(
            sidebar,
            values=["Light", "Dark"],
            variable=self.appearance_var,
            command=on_appearance_change,
        )
        mode_switch.grid(row=7, column=0, padx=16, pady=(0, 20), sticky="ew")

    def _build_pages(self):
        self.page_container = ctk.CTkFrame(self)
        self.page_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        self.page_quick = ctk.CTkFrame(self.page_container)
        self.page_dwt = ctk.CTkFrame(self.page_container)

        for page in (self.page_quick, self.page_dwt):
            page.grid(row=0, column=0, sticky="nsew")

        self._build_page_quick()
        self._build_page_dwt()

    def show_page(self, which: str):
        if which == "quick":
            self.page_quick.tkraise()
            # Selected = solid color, Unselected = transparent
            self.btn_quick.configure(fg_color=("gray35", "gray20"))
            self.btn_dwt.configure(fg_color="transparent")
        else:
            self.page_dwt.tkraise()
            self.btn_dwt.configure(fg_color=("gray35", "gray20"))
            self.btn_quick.configure(fg_color="transparent")

    # ---------- PAGE: QUICK OHLCV ----------

    def _build_page_quick(self):
        page = self.page_quick
        page.grid_rowconfigure(3, weight=1)
        page.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            page,
            text="📊 Quick OHLCV Forecast",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        desc = ctk.CTkLabel(
            page,
            text="Type a ticker, choose how much history to use, and get tomorrow's predicted close.\n"
                 "Everything comes straight from yfinance + Gemini. No file paths, no config.",
            font=ctk.CTkFont(size=13),
            text_color=("gray80", "gray70"),
            justify="left",
        )
        desc.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="w")

        # Input card
        card = ctk.CTkFrame(page, corner_radius=18)
        card.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Row: ticker
        ctk.CTkLabel(card, text="1. Ticker", anchor="w").grid(
            row=0, column=0, padx=12, pady=(12, 0), sticky="w"
        )
        self.quick_ticker = ctk.CTkEntry(card, placeholder_text="e.g. NVDA, AAPL, SPY")
        self.quick_ticker.insert(0, "NVDA")
        self.quick_ticker.grid(row=1, column=0, padx=12, pady=(0, 2), sticky="ew")
        # Description under ticker
        quick_ticker_help = ctk.CTkLabel(
            card,
            text="Stock / ETF symbol to forecast (e.g., NVDA, SPY, QQQ).",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray60"),
        )
        quick_ticker_help.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        # Row: history segmented buttons
        ctk.CTkLabel(card, text="2. History window", anchor="w").grid(
            row=0, column=1, padx=12, pady=(12, 0), sticky="w"
        )
        self.quick_days_var = ctk.StringVar(value="120")
        self.quick_days_button = ctk.CTkSegmentedButton(
            card,
            values=["60", "120", "240"],
            variable=self.quick_days_var,
        )
        self.quick_days_button.grid(row=1, column=1, padx=12, pady=(0, 2), sticky="ew")
        quick_days_help = ctk.CTkLabel(
            card,
            text="How many past trading days to feed into Gemini.",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray60"),
        )
        quick_days_help.grid(row=2, column=1, padx=12, pady=(0, 10), sticky="w")

        # Row: temperature slider
        ctk.CTkLabel(card, text="3. Temperature", anchor="w").grid(
            row=0, column=2, padx=12, pady=(12, 0), sticky="w"
        )
        self.quick_temp_var = ctk.DoubleVar(value=0.2)
        self.quick_temp_slider = ctk.CTkSlider(
            card,
            from_=0.0,
            to=1.0,
            number_of_steps=10,
            variable=self.quick_temp_var,
        )
        self.quick_temp_slider.grid(row=1, column=2, padx=12, pady=(0, 2), sticky="ew")
        self.quick_temp_label = ctk.CTkLabel(
            card,
            text="Very stable (0.0–0.2)",
            font=ctk.CTkFont(size=11),
            text_color=("gray80", "gray70"),
        )
        self.quick_temp_label.grid(row=2, column=2, padx=12, pady=(0, 10), sticky="w")

        def _update_temp_label(value):
            v = float(value)
            if v <= 0.2:
                txt = "Very stable (0.0–0.2)"
            elif v <= 0.5:
                txt = "Balanced (0.3–0.5)"
            else:
                txt = "More exploratory (0.6–1.0)"
            self.quick_temp_label.configure(text=txt)

        self.quick_temp_slider.configure(command=_update_temp_label)

        # Row: compare with trendline
        self.quick_linear_var = ctk.BooleanVar(value=True)
        self.quick_linear_chk = ctk.CTkCheckBox(
            card,
            text="Compare against simple trendline",
            variable=self.quick_linear_var,
        )
        self.quick_linear_chk.grid(row=1, column=3, padx=12, pady=(0, 2), sticky="w")
        quick_linear_help = ctk.CTkLabel(
            card,
            text="Adds a simple linear regression baseline for comparison.",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray60"),
        )
        quick_linear_help.grid(row=2, column=3, padx=12, pady=(0, 10), sticky="w")

        # Run button
        run_btn = ctk.CTkButton(
            card,
            text="Run quick forecast",
            height=42,
            command=self.run_quick_forecast,
        )
        run_btn.grid(row=3, column=0, columnspan=4, padx=12, pady=(4, 14), sticky="ew")

        # Output card
        out_card = ctk.CTkFrame(page, corner_radius=18)
        out_card.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        out_card.grid_rowconfigure(0, weight=1)
        out_card.grid_columnconfigure(0, weight=1)

        self.quick_output = ctk.CTkTextbox(out_card, wrap="word")
        self.quick_output.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

    def safe_log(self, textbox: ctk.CTkTextbox, message: str, clear: bool = False):
        if clear:
            textbox.delete("1.0", "end")
        textbox.insert("end", message + "\n")
        textbox.see("end")
        textbox.update_idletasks()

    def run_quick_forecast(self):
        ticker = self.quick_ticker.get().strip().upper()
        if not ticker:
            self.safe_log(self.quick_output, "❌ Please enter a ticker.", clear=True)
            return

        try:
            days = int(self.quick_days_var.get())
        except ValueError:
            days = 120

        temp = float(self.quick_temp_var.get())

        self.safe_log(
            self.quick_output,
            f"▶ Running quick forecast for {ticker} (last {days} days, temp={temp:.2f})…",
            clear=True,
        )

        try:
            # Gemini forecast (numeric only for this tab)
            gem_pred = predict_from_OHLCV.predict_with_gemini(
                ticker=ticker,
                days=days,
                temperature=temp,
            )

            # Optional baseline
            lin_pred = None
            if self.quick_linear_var.get():
                self.safe_log(self.quick_output, "⏳ Computing simple trendline baseline…")
                lin_pred = predict_from_OHLCV.predict_with_linear(ticker)

            self.safe_log(self.quick_output, "\n──────────── RESULT CARD ────────────")
            self.safe_log(self.quick_output, f"• Ticker: {ticker}")
            self.safe_log(self.quick_output, f"• History used: {days} trading days")
            self.safe_log(self.quick_output, f"• Temperature: {temp:.2f}")

            if gem_pred is not None:
                self.safe_log(
                    self.quick_output,
                    f"\n🤖 Gemini next-day close: {gem_pred:.4f}",
                )
            else:
                self.safe_log(
                    self.quick_output,
                    "\n❌ Gemini prediction failed (check console for details).",
                )

            if lin_pred is not None:
                self.safe_log(
                    self.quick_output,
                    f"📈 Trendline baseline: {lin_pred:.4f}",
                )

        except Exception as e:
            self.safe_log(self.quick_output, "\n❌ Error during forecast:", clear=False)
            self.safe_log(self.quick_output, str(e))
            self.safe_log(self.quick_output, traceback.format_exc())

    # ---------- PAGE: DWT CSV ----------

    def _build_page_dwt(self):
        page = self.page_dwt
        page.grid_rowconfigure(3, weight=1)
        page.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            page,
            text="🌊 DWT-Smoothed Forecast",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        desc = ctk.CTkLabel(
            page,
            text="Behind the scenes, this tab builds a DWT-smoothed price series, then asks Gemini to forecast\n"
                 "the next DWT close. All DWT CSVs live in ./outputs and are handled automatically.",
            font=ctk.CTkFont(size=13),
            text_color=("gray80", "gray70"),
            justify="left",
        )
        desc.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="w")

        # Input card
        card = ctk.CTkFrame(page, corner_radius=18)
        card.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        card.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(card, text="1. Ticker", anchor="w").grid(
            row=0, column=0, padx=12, pady=(12, 0), sticky="w"
        )
        self.dwt_ticker = ctk.CTkEntry(card, placeholder_text="e.g. NVDA, TSLA, QQQ")
        self.dwt_ticker.insert(0, "NVDA")
        self.dwt_ticker.grid(row=1, column=0, padx=12, pady=(0, 2), sticky="ew")
        dwt_ticker_help = ctk.CTkLabel(
            card,
            text="Ticker to build a DWT-smoothed series and forecast on.",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray60"),
        )
        dwt_ticker_help.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        ctk.CTkLabel(card, text="2. Temperature", anchor="w").grid(
            row=0, column=1, padx=12, pady=(12, 0), sticky="w"
        )
        self.dwt_temp_var = ctk.DoubleVar(value=0.2)
        self.dwt_temp_slider = ctk.CTkSlider(
            card,
            from_=0.0,
            to=1.0,
            number_of_steps=10,
            variable=self.dwt_temp_var,
        )
        self.dwt_temp_slider.grid(row=1, column=1, padx=12, pady=(0, 2), sticky="ew")

        self.dwt_temp_label = ctk.CTkLabel(
            card,
            text="Balanced",
            font=ctk.CTkFont(size=11),
            text_color=("gray80", "gray70"),
        )
        self.dwt_temp_label.grid(row=2, column=1, padx=12, pady=(0, 10), sticky="w")

        def _update_dwt_temp_label(value):
            v = float(value)
            if v <= 0.2:
                txt = "Very stable (0.0–0.2)"
            elif v <= 0.5:
                txt = "Balanced (0.3–0.5)"
            else:
                txt = "More exploratory (0.6–1.0)"
            self.dwt_temp_label.configure(text=txt)

        self.dwt_temp_slider.configure(command=_update_dwt_temp_label)

        # Small advanced section toggle (for power users)
        self.dwt_advanced_open = False
        self.dwt_advanced_frame = ctk.CTkFrame(card, corner_radius=12)

        def toggle_advanced():
            if self.dwt_advanced_open:
                self.dwt_advanced_frame.grid_forget()
            else:
                self.dwt_advanced_frame.grid(
                    row=3, column=0, columnspan=3, padx=12, pady=(0, 12), sticky="ew"
                )
            self.dwt_advanced_open = not self.dwt_advanced_open

        adv_btn = ctk.CTkButton(
            card,
            text="Show advanced DWT options",
            fg_color=("gray25", "gray25"),
            hover_color=("gray35", "gray35"),
            height=32,
            command=toggle_advanced,
        )
        adv_btn.grid(row=2, column=2, padx=12, pady=(0, 10), sticky="e")

        # Advanced content
        self.dwt_advanced_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(self.dwt_advanced_frame, text="Wavelet", anchor="w").grid(
            row=0, column=0, padx=8, pady=(8, 0), sticky="w"
        )
        self.dwt_wavelet = ctk.CTkComboBox(
            self.dwt_advanced_frame,
            values=["db4", "sym5", "coif3"],
        )
        self.dwt_wavelet.set("db4")
        self.dwt_wavelet.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(self.dwt_advanced_frame, text="Level", anchor="w").grid(
            row=0, column=1, padx=8, pady=(8, 0), sticky="w"
        )
        self.dwt_level = ctk.CTkComboBox(
            self.dwt_advanced_frame,
            values=["auto", "2", "3", "4", "5"],
        )
        self.dwt_level.set("auto")
        self.dwt_level.grid(row=1, column=1, padx=8, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(self.dwt_advanced_frame, text="Threshold", anchor="w").grid(
            row=0, column=2, padx=8, pady=(8, 0), sticky="w"
        )
        self.dwt_thresh = ctk.CTkComboBox(
            self.dwt_advanced_frame,
            values=["soft", "hard"],
        )
        self.dwt_thresh.set("soft")
        self.dwt_thresh.grid(row=1, column=2, padx=8, pady=(0, 8), sticky="ew")

        # Run button
        run_btn = ctk.CTkButton(
            card,
            text="Run DWT-smoothed forecast",
            height=42,
            command=self.run_dwt_forecast,
        )
        run_btn.grid(row=4, column=0, columnspan=3, padx=12, pady=(4, 14), sticky="ew")

        # Output card
        out_card = ctk.CTkFrame(page, corner_radius=18)
        out_card.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        out_card.grid_rowconfigure(1, weight=1)
        out_card.grid_columnconfigure(0, weight=1)

        # DWT plot image label (graph shown here)
        self.dwt_image_label = ctk.CTkLabel(
            out_card,
            text="DWT overlay plot will appear here after running.",
            anchor="center",
            justify="center",
        )
        self.dwt_image_label.grid(row=0, column=0, padx=12, pady=(12, 0), sticky="n")

        # Text output (including Gemini explanation)
        self.dwt_output = ctk.CTkTextbox(out_card, wrap="word")
        self.dwt_output.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")

    def run_dwt_forecast(self):
        ticker = self.dwt_ticker.get().strip().upper()
        if not ticker:
            self.safe_log(self.dwt_output, "❌ Please enter a ticker.", clear=True)
            return

        temp = float(self.dwt_temp_var.get())

        # Advanced or defaults
        wavelet = self.dwt_wavelet.get() if self.dwt_advanced_open else "db4"
        level_str = self.dwt_level.get() if self.dwt_advanced_open else "auto"
        thresh = self.dwt_thresh.get() if self.dwt_advanced_open else "soft"

        if level_str.lower() == "auto":
            level = None
        else:
            try:
                level = int(level_str)
            except ValueError:
                level = None

        self.safe_log(
            self.dwt_output,
            f"▶ Building DWT series for {ticker} (wavelet={wavelet}, level={level_str}, thresh={thresh})…",
            clear=True,
        )

        try:
            csv_path, combo_path = ensure_dwt_csv_for_ticker(
                ticker=ticker,
                save_dir=Path("outputs"),
                wavelet=wavelet,
                level=level,
                threshold_mode=thresh,
            )
            self.safe_log(self.dwt_output, f"✅ DWT CSV ready: {csv_path}")

            # Try to show the overlay PNG
            if PilImage is None:
                self.safe_log(
                    self.dwt_output,
                    "ℹ️ Install 'pillow' (pip install pillow) to see the DWT plot image in the app.",
                )
            elif combo_path is not None and combo_path.exists():
                try:
                    img = PilImage.open(combo_path)
                    # Resize to fit nicely in the card
                    self.dwt_image = ctk.CTkImage(
                        light_image=img,
                        dark_image=img,
                        size=(800, 300),
                    )
                    self.dwt_image_label.configure(image=self.dwt_image, text="")
                except Exception as e:
                    self.safe_log(self.dwt_output, f"⚠️ Could not load plot image: {e}")
            else:
                self.dwt_image_label.configure(
                    image=None,
                    text="Overlay PNG not found yet. It should be saved next run.",
                )

            self.safe_log(self.dwt_output, "🤖 Asking Gemini to forecast next DWT close…")
            pred, explanation = dwt_predict_with_explanation(
                csv_path=str(csv_path),
                temperature=temp,
            )

            self.safe_log(self.dwt_output, "\n──────────── RESULT CARD ────────────")
            self.safe_log(self.dwt_output, f"• Ticker: {ticker}")
            self.safe_log(self.dwt_output, f"• CSV used: {csv_path}")
            self.safe_log(self.dwt_output, f"• Temperature: {temp:.2f}")

            if pred is not None:
                self.safe_log(
                    self.dwt_output,
                    f"\n🤖 Gemini next-day DWT close: {pred:.4f}",
                )
            else:
                self.safe_log(
                    self.dwt_output,
                    "\n❌ Gemini prediction failed to produce a clean number.",
                )

            if explanation:
                self.safe_log(
                    self.dwt_output,
                    "\n🧠 Gemini's explanation:\n" + explanation,
                )

        except Exception as e:
            self.safe_log(self.dwt_output, "\n❌ Error during DWT forecast:", clear=False)
            self.safe_log(self.dwt_output, str(e))
            self.safe_log(self.dwt_output, traceback.format_exc())


if __name__ == "__main__":
    app = StockApp()
    app.mainloop()
