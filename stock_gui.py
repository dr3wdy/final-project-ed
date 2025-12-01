#!/usr/bin/env python3
from pathlib import Path
import traceback
from io import StringIO
import contextlib

import customtkinter as ctk

# Optional: Pillow for image display
try:
    from PIL import Image as PilImage
except ImportError:
    PilImage = None

import dwt_tool
import predict_from_dwt_csv as dwt_pred


def ensure_dwt_csv_for_ticker(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    save_dir: Path = Path("outputs"),
    wavelet: str = "db4",
    level=None,
    threshold_mode: str = "soft",
):
    """
    Build or reuse a DWT CSV for the given ticker *and date range*.

    Date range:
        - start: 'YYYY-MM-DD' or None for earliest available
        - end  : 'YYYY-MM-DD' or None for latest available

    Filenames encode the range so different ranges don't collide:
        outputs/{TICKER}_{START}_{END}_original_and_denoised.csv
        outputs/{TICKER}_{START}_{END}_original_vs_denoised.png
    """
    ticker = ticker.upper().strip()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build a safe tag that encodes the date range in the file name
    safe_start = (start or "MIN").replace("-", "")
    safe_end = (end or "MAX").replace("-", "")
    tag = f"{ticker}_{safe_start}_{safe_end}"

    csv_path = save_dir / f"{tag}_original_and_denoised.csv"
    combo_plot_path = save_dir / f"{tag}_original_vs_denoised.png"

    # If we've already built this exact range, reuse it
    if csv_path.exists():
        return csv_path, (combo_plot_path if combo_plot_path.exists() else None)

    # Fetch OHLCV for the chosen range
    df = dwt_tool.fetch_prices(
        ticker,
        start=start or None,
        end=end or None,
        interval="1d",
    )
    close = df["Close"].astype(float)

    # DWT denoise
    close_denoised = dwt_tool.dwt_denoise(
        close,
        wavelet=wavelet,
        level=level,
        threshold_mode=threshold_mode,
    )

    # Save plots (original, denoised, overlay) with the range tag
    _, _, combo_from_func = dwt_tool.plot_series(
        original=close,
        denoised=close_denoised,
        save_dir=save_dir,
        ticker=tag,  # use tag so filenames include dates
    )
    combo_plot_path = Path(combo_from_func)

    # Save CSV with both series
    out_df = close.to_frame("Close").join(
        close_denoised.rename("Close_DWT")
    )
    out_df.to_csv(csv_path)

    return csv_path, (combo_plot_path if combo_plot_path.exists() else None)




class DWTApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.title("WaveLens – DWT Forecast")
        self.geometry("1050x600")
        self.minsize(900, 500)

        self.dwt_image = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    # ---------- LEFT PANEL: Controls ----------

    def _build_left_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=0)
        panel.grid(row=0, column=0, sticky="nsw")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(18, weight=1)

        title = ctk.CTkLabel(
            panel,
            text="WaveLens",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, padx=16, pady=(16, 0), sticky="w")

        subtitle = ctk.CTkLabel(
            panel,
            text="DWT-smoothed\nGemini forecast",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=("gray70", "gray70"),
        )
        subtitle.grid(row=1, column=0, padx=16, pady=(2, 12), sticky="w")

        # --- Ticker ---
        ctk.CTkLabel(panel, text="Ticker", anchor="w").grid(
            row=2, column=0, padx=16, pady=(8, 0), sticky="w"
        )
        self.ticker_entry = ctk.CTkEntry(panel, placeholder_text="e.g. NVDA, TSLA, QQQ")
        self.ticker_entry.insert(0, "NVDA")
        self.ticker_entry.grid(row=3, column=0, padx=16, pady=(0, 2), sticky="ew")
        ctk.CTkLabel(
            panel,
            text="Symbol to build the DWT-smoothed price series and forecast.",
            font=ctk.CTkFont(size=11),
            text_color=("gray70", "gray60"),
        ).grid(row=4, column=0, padx=16, pady=(0, 8), sticky="w")

        # --- Start date ---
        ctk.CTkLabel(panel, text="Start date (optional)", anchor="w").grid(
            row=5, column=0, padx=16, pady=(4, 0), sticky="w"
        )
        self.start_entry = ctk.CTkEntry(panel, placeholder_text="YYYY-MM-DD")
        self.start_entry.grid(row=6, column=0, padx=16, pady=(0, 2), sticky="ew")
        ctk.CTkLabel(
            panel,
            text="First date to fetch (e.g., 2023-01-01).",
            font=ctk.CTkFont(size=10),
            text_color=("gray70", "gray60"),
        ).grid(row=7, column=0, padx=16, pady=(0, 4), sticky="w")

        # --- End date ---
        ctk.CTkLabel(panel, text="End date (optional)", anchor="w").grid(
            row=8, column=0, padx=16, pady=(4, 0), sticky="w"
        )
        self.end_entry = ctk.CTkEntry(panel, placeholder_text="YYYY-MM-DD")
        self.end_entry.grid(row=9, column=0, padx=16, pady=(0, 2), sticky="ew")
        ctk.CTkLabel(
            panel,
            text="Last date (exclusive). Make sure it is at least a month from start",
            font=ctk.CTkFont(size=10),
            text_color=("gray70", "gray60"),
        ).grid(row=10, column=0, padx=16, pady=(0, 8), sticky="w")

        # --- Temperature ---
        ctk.CTkLabel(panel, text="Temperature", anchor="w").grid(
            row=11, column=0, padx=16, pady=(8, 0), sticky="w"
        )
        self.temp_var = ctk.DoubleVar(value=0.2)
        self.temp_slider = ctk.CTkSlider(
            panel,
            from_=0.0,
            to=1.0,
            number_of_steps=10,
            variable=self.temp_var,
        )
        self.temp_slider.grid(row=12, column=0, padx=16, pady=(0, 2), sticky="ew")

        self.temp_label = ctk.CTkLabel(
            panel,
            text="Model randomness: Very stable (0.0–0.2)\nLower = more stable, higher = more exploratory.",
            font=ctk.CTkFont(size=11),
            text_color=("gray80", "gray70"),
        )
        self.temp_label.grid(row=13, column=0, padx=16, pady=(0, 8), sticky="w")

        def _update_temp(value):
            v = float(value)
            if v <= 0.2:
                band = "Very stable (0.0–0.2)"
            elif v <= 0.5:
                band = "Balanced (0.3–0.5)"
            else:
                band = "More exploratory (0.6–1.0)"
            self.temp_label.configure(
                text=f"Model randomness: {band}\nLower = more stable, higher = more exploratory."
            )

        self.temp_slider.configure(command=_update_temp)

        # --- Advanced options ---
        self.adv_open = False
        self.adv_frame = ctk.CTkFrame(panel, corner_radius=12)

        def toggle_advanced():
            if self.adv_open:
                self.adv_frame.grid_forget()
            else:
                self.adv_frame.grid(row=15, column=0, padx=16, pady=(4, 8), sticky="ew")
            self.adv_open = not self.adv_open

        adv_btn = ctk.CTkButton(
            panel,
            text="Show advanced DWT options",
            fg_color=("gray25", "gray25"),
            hover_color=("gray35", "gray35"),
            height=32,
            command=toggle_advanced,
        )
        adv_btn.grid(row=14, column=0, padx=16, pady=(4, 4), sticky="ew")

        # Advanced controls
        self.adv_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(self.adv_frame, text="Wavelet", anchor="w").grid(
            row=0, column=0, padx=8, pady=(6, 0), sticky="w"
        )
        self.wavelet_cb = ctk.CTkComboBox(
            self.adv_frame,
            values=["db4", "sym5", "coif3"],
        )
        self.wavelet_cb.set("db4")
        self.wavelet_cb.grid(row=1, column=0, padx=8, pady=(0, 2), sticky="ew")

        ctk.CTkLabel(self.adv_frame, text="Level", anchor="w").grid(
            row=0, column=1, padx=8, pady=(6, 0), sticky="w"
        )
        self.level_cb = ctk.CTkComboBox(
            self.adv_frame,
            values=["auto", "2", "3", "4", "5"],
        )
        self.level_cb.set("auto")
        self.level_cb.grid(row=1, column=1, padx=8, pady=(0, 2), sticky="ew")

        ctk.CTkLabel(self.adv_frame, text="Threshold", anchor="w").grid(
            row=0, column=2, padx=8, pady=(6, 0), sticky="w"
        )
        self.thresh_cb = ctk.CTkComboBox(
            self.adv_frame,
            values=["soft", "hard"],
        )
        self.thresh_cb.set("soft")
        self.thresh_cb.grid(row=1, column=2, padx=8, pady=(0, 2), sticky="ew")

        # Explanations
        wavelet_help = ctk.CTkLabel(
            self.adv_frame,
            text="Wavelet: which smoothing family to use (db4 is a solid default).",
            font=ctk.CTkFont(size=10),
            text_color=("gray70", "gray60"),
        )
        wavelet_help.grid(row=2, column=0, padx=8, pady=(0, 6), sticky="w")

        level_help = ctk.CTkLabel(
            self.adv_frame,
            text="Level: how many DWT layers. Higher = more smoothing. 'auto' lets the app choose.",
            font=ctk.CTkFont(size=10),
            text_color=("gray70", "gray60"),
        )
        level_help.grid(row=2, column=1, padx=8, pady=(0, 6), sticky="w")

        thresh_help = ctk.CTkLabel(
            self.adv_frame,
            text="Threshold: soft = gentler denoising, hard = more aggressive noise removal.",
            font=ctk.CTkFont(size=10),
            text_color=("gray70", "gray60"),
        )
        thresh_help.grid(row=2, column=2, padx=8, pady=(0, 6), sticky="w")

        # --- Progress bar ---
        self.progress = ctk.CTkProgressBar(panel, mode="indeterminate")
        self.progress.grid(row=16, column=0, padx=16, pady=(4, 4), sticky="ew")
        self.progress.stop()

        # --- Run button ---
        run_btn = ctk.CTkButton(
            panel,
            text="Run DWT forecast",
            height=42,
            command=self.run_forecast,
        )
        run_btn.grid(row=17, column=0, padx=16, pady=(8, 16), sticky="ew")


    # ---------- RIGHT PANEL: Image + Output ----------

    def _build_right_panel(self):
        frame = ctk.CTkFrame(self, corner_radius=0)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Image on left
        self.image_label = ctk.CTkLabel(
            frame,
            text="DWT overlay plot will appear here.",
            anchor="center",
            justify="center",
        )
        self.image_label.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        # Text output on right
        self.output_box = ctk.CTkTextbox(frame, wrap="word")
        self.output_box.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")

    def log(self, msg: str, clear: bool = False):
        if clear:
            self.output_box.delete("1.0", "end")
        self.output_box.insert("end", msg + "\n")
        self.output_box.see("end")
        self.output_box.update_idletasks()

    # ---------- Main action ----------

    def run_forecast(self):
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            self.log("❌ Please enter a ticker.", clear=True)
            return

        temp = float(self.temp_var.get())

        # Optional dates (empty string -> None)
        start_str = self.start_entry.get().strip() or None
        end_str = self.end_entry.get().strip() or None

        # Advanced or defaults
        if self.adv_open:
            wavelet = self.wavelet_cb.get()
            level_str = self.level_cb.get()
            thresh = self.thresh_cb.get()
        else:
            wavelet = "db4"
            level_str = "auto"
            thresh = "soft"

        if level_str.lower() == "auto":
            level = None
        else:
            try:
                level = int(level_str)
            except ValueError:
                level = None

        # Log what we're about to do
        range_text = f"{start_str or 'earliest'} → {end_str or 'latest'}"
        self.log(
            f"▶ Building DWT series for {ticker} over {range_text} "
            f"(wavelet={wavelet}, level={level_str}, thresh={thresh})…",
            clear=True,
        )

        # Start progress bar
        self.progress.start()

        try:
            csv_path, combo_path = ensure_dwt_csv_for_ticker(
                ticker=ticker,
                start=start_str,
                end=end_str,
                save_dir=Path("outputs"),
                wavelet=wavelet,
                level=level,
                threshold_mode=thresh,
            )
            self.log(f"✅ DWT CSV ready: {csv_path}")

            # Show image if possible
            if PilImage is None:
                self.log("ℹ️ Install 'pillow' (py -m pip install pillow) to see the DWT plot image.")
            elif combo_path is not None and combo_path.exists():
                img = PilImage.open(combo_path)
                self.dwt_image = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=(500, 260),
                )
                self.image_label.configure(image=self.dwt_image, text="")
            else:
                self.image_label.configure(
                    image=None,
                    text="Overlay PNG not found yet; it should be generated on first run.",
                )

            self.log(f"🤖 Calling Gemini via predict_from_dwt_csv_gemini (temp={temp:.2f})…")

            # Capture all printed output from the predictor so we can show rationale too
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                pred = dwt_pred.predict_from_dwt_csv_gemini(str(csv_path), temperature=temp)
            console = buf.getvalue()

            self.log("\n──────────── GEMINI OUTPUT ────────────")
            if console.strip():
                for line in console.strip().splitlines():
                    self.log(line)
            else:
                self.log("(No output captured from predictor.)")

            if pred is not None:
                self.log(f"\n[GUI] Parsed predicted next DWT close: {pred:.4f}")
            else:
                self.log("\n[GUI] No numeric prediction returned (see output above).")

        except Exception as e:
            self.log("❌ Error during DWT forecast:")
            self.log(str(e))
            self.log(traceback.format_exc())
        finally:
            # Stop progress bar
            self.progress.stop()



if __name__ == "__main__":
    app = DWTApp()
    app.mainloop()
