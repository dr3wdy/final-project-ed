Installs: 
py -m pip install --upgrade pip setuptools wheel
py -m pip install yfinance pywavelets matplotlib pandas numpy
py -m pip install google-generativeai python-dotenv
py -m pip install scikit-learn
py -m pip install yfinance pywavelets matplotlib pandas numpy
py -m pip install google-generativeai python-dotenv
py -m pip install pillow
py -m pip install customtkinter

### How to call
py stock_gui.py


### How to call (not using gui):
## OHLVC:
py predict_from_OHLVC.py
# then, enter ticker you want to evaluate
## Predicict from DWT:
# Run DWT tool with your Ticker, start and end
py dwt_tool.py --ticker NVDA --start 2024-01-01 --end 2025-11-01 --save_dir outputs
# Run Predictor
py predict_from_dwt_csv.py
# then paste your CSV file Path: C:\...\outputs\NVDA_original_and_denoised.csv