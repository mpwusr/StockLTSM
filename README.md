# README.md

Project: StockLTSMTransformerQuantum
Author: Michael P. Williams
Repository: [https://github.com/mpwusr/StockLTSMTransformerQuantum](https://github.com/mpwusr/StockLTSMTransformerQuantum)

---

## OVERVIEW

StockLTSMTransformerQuantum is a PyQt5-based application for forecasting stock prices using classical (LSTM, Transformer, GRU+CNN) and quantum (QML via PennyLane) neural networks. It supports real-time diagnostics, visualization, and trading strategy simulation.

---

## REQUIREMENTS

Python 3.12 or later

Install required packages using:

```
pip install -r requirements.txt
```

Note:

* The `ta` package wraps TA-Lib indicators. If you experience build errors on macOS, install TA-Lib via Homebrew:

  brew install ta-lib

---

## ENVIRONMENT VARIABLES

Create a `.env` file in the project root with your API keys and desired tickers:

Example `.env`:
POLYGON\_API\_KEY=your\_polygon\_api\_key\_here
ALPHAVANTAGE\_API\_KEY=your\_alpha\_api\_key\_here
TICKERS=AAPL,TSLA,GOOGL

---

## HOW TO RUN

1. Activate your virtual environment:

   source .venv/bin/activate  (on macOS/Linux)
   .venv\Scripts\activate     (on Windows)

2. Set up your `.env` file with the correct keys and tickers.

3. Launch the GUI:

   python main.py

4. In the GUI:

   * Select a stock ticker and model (LSTM, Transformer, GRUCNN, QML)
   * Choose a data source (Polygon, Yahoo, AlphaVantage)
   * Click 'Run' to start training and forecasting

---

## OUTPUT

* Forecast CSVs are saved to: `forecasts/{ticker}/`
* Training metrics and logs are saved to: `training_logs/{ticker}/`
* Debug trading logs (optional) saved to: `logs/trade_debug.txt`

---

## VERSION TRACKING

To log your Python and package versions, run:

```
python versions.py
```

## Sample `versions.py`:

import pkg\_resources
import platform

print(f"Python Version: {platform.python\_version()}")
print("Installed Package Versions:")

for dist in sorted(pkg\_resources.working\_set, key=lambda d: d.project\_name.lower()):
print(f"{dist.project\_name}=={dist.version}")

---

## NOTES

* Only a subset of historical stock data is included in this ZIP to limit size.
* All models are trained locally using CPU; no cloud services required.
* GUI includes forecast plots, model metrics (MAE, MAPE, R², Sharpe), and optional debugging overlay.

---

## LICENSE

MIT License (see GitHub repository)

---

Contact: [michaelpwilliams@vt.edu](mailto:michaelpwilliams@vt.edu)
