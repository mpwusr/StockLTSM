# README.md

Project: StockLTSMTransformerQuantum
Author: Michael P. Williams

---

## OVERVIEW

StockLTSMTransformerQuantum is a PyQt5-based application for forecasting stock prices using classical (LSTM, Transformer, GRU+CNN) and quantum (QML via PennyLane) neural networks. It supports real-time diagnostics, visualization, and trading strategy simulation.

The GUI enables **dual-stock comparison** mode, allowing users to select two different tickers and compare:

* Their forecast trajectories (Short, Medium, Long horizon)
* Training performance metrics (Loss, MAE, MAPE, R², Sharpe Ratio)

---

## REQUIREMENTS

Unzip the ZIP file to your filesystem (Mac example):

```bash
% unzip StockLTSMTransformerQuantum.zip
Archive:  StockLTSMTransformerQuantum.zip
   creating: /Users/yourname/StockLTSMTransformerQuantum/
  inflating: StockLTSMTransformerQuantum/models.py
  inflating: StockLTSMTransformerQuantum/requirements.txt
  inflating: StockLTSMTransformerQuantum/versions.txt
 extracting: StockLTSMTransformerQuantum/__init__.py
  inflating: StockLTSMTransformerQuantum/README.md
  inflating: StockLTSMTransformerQuantum/.env
  inflating: StockLTSMTransformerQuantum/README.txt
  inflating: StockLTSMTransformerQuantum/versions.py
  inflating: StockLTSMTransformerQuantum/main.py
% cd StockLTSMTransformerQuantum
```

Python 3.12 or later

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on macOS/Linux
.venv\Scripts\activate     # on Windows
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

Note:

* The `ta` package wraps TA-Lib indicators. If you experience build errors on macOS, install TA-Lib via Homebrew:

```bash
brew install ta-lib
```

---

## ENVIRONMENT VARIABLES

Create a `.env` file in the project root with your API keys and desired tickers:

Example `.env`:

```
POLYGON_API_KEY=your_polygon_api_key_here or ="" for fallback
ALPHAVANTAGE_API_KEY=your_alpha_api_key_here or ="" for fallback
TICKERS=AAPL,TSLA,GOOGL
```

### Data Source Notes:

* You can choose from **Polygon**, **Yahoo Finance**, or **AlphaVantage** as your data source.
* If an API key is missing or the data source fails, the app will automatically **fall back to Yahoo Finance** to ensure continuity.
* proper syntax for fallback is in zip file ="""
---

## HOW TO RUN

1. Activate your virtual environment:

```bash
source .venv/bin/activate  # on macOS/Linux
.venv\Scripts\activate     # on Windows
```

2. Set up your `.env` file with the correct keys and tickers.

3. Launch the GUI:

```bash
python3 main.py
```

4. In the GUI:

   * Select a stock ticker and model (LSTM, Transformer, GRUCNN, QML)
   * Currently, there are bugs with Stock Comparison - Optionally enable comparison and select a second ticker
   * Choose a data source—focus on Yahoo and pick only Yahoo (Polygon, Yahoo, AlphaVantage)
   * the included .env file does not have API keys for Polygon or Alphavantage so Yahoo is  
   * only option without required API keys-leave .env as-is as it will default to Yahoo
   * Click 'Run' to start training and forecasting

---

## OUTPUT

* Forecast CSVs are saved to: `forecasts/{ticker}/`
* Training metrics and logs are saved to: `training_logs/{ticker}/`
* Debug trading logs (optional) saved to: `logs/trade_debug.txt`

---

## VERSION TRACKING

To log your Python and package versions, run:

```bash
python3 versions.py
```

### Sample `versions.py`

This version writes both to the terminal and to `versions.txt`:

```python
import platform
import pkg_resources
import sys

output_file = "versions.txt"

class Tee:
    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

with open(output_file, 'w') as f:
    sys.stdout = Tee(sys.__stdout__, f)

    print("============================")
    print(f"Python Version: {platform.python_version()}")
    print("============================")
    print("Installed Package Versions:")

    for dist in sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower()):
        print(f"{dist.project_name}=={dist.version}")

    sys.stdout = sys.__stdout__
```

---

## NOTES

* Only a subset of historical stock data is included in this ZIP to limit size.
* All models are trained locally using CPU; no cloud services required.
* GUI includes:

  * Forecast plots across 3 time horizons
  * Dual-stock comparison support
  * Model performance metrics
  * Toggleable debug trading logs

---

## LICENSE

MIT License

---

Contact: [mw00066@vt.edu](mailto:mw00066@vt.edu)
