# README.txt

Project: StockLTSMTransformerQuantum
Author: Michael P. Williams

---

## OVERVIEW

StockLTSMTransformerQuantum is a PyQt5-based application for forecasting stock prices using classical (LSTM, Transformer, GRU+CNN) and quantum (QML via PennyLane) neural networks. It supports real-time diagnostics, visualization, and trading strategy simulation.

---

## REQUIREMENTS

Python 3.12 or later

Unzip zip file to filesystem (Mac OSX example)

```
% unzip STOCKLTSMTransformerQuantum.zip
Archive:  STOCKLTSMTransformerQuantum.zip
   creating: /Users/michaelwilliams/STOCKLSTMTransformerQuantum
  inflating: STOCKLSTMTransformerQuantum/models.py
  inflating: STOCKLSTMTransformerQuantum/requirements.txt
  inflating: STOCKLSTMTransformerQuantum/versions.txt
 extracting: STOCKLSTMTransformerQuantum/__init__.py
  inflating: STOCKLSTMTransformerQuantum/README.md
  inflating: STOCKLSTMTransformerQuantum/.env
  inflating: STOCKLSTMTransformerQuantum/README.txt
  inflating: STOCKLSTMTransformerQuantum/versions.py
  inflating: STOCKLSTMTransformerQuantum/main.py
 % cd STOCKLSTMTransformerQuantum
STOCKLSTMTransformerQuantum %
```

Create and activate a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate  (on macOS/Linux)
.venv\Scripts\activate     (on Windows)
```

Then install the required packages:

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

   python3 main.py

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
python3 versions.py
```

## Sample `versions.py`:

# versions.py

import platform
import pkg_resources
import sys

# Define the output file (if redirected)
output_file = "versions.txt"

# Custom class to write to both stdout and a file
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

# Open the output file and set up dual output
with open(output_file, 'w') as f:
    # Redirect stdout to both terminal and file
    sys.stdout = Tee(sys.__stdout__, f)

    # Your original code
    print("============================")
    print(f"Python Version: {platform.python_version()}")
    print("============================")
    print("Redirect to text file or stdout")
    print("Installed Package Versions:")

    for dist in sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower()):
        print(f"{dist.project_name}=={dist.version}")

# Restore original stdout
sys.stdout = sys.__stdout__


---

## NOTES

* Only a subset of historical stock data is included in this ZIP to limit size.
* All models are trained locally using CPU; no cloud services required.
* GUI includes forecast plots, model metrics (MAE, MAPE, R², Sharpe), and optional debugging overlay.

---

## LICENSE

MIT License

---

Contact: [michaelpwilliams@vt.edu](mailto:michaelpwilliams@vt.edu)
