# StockLTSMTransformerQuantum

A PyQt5-based AI stock forecasting app that combines classical and quantum machine learning models to simulate, visualize, and evaluate trading strategies across multiple time horizons (Short / Medium / Long).

---

## Overview

**StockLTSMTransformerQuantum** is an end-to-end forecasting and trading simulation toolkit that enables side-by-side comparison of:

- LSTM (Long Short-Term Memory)
- Transformer
- GRU + CNN Hybrid
- QML (Quantum Machine Learning via PennyLane)

It features a GUI for interactive use, real-time model training diagnostics, metrics visualizations, trading signal simulation, and optional debug logging.

---

## Features

- GUI built with PyQt5
- Multi-tab layout (Forecasts + Training Diagnostics)
- Model comparison with MAE, MAPE, R², Sharpe Ratio
- Toggleable debug logs
- Support for both Yahoo Finance and Polygon.io as data providers
- Training and forecast export (CSV)
- Training history auto-saved per model
- "Save Forecast" and "Open Folder" buttons
- Color-coded plots and dropdown data source switching
- Robust exception handling and logging

---

## Models

| Model        | Architecture             | Highlights                       |
|--------------|--------------------------|-----------------------------------|
| `LSTM`       | 2-layer LSTM             | Great for sequential time series |
| `Transformer`| Multi-head Attention     | Captures long-term dependencies  |
| `GRUCNN`     | Conv1D + GRU             | Combines pattern extraction + memory |
| `QML`        | PennyLane quantum model  | Lightweight and interpretable    |

---

## GUI Preview

| Forecast Tab                          | Training Diagnostics Tab            |
|--------------------------------------|-------------------------------------|
| ![Forecast](assets/forecast_sample.png) | ![Diagnostics](assets/diagnostics_sample.png) |

---

## Installation

```bash
git clone https://github.com/mpwusr/StockLTSMTransformerQuantum.git
cd StockLTSMTransformerQuantum
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Usage

1. Set your `.env` file:

```
POLYGON_API_KEY=your_key_here
```

2. Launch the GUI:

```bash
python main.py
```

3. Select ticker, model, and data source (Polygon/Yahoo).
4. Train model, visualize predictions, and export results.

---

## Directory Structure

```
├── data_provider.py         # Unified stock data API
├── main.py                  # GUI entry point
├── models.py                # Model definitions
├── quantum.py               # QML prediction logic
├── trading.py               # Strategy simulation + metrics
├── utils.py                 # Ticker loading, helpers
├── forecasts/               # Exported forecasts
├── training_logs/           # Per-model training logs
└── requirements.txt
```

---

## Sample Output

```text
Training log saved to training_logs/AAPL/lstm_training_log_20250420_1012.csv
 Saved forecast CSVs:
  - LSTM_Short_20250420_1015.csv
  - LSTM_Medium_20250420_1015.csv
  - LSTM_Long_20250420_1015.csv
```

---

## Model Comparison Snapshot

| Model      | Val MAE | Sharpe | MAPE (%) | R²     |
|------------|---------|--------|----------|--------|
| LSTM       | 3.12    | 1.45   | 6.78     | 0.81   |
| Transformer| 2.87    | 1.32   | 6.05     | 0.85   |
| GRUCNN     | 3.01    | 1.51   | 6.40     | 0.83   |
| QML        | 3.25    | 1.12   | 7.15     | 0.79   |

---

## Built With

- Python 3.12
- PyQt5
- TensorFlow / Keras
- PennyLane (for QML)
- yFinance + Polygon.io
- scikit-learn
- TA-Lib (via `ta` package)
- matplotlib

---

## License

MIT License © [mpwusr](https://github.com/mpwusr)

---

## Acknowledgments
based to some extent on https://medium.com/@albertoglvz25/predicting-stock-prices-with-an-lstm-model-in-python-26c7377b8ecb

Thanks to [PennyLane](https://pennylane.ai/), [Keras](https://keras.io/), and [Polygon.io](https://polygon.io/) for APIs and tooling.
