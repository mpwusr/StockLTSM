import os
import sys
import time
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane.numpy as qnp
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar, QTabWidget, QCheckBox, QLineEdit, QSizePolicy
)
from dotenv import load_dotenv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, r2_score
)

from models import build_lstm_model, build_transformer_model, build_gru_cnn_model, trading_strategy, \
    optimize_quantum_weights, quantum_predict_future
import requests

ALPHA_BASE_URL = "https://www.alphavantage.co/query"

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

MODEL_COLORS = {
    "LSTM": "blue",
    "Transformer": "orange",
    "GRUCNN": "purple",
    "QML": "green"
}

MODEL_LABELS = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "GRUCNN": "GRUCNN",
    "QML": "Quantum Machine Learning"
}

load_dotenv()

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def log(msg):
    print(f"[data_provider] {msg}")


def load_tickers_from_env():
    load_dotenv()
    return os.getenv("TICKERS", "").split(",")


def fetch_from_yahoo(ticker, start="2020-01-01", end="2025-04-05", retries=3, pause=1):
    for attempt in range(1, retries+1):
        try:
            log(f"Fetching {ticker} from Yahoo Finance (attempt {attempt})")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )
            if df.empty:
                raise ValueError(f"No data from Yahoo for {ticker}")
            return df
        except Exception as e:
            log(f"Yahoo fetch error: {e}")
            if attempt < retries:
                log(f"Sleeping {pause}s before retry…")
                time.sleep(pause)
            else:
                raise


def fetch_from_polygon(ticker, start="2020-01-01", end="2025-04-05", retries=3, backoff=2):
    POLYGON_KEY = os.getenv("POLYGON_API_KEY")
    if not POLYGON_KEY:
        raise RuntimeError("POLYGON_API_KEY not found in environment")

    all_results = []
    current_start = start
    limit = 50000  # Polygon's max per request

    while True:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{current_start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
            "apiKey": POLYGON_KEY
        }

        # retry loop
        for attempt in range(1, retries + 1):
            try:
                log(f"Attempt {attempt}: Fetching {ticker} from Polygon ({current_start} → {end})")
                res = requests.get(url, params=params, timeout=10)
                res.raise_for_status()
                data = res.json()
                batch = data.get("results", [])
                break
            except Exception as e:
                log(f"Polygon fetch error: {e}")
                if attempt < retries:
                    wait = min(30, backoff ** attempt)
                    log(f"Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to fetch {ticker} from Polygon after {retries} attempts.") from e

        # no more data → done
        if not batch:
            break

        # collect this batch
        all_results.extend(batch)

        # if fewer than our limit, we've exhausted the range
        if len(batch) < limit:
            break

        # otherwise bump `current_start` to the day after the last bar
        last_ts = batch[-1]["t"]  # in ms
        last_date = datetime.utcfromtimestamp(last_ts / 1000).date()
        current_start = (last_date + timedelta(days=1)).isoformat()

    if not all_results:
        raise ValueError(f"No data returned for {ticker} in given range.")

    # assemble DataFrame
    df = pd.DataFrame(all_results)[['o', 'h', 'l', 'c', 'v', 't']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 't']
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def resolve_date_range(start=None, end=None, years_ago=None):
    today = datetime.today().date()
    if years_ago:
        start_date = today - timedelta(days=365 * years_ago)
        end_date = today
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else datetime(2020, 1, 1).date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else today
    if start_date > end_date:
        raise ValueError("Start date must be before end date.")
    return start_date.isoformat(), end_date.isoformat()


def fetch_from_alphavantage(ticker: str, start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment")

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }
    resp = requests.get(ALPHA_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Check for rate‐limit or errors
    if "Error Message" in data:
        raise RuntimeError(f"AlphaVantage error: {data['Error Message']}")
    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"Unexpected response format: {data}")

    # Build DataFrame
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "6. volume": "Volume",
    }, inplace=True)

    # Filter by date
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.today()
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    # Cast to numeric
    df = df.astype({
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float,
    })

    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_stock_data(ticker, source="polygon", start=None, end=None, years_ago=None, use_cache=True):
    custom_range = (start is not None and end is not None and years_ago is None)
    start, end = resolve_date_range(start, end, years_ago)
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{source}_{start}_{end}.csv")
    if custom_range and os.path.exists(cache_file):
        log(f"Custom range selected: removing old cache {cache_file}")
        os.remove(cache_file)


    if use_cache and os.path.exists(cache_file):
        log(f"Loading {ticker} from cache")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    try:
        days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days

        if source == "polygon":
            df = fetch_from_polygon(ticker, start, end)
        elif source == "yahoo":
            df = fetch_from_yahoo(ticker, start, end)
        elif source == "alphavantage":
            df = fetch_from_alphavantage(ticker, start=start, end=end)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    except Exception as e:
        log(f"Primary source failed: {e}")
        if source != "yahoo":
            log("Falling back to Yahoo Finance")
            df = fetch_from_yahoo(ticker, start, end)
        else:
            raise
    if use_cache:
        df.to_csv(cache_file)
        log(f"Saved {ticker} data to cache: {cache_file}")
    return df


class MetricsTrackingCallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.history = {
            "train_mae": [],
            "val_mae": [],
            "val_mape": [],
            "val_r2": [],
            "val_sharpe": []
        }

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train, verbose=0).squeeze()
        y_val_pred = self.model.predict(self.X_val, verbose=0).squeeze()
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        val_mae = mean_absolute_error(self.y_val, y_val_pred)
        val_mape = mean_absolute_percentage_error(self.y_val, y_val_pred)
        val_r2 = r2_score(self.y_val, y_val_pred)
        daily_returns = np.diff(y_val_pred) / y_val_pred[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) != 0 else 0.0
        self.history["train_mae"].append(train_mae)
        self.history["val_mae"].append(val_mae)
        self.history["val_mape"].append(val_mape)
        self.history["val_r2"].append(val_r2)
        self.history["val_sharpe"].append(sharpe)
        if logs is not None:
            logs["val_mae"] = val_mae


def fetch_and_prepare_data(ticker, source="polygon", start=None, end=None, years_ago=None):
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from sklearn.preprocessing import MinMaxScaler
    data = fetch_stock_data(ticker, source=source, start=start, end=end, years_ago=years_ago)
    close_prices = data['Close'].squeeze()
    data['SMA_20'] = SMAIndicator(close_prices, window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
    data['RSI'] = RSIIndicator(close_prices, window=14).rsi()
    macd = MACD(close_prices)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data_clean = data.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_clean)
    X, y = [], []
    look_back = 60
    close_index = data_clean.columns.get_loc('Close')
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, close_index])
    return np.array(X), np.array(y), data_clean, scaler, look_back


def predict_future(model, last_sequence, scaler, future_days):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred  # Only update Close (assumed first feature)
    predictions = np.array(predictions).reshape(-1, 1)
    num_features = scaler.scale_.shape[0]
    padding = num_features - 1
    padded = np.concatenate((predictions, np.zeros((len(predictions), padding))), axis=1)
    return np.maximum(scaler.inverse_transform(padded)[:, 0], 0)


class ModelTrainerThread(QThread):
    finished = pyqtSignal(dict, str, object, dict)


    def __init__(self, ticker, X, y, data, scaler, look_back, model_builder, predictor, use_quantum=False):
        super().__init__()
        self.ticker = ticker
        self.X, self.y = X, y
        self.data = data
        self.scaler = scaler
        self.look_back = look_back
        self.model_builder = model_builder
        self.predictor = predictor
        self.use_quantum = use_quantum
        self.model_dir = f"models/{ticker}"
        os.makedirs(self.model_dir, exist_ok=True)

    def run(self):
        results = {}
        last_date = self.data.index[-1]
        train_size = int(len(self.X) * 0.8)
        X_train, X_val = self.X[:train_size], self.X[train_size:]
        y_train, y_val = self.y[:train_size], self.y[train_size:]

        if self.use_quantum:
            weights_path = f"{self.model_dir}/quantum_weights.npy"
            if os.path.exists(weights_path):
                weights = qnp.load(weights_path)
            else:
                weights = qnp.random.random(4, requires_grad=True)
            weights = self.predictor.optimize(X_train, y_train, iterations=100, verbose=True)
            qnp.save(weights_path, weights)
            pred_func = lambda seq, days, horizon: self.predictor.predict(
                seq, weights, self.scaler, self.look_back, days, horizon
            )
            self.history = {}
        else:
            model_key = (
                "lstm_model" if self.model_builder == build_lstm_model else
                "transformer_model" if self.model_builder == build_transformer_model else
                "gru_cnn_model"
            )
            model_path = f"{self.model_dir}/{model_key}.keras"
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                except Exception as e:
                    print(f"Model load failed: {e}. Rebuilding.")
                    model = self.model_builder((self.X.shape[1], self.X.shape[2]))
            else:
                model = self.model_builder((self.X.shape[1], self.X.shape[2]))

            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
            early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            metrics_tracker = MetricsTrackingCallback(X_train, y_train, X_val, y_val)

            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[checkpoint, early_stopping, metrics_tracker],
                verbose=0
            )
            model.save(model_path)
            self.history = history.history
            self.history.update(metrics_tracker.history)

            pred_func = lambda seq, days, horizon: predict_future(model, seq, self.scaler, days)
        last_seq = self.X[-1]
        for label, days in {"Short": 30, "Medium": 90, "Long": 365}.items():
            results[label] = pred_func(last_seq, days, label)
        self.finished.emit(results, self.ticker, last_date, self.history)
        import pandas as pd
        from datetime import datetime
        log_dir = os.path.join("training_logs", self.ticker)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = (
            "lstm" if self.model_builder == build_lstm_model else
            "transformer" if self.model_builder == build_transformer_model else
            "gru_cnn"
        )
        csv_path = os.path.join(log_dir, f"{model_tag}_training_log_{timestamp}.csv")
        pd.DataFrame(self.history).to_csv(csv_path, index=False)
        print(f" Training log saved to {csv_path}")


class StockTradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.training_axes_initialized = None
        self.compare_forecast = None
        self.compare_history = None
        self.compare_ticker = None
        self.compare_thread = None

        self.last_forecast = None
        self.last_history = None
        self.last_ticker = None
        self.last_date = None
        self.forecast_folder_path = None
        self.setWindowTitle("Stock Trading GUI")
        self.setGeometry(100, 100, 1200, 1000)  # was 800

        self.tickers = load_tickers_from_env()
        self.ticker_data = {}
        self.ticker_data = {}
        self.current_data_key = None
        self.compare_data_key = None
        self.output_texts = {}
        self.training_thread = None
        self.current_model = ""
        main = QWidget(self)
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(self.tickers)
        layout.addWidget(QLabel("Select Ticker:"))
        layout.addWidget(self.ticker_combo)
        self.compare_combo = QComboBox()
        self.compare_combo.addItems(self.tickers)
        layout.addWidget(QLabel("Compare With (Optional):"))
        layout.addWidget(self.compare_combo)
        self.compare_checkbox = QCheckBox("Enable Stock Comparison")
        layout.addWidget(self.compare_checkbox)
        self.save_button = QPushButton("Save Forecast")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_forecast)
        layout.addWidget(self.save_button)
        self.last_forecast = None
        self.last_ticker = None
        self.current_model = None
        button_layout = QHBoxLayout()
        self.lstm_btn = QPushButton("Run LSTM")
        self.trans_btn = QPushButton("Run Transformer")
        self.gru_cnn_btn = QPushButton("Run GRUCNN")
        self.q_btn = QPushButton("Run QML")
        self.debug_checkbox = QCheckBox("Debug Trading Logs")
        layout.addWidget(self.debug_checkbox)
        self.view_folder_button = QPushButton("View Folder")
        self.view_folder_button.setEnabled(False)
        self.view_folder_button.clicked.connect(self.open_forecast_folder)
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.view_folder_button)
        layout.addLayout(button_row)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["polygon", "yahoo", "alphavantage"])
        layout.addWidget(QLabel("Select Data Source:"))
        layout.addWidget(self.source_combo)
        layout.addWidget(QLabel("Select Date Range:"))
        self.date_range_combo = QComboBox()
        self.date_range_combo.addItems([
            "Default (2020–Today)", "Last 2 years", "Last 3 years", "Last 5 years", "Custom Range"
        ])
        layout.addWidget(self.date_range_combo)
        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        self.start_input.setPlaceholderText("Start Date (YYYY-MM-DD)")
        self.end_input.setPlaceholderText("End Date (YYYY-MM-DD)")
        self.start_input.hide()
        self.end_input.hide()
        layout.addWidget(self.start_input)
        layout.addWidget(self.end_input)
        self.date_range_combo.currentIndexChanged.connect(self.toggle_custom_date_inputs)
        for b in [self.lstm_btn, self.trans_btn, self.gru_cnn_btn, self.q_btn]:
            button_layout.addWidget(b)
        layout.addLayout(button_layout)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.tabs = QTabWidget()
        self.forecast_tab = QWidget()
        self.training_tab = QWidget()
        self.tabs.addTab(self.forecast_tab, "Forecast")
        self.tabs.addTab(self.training_tab, "Training Diagnostics")
        layout.addWidget(self.tabs)
        self.forecast_layout = QVBoxLayout(self.forecast_tab)
        output_layout = QHBoxLayout()

        for label in ["Short", "Medium", "Long"]:
            box = QTextEdit()
            box.setReadOnly(True)
            self.output_texts[label] = box
            output_layout.addWidget(box)
        self.forecast_layout.addLayout(output_layout)
        self.forecast_figure = plt.Figure()
        self.forecast_canvas = FigureCanvas(self.forecast_figure)
        self.forecast_layout.addWidget(self.forecast_canvas)

        self.training_layout = QVBoxLayout(self.training_tab)
        self.training_layout.setContentsMargins(0, 0, 0, 0)
        self.training_layout.setSpacing(0)
        self.training_layout.setAlignment(Qt.AlignTop)

        self.training_figure = plt.Figure(figsize=(12, 6), constrained_layout=True)
        self.training_canvas = FigureCanvas(self.training_figure)

        self.training_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.training_canvas.setMinimumHeight(600)

        self.training_layout.addWidget(self.training_canvas)

        self.lstm_btn.clicked.connect(lambda: self.run("LSTM"))
        self.trans_btn.clicked.connect(lambda: self.run("Transformer"))
        self.gru_cnn_btn.clicked.connect(lambda: self.run("GRUCNN"))
        self.q_btn.clicked.connect(lambda: self.run("QML"))

    def toggle_custom_date_inputs(self):
        is_custom = self.date_range_combo.currentText() == "Custom Range"
        self.start_input.setVisible(is_custom)
        self.end_input.setVisible(is_custom)

    def save_forecast(self):
        if not self.last_forecast or not self.last_ticker or not self.current_model:
            return
        from datetime import datetime
        import pandas as pd
        from PyQt5.QtWidgets import QMessageBox
        output_dir = os.path.join("forecasts", self.last_ticker)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        for label, preds in self.last_forecast.items():
            dates = [
                self.ticker_data[self.last_ticker]['data'].index[-1] + timedelta(days=i)
                for i in range(1, len(preds) + 1)
            ]
            df = pd.DataFrame({"Date": dates, "Forecast": preds})
            fname = f"{self.current_model}_{label}_{timestamp}.csv"
            full_path = os.path.join(output_dir, fname)
            df.to_csv(full_path, index=False)
            saved_files.append(fname)
        self.save_button.setEnabled(False)
        QMessageBox.information(
            self,
            "Forecast Saved",
            f"Saved forecast CSVs:\n\n" + "\n".join(saved_files),
            QMessageBox.Ok
        )
        self.view_folder_button.setEnabled(True)
        self.forecast_folder_path = output_dir  # Store path for the viewer

    def open_forecast_folder(self):
        if not hasattr(self, "forecast_folder_path") or not os.path.isdir(self.forecast_folder_path):
            return

        path = os.path.abspath(self.forecast_folder_path)

        if sys.platform.startswith("darwin"):
            os.system(f"open '{path}'")
        elif os.name == "nt":
            os.startfile(path)
        elif os.name == "posix":
            os.system(f"xdg-open '{path}'")

    def run(self, mode):
        self.progress.show()
        self.toggle_buttons(False, mode)
        self.debug_enabled = self.debug_checkbox.isChecked()
        self.current_model = mode
        self.training_axes_initialized = False
        self.forecast_figure.clear()
        self.training_figure.clear()

        ticker = self.ticker_combo.currentText()
        compare_enabled = self.compare_checkbox.isChecked()
        compare_ticker = self.compare_combo.currentText()
        source = self.source_combo.currentText()
        range_option = self.date_range_combo.currentText()
        start = end = None
        years_ago = None

        if "2" in range_option:
            years_ago = 2
        elif "3" in range_option:
            years_ago = 3
        elif "5" in range_option:
            years_ago = 5
        elif "Custom" in range_option:
            start = self.start_input.text().strip()
            end = self.end_input.text().strip()

        key = (ticker, source, start, end, years_ago)
        if key not in self.ticker_data:
            X, y, data, scaler, look_back = fetch_and_prepare_data(
                ticker, source=source, start=start, end=end, years_ago=years_ago
            )
            self.ticker_data[key] = {"X": X, "y": y, "data": data, "scaler": scaler, "look_back": look_back}
        self.current_data_key = key
        X, y, data, scaler, look_back = self.ticker_data[key].values()

        if compare_enabled and compare_ticker != ticker:
            compare_key = (compare_ticker, source, start, end, years_ago)
            if compare_key not in self.ticker_data:
                X2, y2, data2, scaler2, look_back2 = fetch_and_prepare_data(
                    compare_ticker, source=source, start=start, end=end, years_ago=years_ago
                )
                self.ticker_data[compare_key] = {"X": X2, "y": y2, "data": data2, "scaler": scaler2,
                                                 "look_back": look_back2}
            self.compare_data_key = compare_key
            X2, y2, data2, scaler2, look_back2 = self.ticker_data[compare_key].values()

        self.current_data_key = key

        if mode == "Quantum":
            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         lambda _: None,
                                         type("QuantumWrap", (), {
                                             "optimize": optimize_quantum_weights,
                                             "predict": quantum_predict_future
                                         }),
                                         use_quantum=True)
        else:
            model_fn = {
                "LSTM": build_lstm_model,
                "Transformer": build_transformer_model,
                "GRUCNN": build_gru_cnn_model
            }.get(mode, build_lstm_model)

            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         model_fn, predict_future)

        trainer.finished.connect(self.on_complete)
        self.training_thread = trainer
        trainer.start()
        if compare_enabled and compare_ticker != ticker:
            compare_key = (compare_ticker, source, start, end, years_ago)
            if compare_key not in self.ticker_data:
                X2, y2, data2, scaler2, look_back2 = fetch_and_prepare_data(
                    compare_ticker, source=source, start=start, end=end, years_ago=years_ago
                )
                self.ticker_data[compare_key] = {"X": X2, "y": y2, "data": data2, "scaler": scaler2,
                                                 "look_back": look_back2}
            self.compare_data_key = compare_key
            X2, y2, data2, scaler2, look_back2 = self.ticker_data[compare_key].values()

            model_fn = {
                "LSTM": build_lstm_model,
                "Transformer": build_transformer_model,
                "GRUCNN": build_gru_cnn_model
            }.get(mode, build_lstm_model)

            self.compare_thread = ModelTrainerThread(compare_ticker, X2, y2, data2, scaler2, look_back2,
                                                     model_fn, predict_future)
            self.compare_thread.finished.connect(
                lambda results, t, d, h: self.on_compare_complete(results, t, d, h)
            )
            self.compare_thread.start()

    def on_compare_complete(self, compare_results, compare_ticker, last_date, history):
        self.compare_forecast = compare_results
        self.compare_history = history
        self.compare_ticker = compare_ticker
        self.compare_last_date = last_date
        self.compare_thread = None

        if self.last_forecast is not None:
            self.plot_training_curves(self.compare_history, label_prefix=self.compare_ticker)

    def toggle_buttons(self, enable, running_label=""):
        for btn, name in [
            (self.lstm_btn, "LSTM"),
            (self.trans_btn, "Transformer"),
            (self.gru_cnn_btn, "GRUCNN"),
            (self.q_btn, "QML")
        ]:
            btn.setEnabled(enable)
            btn.setText("Running..." if name == running_label and not enable else f"Run {name}")

    def on_complete(self, results, ticker, last_date, history):
        self.progress.hide()
        self.toggle_buttons(True)

        self.last_forecast = results
        self.last_ticker = ticker
        self.last_history = history
        self.last_date = last_date
        self.save_button.setEnabled(True)

        for label, preds in results.items():
            future_dates = [last_date + timedelta(days=i) for i in range(1, len(preds) + 1)]
            trades, cash, go, stats = trading_strategy(preds, last_date, future_dates,
                                                       verbose=self.debug_checkbox.isChecked())
            summary = f"Initial $10K\n{label} Trades:\n" + "\n".join(trades) + f"\nCash: {cash}\nDecision: {go}\n"
            summary += "\n" + "\n".join(f"{k}: {v}" for k, v in stats.items())
            self.output_texts[label].setText(summary)

        self.plot_forecast(self.last_forecast, last_date)

        self.training_axes_initialized = False  # clear axes before drawing
        self.plot_training_curves(history, label_prefix=self.last_ticker)

    def plot_forecast(self, pred_dict, last_date, compare_results=None, compare_ticker=None):
        self.forecast_figure.clear()
        ax = self.forecast_figure.add_subplot(111)

        # Base forecast (primary ticker)
        real = self.ticker_data[self.current_data_key]["data"]
        ax.plot(real.index, real["Close"], label=f"{self.last_ticker} Actual", color="black")

        base = [last_date + timedelta(days=i) for i in range(1, 366)]
        ax.plot(base[:30], pred_dict["Short"], label=f"{self.last_ticker} Short", color="blue")
        ax.plot(base[30:90], pred_dict["Medium"][-60:], label=f"{self.last_ticker} Medium", color="orange")
        ax.plot(base[90:], pred_dict["Long"][-275:], label=f"{self.last_ticker} Long", color="green")

        if self.compare_forecast and self.compare_data_key:
            compare_real = self.ticker_data[self.compare_data_key]["data"]
            ax.plot(compare_real.index, compare_real["Close"], label=f"{self.compare_ticker} Actual", linestyle="--",
                    color="gray")

            base2 = [self.compare_last_date + timedelta(days=i) for i in range(1, 366)]
            ax.plot(base2[:30], self.compare_forecast["Short"], label=f"{self.compare_ticker} Short", linestyle="--",
                    color="purple")
            ax.plot(base2[30:90], self.compare_forecast["Medium"][-60:], label=f"{self.compare_ticker} Medium",
                    linestyle="--", color="brown")
            ax.plot(base2[90:], self.compare_forecast["Long"][-275:], label=f"{self.compare_ticker} Long",
                    linestyle="--", color="red")

        ax.set_xlim(real.index.min(), base[-1])
        ax.set_title("Forecast Comparison")
        ax.legend(loc="best")
        ax.figure.autofmt_xdate()
        self.forecast_canvas.draw()

    def plot_training_curves(self, history, label_prefix=""):
        if not history:
            print("No training history available.")
            return

        if not hasattr(self, "training_axes_initialized") or not self.training_axes_initialized:
            self.training_figure.clear()
            self.training_figure.set_constrained_layout(False)
            self.training_figure.subplots_adjust(right=0.78)
            self.gs = self.training_figure.add_gridspec(3, 1)

            self.ax_loss = self.training_figure.add_subplot(self.gs[0, 0])
            self.ax_mae = self.training_figure.add_subplot(self.gs[1, 0])
            self.ax_val = self.training_figure.add_subplot(self.gs[2, 0])

            #self.ax_loss.set_title("Loss", loc="left")
            #self.ax_mae.set_title("Mean Absolute Error", loc="left")
            #self.ax_val.set_title("Validation Metrics", loc="left")

            self.ax_loss.set_ylabel("MSE")
            self.ax_mae.set_ylabel("MAE")
            self.ax_val.set_ylabel("Metric Value")
            self.ax_val.set_ylim(bottom=-5, top=10)

            self.training_axes_initialized = True

        def lbl(name):
            return f"{label_prefix} {name}" if label_prefix else name

        if "loss" in history and len(history["loss"]):
            self.ax_loss.plot(history["loss"], label=lbl("Train Loss"))
        if "val_loss" in history and len(history["val_loss"]):
            self.ax_loss.plot(history["val_loss"], label=lbl("Val Loss"))
        self.ax_loss.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

        if "train_mae" in history and len(history["train_mae"]):
            self.ax_mae.plot(history["train_mae"], label=lbl("Train MAE"))
        if "val_mae" in history and len(history["val_mae"]):
            self.ax_mae.plot(history["val_mae"], label=lbl("Val MAE"))
        self.ax_mae.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

        if "val_mape" in history and len(history["val_mape"]):
            self.ax_val.plot(history["val_mape"], label=lbl("Val MAPE"))
        if "val_r2" in history and len(history["val_r2"]):
            self.ax_val.plot(history["val_r2"], label=lbl("Val R²"))
        if "val_sharpe" in history and len(history["val_sharpe"]):
            self.ax_val.plot(history["val_sharpe"], label=lbl("Sharpe Ratio"))
        self.ax_val.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

        self.training_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = StockTradingGUI()
    win.show()
    sys.exit(app.exec_())
