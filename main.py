
import os
import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane.numpy as qnp
import requests
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar, QTabWidget, QCheckBox
)
from dotenv import load_dotenv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model

from models import build_lstm_model, build_transformer_model, build_gru_cnn_model, trading_strategy, \
    optimize_quantum_weights, quantum_predict_future

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


### ---------------------------
### Yahoo Finance (via yFinance)
### ---------------------------
def fetch_from_yahoo(ticker, start="2020-01-01", end="2025-04-05"):
    import yfinance as yf
    log(f"Fetching {ticker} from Yahoo Finance")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data from Yahoo for {ticker}")
    return df


### ---------------------------
### Polygon.io (via requests)
### ---------------------------
def fetch_from_polygon(ticker, start="2020-01-01", end="2025-04-05", retries=3, backoff=2):
    POLYGON_KEY = os.getenv("POLYGON_API_KEY")
    if not POLYGON_KEY:
        raise RuntimeError("POLYGON_API_KEY not found in environment")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_KEY
    }

    for attempt in range(1, retries + 1):
        try:
            log(f"Attempt {attempt}: Fetching {ticker} from Polygon")
            res = requests.get(url, params=params)
            res.raise_for_status()
            results = res.json().get("results", [])
            if not results:
                raise ValueError("Empty results")
            df = pd.DataFrame(results)
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('t', inplace=True)
            df.rename(columns={
                'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
            }, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            log(f"Polygon fetch error: {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise


### ---------------------------
### Unified interface with fallback + caching
### ---------------------------
def fetch_stock_data(ticker, source="polygon", start="2020-01-01", end="2025-04-05", use_cache=True):
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{source}_{start}_{end}.csv")

    # Try loading from cache
    if use_cache and os.path.exists(cache_file):
        log(f"Loading {ticker} from cache")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Try preferred source
    try:
        if source == "polygon":
            df = fetch_from_polygon(ticker, start, end)
        elif source == "yahoo":
            df = fetch_from_yahoo(ticker, start, end)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    except Exception as e:
        log(f"Primary source failed: {e}")
        if source != "yahoo":
            log("Falling back to Yahoo Finance")
            df = fetch_from_yahoo(ticker, start, end)
        else:
            raise

    # Save to cache
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


def fetch_and_prepare_data(ticker, source="polygon", start="2020-01-01", end="2025-04-05"):
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from sklearn.preprocessing import MinMaxScaler

    data = fetch_stock_data(ticker, source=source, start=start, end=end)

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


def predict_future(model, last_sequence, scaler, look_back, future_days):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred  # Only update Close (assumed first feature)

    predictions = np.array(predictions).reshape(-1, 1)

    # ✅ Auto-determine how many features the scaler expects
    num_features = scaler.scale_.shape[0]
    padding = num_features - 1  # already have Close prediction
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

            pred_func = lambda seq, days, horizon: predict_future(model, seq, self.scaler, self.look_back, days)

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

        print(f"✅ Training log saved to {csv_path}")


class StockTradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.forecast_folder_path = None
        self.setWindowTitle("Stock Trading GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.tickers = load_tickers_from_env()
        self.ticker_data = {}
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
        self.source_combo.addItems(["polygon", "yahoo"])
        layout.addWidget(QLabel("Select Data Source:"))
        layout.addWidget(self.source_combo)

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

        # Forecast tab layout
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

        # Training tab layout
        self.training_layout = QVBoxLayout(self.training_tab)
        self.training_figure = plt.Figure()
        self.training_canvas = FigureCanvas(self.training_figure)
        self.training_layout.addWidget(self.training_canvas)

        self.lstm_btn.clicked.connect(lambda: self.run("LSTM"))
        self.trans_btn.clicked.connect(lambda: self.run("Transformer"))
        self.gru_cnn_btn.clicked.connect(lambda: self.run("GRUCNN"))
        self.q_btn.clicked.connect(lambda: self.run("QML"))

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

        # ✅ Show popup
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

        if sys.platform.startswith("darwin"):  # macOS
            os.system(f"open '{path}'")
        elif os.name == "nt":  # Windows
            os.startfile(path)
        elif os.name == "posix":  # Linux
            os.system(f"xdg-open '{path}'")

    def run(self, mode):
        self.progress.show()
        self.toggle_buttons(False, mode)
        self.debug_enabled = self.debug_checkbox.isChecked()



        self.current_model = mode
        ticker = self.ticker_combo.currentText()
        source = self.source_combo.currentText()
        if ticker not in self.ticker_data:
            X, y, data, scaler, look_back = fetch_and_prepare_data(ticker, source=source)
            self.ticker_data[ticker] = {"X": X, "y": y, "data": data, "scaler": scaler, "look_back": look_back}

        X, y, data, scaler, look_back = self.ticker_data[ticker].values()

        if mode == "Quantum":
            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         lambda _: None,
                                         type("QuantumWrap", (), {
                                             "optimize": optimize_quantum_weights,
                                             "predict": quantum_predict_future
                                         }),
                                         use_quantum=True)
        elif mode == "Transformer":
            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         build_transformer_model,
                                         predict_future)
        elif mode == "GRUCNN":
            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         build_gru_cnn_model,
                                         predict_future)
        else:
            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         build_lstm_model,
                                         predict_future)

        trainer.finished.connect(self.on_complete)
        self.training_thread = trainer
        trainer.start()

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
        self.plot_forecast(results, last_date)
        self.plot_training_curves(history)
        self.progress.hide()
        self.toggle_buttons(True)
        self.plot_forecast(results, last_date)
        self.plot_training_curves(history)

        self.last_forecast = results
        self.last_ticker = ticker
        self.save_button.setEnabled(True)

        for label, preds in results.items():
            future_dates = [last_date + timedelta(days=i) for i in range(1, len(preds) + 1)]
            trades, cash, go, stats = trading_strategy(preds, last_date, future_dates, verbose=self.debug_enabled)
            summary = f"Initial $10K\n{label} Trades:\n" + "\n".join(trades) + f"\nCash: {cash}\nDecision: {go}\n"
            summary += "\n" + "\n".join(f"{k}: {v}" for k, v in stats.items())
            self.output_texts[label].setText(summary)

    def plot_forecast(self, pred_dict, last_date):
        self.forecast_figure.clear()
        ax = self.forecast_figure.add_subplot(111)
        real = self.ticker_data[self.ticker_combo.currentText()]["data"]
        ax.plot(real.index, real["Close"], label="Actual", color="blue")
        base = [last_date + timedelta(days=i) for i in range(1, 366)]
        ax.plot(base[:30], pred_dict["Short"], label="Short", color="green")
        ax.plot(base[30:90], pred_dict["Medium"][-60:], label="Medium", color="orange")
        ax.plot(base[90:], pred_dict["Long"][-275:], label="Long", color="red")
        ax.set_title(f"{self.ticker_combo.currentText()} {self.current_model} Forecast")
        ax.legend()
        self.forecast_canvas.draw()

    def plot_training_curves(self, history):
        self.training_figure.clear()
        if not history:
            print("No training history available.")
            return

        print("Training history keys:", list(history.keys()))

        # Use constrained layout for spacing
        self.training_figure.set_constrained_layout(True)
        gs = self.training_figure.add_gridspec(3, 1)

        # Loss plot
        ax1 = self.training_figure.add_subplot(gs[0, 0])
        ax1.plot(history.get("loss", []), label="Train Loss")
        ax1.plot(history.get("val_loss", []), label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_ylabel("MSE")
        ax1.legend()

        # MAE plot
        ax2 = self.training_figure.add_subplot(gs[1, 0])
        ax2.plot(history.get("train_mae", []), label="Train MAE")
        ax2.plot(history.get("val_mae", []), label="Val MAE")
        ax2.set_title("Mean Absolute Error")
        ax2.set_ylabel("MAE")
        ax2.legend()

        # Validation metrics plot
        ax3 = self.training_figure.add_subplot(gs[2, 0])
        ax3.plot(history.get("val_mape", []), label="Val MAPE")
        ax3.plot(history.get("val_r2", []), label="Val R²")
        ax3.plot(history.get("val_sharpe", []), label="Sharpe Ratio")
        ax3.set_title("Validation Metrics")
        ax3.set_ylabel("Metric Value")
        ax3.legend()

        self.training_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = StockTradingGUI()
    win.show()
    sys.exit(app.exec_())
