import sys
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from models import build_lstm_model, build_transformer_model
from quantum import optimize_quantum_weights, quantum_predict_future
from utils import load_tickers_from_env

# Add imports for fine-tuning and model persistence
import os
import pennylane.numpy as qnp
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Fetch and preprocess stock data
def fetch_and_prepare_data(ticker, start="2020-01-01", end="2025-04-05"):
    data = yf.download(ticker, start=start, end=end)
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from sklearn.preprocessing import MinMaxScaler

    close_prices = data['Close'].squeeze()
    data = data[['Close', 'Volume']]
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
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), data_clean, scaler, look_back

# Strategy
def moving_average(data, window=3):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def trading_strategy(predictions, last_date, future_dates):
    positions, cash, shares = [], 10000, 0
    initial_cash = cash
    stop_loss, take_profit, window = 0.05, 0.1, 3
    ma_predictions = moving_average(predictions, window)
    ma_indices = range(window - 1, len(predictions))
    entry_price = 0
    for i in ma_indices:
        pred_price = predictions[i]
        trade_date = future_dates[i].strftime("%m%d%Y")
        ma_current = ma_predictions[i - (window - 1)]
        ma_previous = ma_predictions[i - window] if i > window - 1 else ma_predictions[0]
        if ma_current > ma_previous and cash > pred_price and pred_price > 0:
            shares_to_buy = max(1, int(cash / pred_price))
            shares += shares_to_buy
            cash -= shares_to_buy * pred_price
            entry_price = pred_price
            positions.append(f"{trade_date} Buy {shares_to_buy} shares at ${pred_price:.2f}")
        elif shares > 0:
            if pred_price <= entry_price * (1 - stop_loss) or pred_price >= entry_price * (1 + take_profit):
                cash += shares * pred_price
                positions.append(f"{trade_date} Sell {shares} shares at ${pred_price:.2f}")
                shares = 0
    return positions, f"${cash:.2f}", "Go" if cash > initial_cash else "No Go"

# Predictor
def predict_future(model, last_sequence, scaler, look_back, future_days):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred
    predictions = np.array(predictions).reshape(-1, 1)
    padded = np.concatenate((predictions, np.zeros((len(predictions), 6))), axis=1)
    return np.maximum(scaler.inverse_transform(padded)[:, 0], 0)

# Thread (updated for fine-tuning)
class ModelTrainerThread(QThread):
    finished = pyqtSignal(dict, str, object)

    def __init__(self, ticker, X, y, data, scaler, look_back, model_builder, predictor, use_quantum=False):
        super().__init__()
        self.ticker = ticker
        self.X, self.y, self.data, self.scaler, self.look_back = X, y, data, scaler, look_back
        self.model_builder = model_builder
        self.predictor = predictor
        self.use_quantum = use_quantum
        self.model_dir = f"models/{ticker}"
        os.makedirs(self.model_dir, exist_ok=True)

    def run(self):
        results = {}
        last_date = self.data.index[-1]

        # Split data into train and validation
        train_size = int(len(self.X) * 0.8)
        X_train, X_val = self.X[:train_size], self.X[train_size:]
        y_train, y_val = self.y[:train_size], self.y[train_size:]

        if self.use_quantum:
            # Load or initialize quantum weights
            weights_path = f"{self.model_dir}/quantum_weights.npy"
            if os.path.exists(weights_path):
                weights = qnp.load(weights_path)
            else:
                weights = qnp.random.random(4, requires_grad=True)  # n_qubits = 4 from quantum.py
            weights = self.predictor.optimize(X_train, y_train, iterations=100, verbose=True)
            qnp.save(weights_path, weights)  # Fixed argument order
            pred_func = lambda seq, days, horizon: self.predictor.predict(seq, weights, self.scaler, self.look_back, days, horizon)
        else:
            # Load or build model
            model_path = f"{self.model_dir}/lstm_model.keras" if self.model_builder == build_lstm_model else f"{self.model_dir}/transformer_model.keras"
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                except Exception as e:
                    print(f"Failed to load model: {e}. Building new model.")
                    model = self.model_builder((self.X.shape[1], self.X.shape[2]))
            else:
                model = self.model_builder((self.X.shape[1], self.X.shape[2]))

            # Fine-tune with validation
            model.compile(optimizer='adam', loss='mse')
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[checkpoint, early_stopping],
                verbose=0
            )
            model.save(model_path)  # Save using .keras format
            pred_func = lambda seq, days, horizon: predict_future(model, seq, self.scaler, self.look_back, days)

        last_seq = self.X[-1]
        for label, days in {"Short": 30, "Medium": 90, "Long": 365}.items():
            results[label] = pred_func(last_seq, days, label)
        self.finished.emit(results, self.ticker, last_date)

# GUI
class StockTradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Trading GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.tickers = load_tickers_from_env()
        self.ticker_data = {}
        self.output_texts = {}
        self.training_thread = None

        main = QWidget(self)
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(self.tickers)
        layout.addWidget(QLabel("Select Ticker:"))
        layout.addWidget(self.ticker_combo)

        button_layout = QHBoxLayout()
        self.lstm_btn = QPushButton("Run LSTM")
        self.trans_btn = QPushButton("Run Transformer")
        self.q_btn = QPushButton("Run Quantum")
        for b in [self.lstm_btn, self.trans_btn, self.q_btn]:
            button_layout.addWidget(b)
        layout.addLayout(button_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        output_layout = QHBoxLayout()
        for label in ["Short", "Medium", "Long"]:
            box = QTextEdit()
            box.setReadOnly(True)
            self.output_texts[label] = box
            output_layout.addWidget(box)
        layout.addLayout(output_layout)

        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.lstm_btn.clicked.connect(lambda: self.run("LSTM"))
        self.trans_btn.clicked.connect(lambda: self.run("Transformer"))
        self.q_btn.clicked.connect(lambda: self.run("Quantum"))

    def run(self, mode):
        self.progress.show()
        self.toggle_buttons(False, mode)

        ticker = self.ticker_combo.currentText()
        if ticker not in self.ticker_data:
            X, y, data, scaler, look_back = fetch_and_prepare_data(ticker)
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
        else:
            trainer = ModelTrainerThread(ticker, X, y, data, scaler, look_back,
                                         build_lstm_model,
                                         predict_future)

        trainer.finished.connect(self.on_complete)
        self.training_thread = trainer
        trainer.start()

    def toggle_buttons(self, enable, running_label=""):
        self.lstm_btn.setEnabled(enable)
        self.trans_btn.setEnabled(enable)
        self.q_btn.setEnabled(enable)

        btn_map = {
            "LSTM": self.lstm_btn,
            "Transformer": self.trans_btn,
            "Quantum": self.q_btn
        }
        for name, btn in btn_map.items():
            if name == running_label:
                btn.setText("Running..." if not enable else f"Run {name}")
            else:
                btn.setText(f"Run {name}")

    def on_complete(self, results, ticker, last_date):
        self.progress.hide()
        self.toggle_buttons(True)
        self.plot(results, last_date)

        for label, preds in results.items():
            future_dates = [last_date + timedelta(days=i) for i in range(1, len(preds) + 1)]
            trades, cash, go = trading_strategy(preds, last_date, future_dates)
            self.output_texts[label].setText(
                f"Initial $10K\n{label} Trades:\n" + "\n".join(trades) + f"\nCash: {cash}\nDecision: {go}"
            )

    def plot(self, pred_dict, last_date):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        real = self.ticker_data[self.ticker_combo.currentText()]["data"]
        ax.plot(real.index, real["Close"], label="Actual", color="blue")
        base = [last_date + timedelta(days=i) for i in range(1, 366)]
        ax.plot(base[:30], pred_dict["Short"], label="Short", color="green")
        ax.plot(base[30:90], pred_dict["Medium"][-60:], label="Medium", color="orange")
        ax.plot(base[90:], pred_dict["Long"][-275:], label="Long", color="red")
        ax.set_title(f"{self.ticker_combo.currentText()} Forecast")
        ax.legend()
        self.canvas.draw()

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = StockTradingGUI()
    win.show()
    sys.exit(app.exec_())