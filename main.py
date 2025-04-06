import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
import pennylane as qml
import pennylane.numpy as qnp  # Corrected import for PennyLane's NumPy
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, \
    QPushButton, QTextEdit
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta


# Data Fetching and Preprocessing
def fetch_and_prepare_data(ticker, start="2020-01-01", end="2025-04-05"):
    data = yf.download(ticker, start=start, end=end)
    close_prices = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].iloc[:, 0]
    data = data[['Close', 'Volume']]

    sma_20 = SMAIndicator(close_prices, window=20).sma_indicator()
    data['SMA_20'] = sma_20
    sma_50 = SMAIndicator(close_prices, window=50).sma_indicator()
    data['SMA_50'] = sma_50
    rsi = RSIIndicator(close_prices, window=14).rsi()
    data['RSI'] = rsi
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
    X, y = np.array(X), np.array(y)
    return X, y, data_clean, scaler, look_back


# Predict Future Prices
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
    return scaler.inverse_transform(padded)[:, 0]


# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Transformer Model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=50)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = x[:, -1, :]
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# Quantum Model with Optimization
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))


def optimize_quantum_weights(X, y, scaler, look_back, iterations=10):
    weights = qnp.random.random(n_qubits, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    y = qnp.array(y, requires_grad=False)  # Convert y to qnp.array for compatibility
    for _ in range(iterations):
        def cost(weights):
            preds = []
            for x in X[:, -1, :n_qubits]:
                pred = quantum_circuit(x, weights)
                preds.append(pred)
            preds = qnp.array(preds)  # Convert list of ArrayBox to qnp.array
            return qnp.mean((preds - y) ** 2)

        weights = opt.step(cost, weights)
    return weights


def quantum_predict_future(last_sequence, weights, scaler, look_back, future_days):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        pred = quantum_circuit(current_sequence[-1, :n_qubits], weights)
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred
    predictions = (np.array(predictions) + 1) / 2
    padded = np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), 6))), axis=1)
    return scaler.inverse_transform(padded)[:, 0]


# Trading Strategy with Go/No Go Evaluation
def trading_strategy(predictions, last_date, future_days):
    positions = []
    cash = 10000
    initial_cash = cash
    shares = 0
    stop_loss = 0.05
    take_profit = 0.1

    for i in range(len(predictions)):
        pred_price = predictions[i]
        current_price = pred_price
        if i > 0 and pred_price > predictions[i - 1] and cash > current_price:
            shares_to_buy = int(cash / current_price)
            shares += shares_to_buy
            cash -= shares_to_buy * current_price
            entry_price = current_price
            positions.append(f"Buy {shares_to_buy} shares at ${current_price:.2f}")
        elif shares > 0:
            if current_price <= entry_price * (1 - stop_loss) or current_price >= entry_price * (1 + take_profit):
                cash += shares * current_price
                positions.append(f"Sell {shares} shares at ${current_price:.2f}")
                shares = 0

    final_cash = cash
    go_no_go = "Go" if final_cash > initial_cash else "No Go"
    return positions, f"${final_cash:.2f}", go_no_go


# GUI Class
class StockTradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Trading Demo")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Ticker Dropdown
        input_layout = QHBoxLayout()
        self.ticker_combo = QComboBox()
        self.tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        self.ticker_combo.addItems(self.tickers)
        input_layout.addWidget(QLabel("Select Ticker:"))
        input_layout.addWidget(self.ticker_combo)
        layout.addLayout(input_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.lstm_button = QPushButton("Run LSTM")
        self.transformer_button = QPushButton("Run Transformer")
        self.quantum_button = QPushButton("Run Quantum")
        button_layout.addWidget(self.lstm_button)
        button_layout.addWidget(self.transformer_button)
        button_layout.addWidget(self.quantum_button)
        layout.addLayout(button_layout)

        # Results in Horizontal Layout (Left to Right)
        results_layout = QHBoxLayout()
        self.horizons = {"Short (30 days)": 30, "Medium (90 days)": 90, "Long (1 year)": 365}
        self.output_texts = {}
        for horizon in self.horizons:
            result_widget = QTextEdit(f"{horizon} Results will appear here...")
            result_widget.setReadOnly(True)
            result_widget.setMinimumWidth(300)
            result_widget.setMinimumHeight(200)
            result_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.output_texts[horizon] = result_widget
            results_layout.addWidget(result_widget)
        layout.addLayout(results_layout)

        # Plot
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.lstm_button.clicked.connect(self.run_lstm)
        self.transformer_button.clicked.connect(self.run_transformer)
        self.quantum_button.clicked.connect(self.run_quantum)

        self.ticker_data = {}

    def fetch_data(self):
        for ticker in self.tickers:
            if ticker not in self.ticker_data:
                X, y, data, scaler, look_back = fetch_and_prepare_data(ticker)
                self.ticker_data[ticker] = {"X": X, "y": y, "data": data, "scaler": scaler, "look_back": look_back}

    def generate_future_dates(self, last_date, future_days):
        return [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    def plot_results(self, predictions_dict, title, last_date):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        historical_data = self.ticker_data[self.ticker_combo.currentText()]["data"]
        ax.plot(historical_data.index, historical_data['Close'].values, label="Actual Price ($)", color="blue")

        # Segment predictions into 30, 90, and 365 days with different colors
        full_predictions = predictions_dict["Long (1 year)"]
        short_dates = self.generate_future_dates(last_date, 30)
        medium_dates = self.generate_future_dates(short_dates[-1], 60)
        long_dates = self.generate_future_dates(medium_dates[-1], 275)

        ax.plot(short_dates, full_predictions[:30], label="Short (30 days) Prediction ($)", color="green")
        ax.plot(medium_dates, full_predictions[30:90], label="Medium (90 days) Prediction ($)", color="blue")
        ax.plot(long_dates, full_predictions[90:365], label="Long (1 year) Prediction ($)", color="red")

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self.canvas.draw()

    def run_lstm(self):
        self.fetch_data()
        results = {ticker: {} for ticker in self.tickers}
        selected_ticker = self.ticker_combo.currentText()
        predictions_dict = {}

        for ticker in self.tickers:
            X, y = self.ticker_data[ticker]["X"], self.ticker_data[ticker]["y"]
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            last_sequence = X[-1]
            for horizon, days in self.horizons.items():
                preds = predict_future(model, last_sequence, self.ticker_data[ticker]["scaler"],
                                       self.ticker_data[ticker]["look_back"], days)
                trades, cash, go_no_go = trading_strategy(preds, self.ticker_data[ticker]["data"].index[-1], days)
                results[ticker][horizon] = go_no_go
                if ticker == selected_ticker:
                    output = [f"Initial Investment: $10,000", f"Trades:\n{'\n'.join(trades)}",
                              f"Money in Reserve: {cash}", f"Decision: {go_no_go}"]
                    self.output_texts[horizon].setText("\n".join(output))
                if horizon == "Long (1 year)":
                    predictions_dict[horizon] = preds

        if selected_ticker:
            self.plot_results(predictions_dict, f"LSTM Future Predictions for {selected_ticker}",
                              self.ticker_data[selected_ticker]["data"].index[-1])
        print("LSTM Go/No Go:", results)

    def run_transformer(self):
        self.fetch_data()
        results = {ticker: {} for ticker in self.tickers}
        selected_ticker = self.ticker_combo.currentText()
        predictions_dict = {}

        for ticker in self.tickers:
            X, y = self.ticker_data[ticker]["X"], self.ticker_data[ticker]["y"]
            model = build_transformer_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            last_sequence = X[-1]
            for horizon, days in self.horizons.items():
                preds = predict_future(model, last_sequence, self.ticker_data[ticker]["scaler"],
                                       self.ticker_data[ticker]["look_back"], days)
                trades, cash, go_no_go = trading_strategy(preds, self.ticker_data[ticker]["data"].index[-1], days)
                results[ticker][horizon] = go_no_go
                if ticker == selected_ticker:
                    output = [f"Initial Investment: $10,000", f"Trades:\n{'\n'.join(trades)}",
                              f"Money in Reserve: {cash}", f"Decision: {go_no_go}"]
                    self.output_texts[horizon].setText("\n".join(output))
                if horizon == "Long (1 year)":
                    predictions_dict[horizon] = preds

        if selected_ticker:
            self.plot_results(predictions_dict, f"Transformer Future Predictions for {selected_ticker}",
                              self.ticker_data[selected_ticker]["data"].index[-1])
        print("Transformer Go/No Go:", results)

    def run_quantum(self):
        self.fetch_data()
        results = {ticker: {} for ticker in self.tickers}
        selected_ticker = self.ticker_combo.currentText()
        predictions_dict = {}

        for ticker in self.tickers:
            X, y = self.ticker_data[ticker]["X"], self.ticker_data[ticker]["y"]
            weights = optimize_quantum_weights(X, y, self.ticker_data[ticker]["scaler"],
                                               self.ticker_data[ticker]["look_back"], iterations=10)
            last_sequence = X[-1]
            for horizon, days in self.horizons.items():
                preds = quantum_predict_future(last_sequence, weights, self.ticker_data[ticker]["scaler"],
                                               self.ticker_data[ticker]["look_back"], days)
                trades, cash, go_no_go = trading_strategy(preds, self.ticker_data[ticker]["data"].index[-1], days)
                results[ticker][horizon] = go_no_go
                if ticker == selected_ticker:
                    output = [f"Initial Investment: $10,000", f"Trades:\n{'\n'.join(trades)}",
                              f"Money in Reserve: {cash}", f"Decision: {go_no_go}"]
                    self.output_texts[horizon].setText("\n".join(output))
                if horizon == "Long (1 year)":
                    predictions_dict[horizon] = preds

        if selected_ticker:
            self.plot_results(predictions_dict, f"Quantum Future Predictions for {selected_ticker}",
                              self.ticker_data[selected_ticker]["data"].index[-1])
        print("Quantum Go/No Go:", results)


# Main Execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockTradingGUI()
    window.show()
    sys.exit(app.exec_())