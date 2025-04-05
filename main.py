import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMA, MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
import pennylane as qml

# Data Fetching and Preprocessing
def fetch_and_prepare_data(ticker, start="2020-01-01", end="2025-04-05"):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close', 'Volume']]
    data['SMA_20'] = SMA(data['Close'], window=20)
    data['SMA_50'] = SMA(data['Close'], window=50)
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.dropna())
    X, y = [], []
    look_back = 60
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predict 'Close'
    return np.array(X), np.array(y), data, scaler

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Transformer Model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=50)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Quantum Model (Simulated)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))

def quantum_predict(X, weights):
    preds = []
    for x in X[:, -1, :n_qubits]:
        pred = quantum_circuit(x, weights)
        preds.append(pred)
    return np.array(preds)

# Trading Strategy
def trading_strategy(predictions, data, scaler):
    predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), 6))), axis=1))[:, 0]
    positions = []
    cash = 10000
    shares = 0
    stop_loss = 0.05
    take_profit = 0.1

    for i in range(len(predictions)):
        pred_price = predictions[i]
        current_price = data['Close'].iloc[i + look_back]
        sma_20 = data['SMA_20'].iloc[i + look_back]
        sma_50 = data['SMA_50'].iloc[i + look_back]
        rsi = data['RSI'].iloc[i + look_back]
        macd = data['MACD'].iloc[i + look_back]
        macd_signal = data['MACD_Signal'].iloc[i + look_back]

        if pred_price > current_price and sma_20 > sma_50 and rsi < 70 and macd > macd_signal and cash > current_price:
            shares_to_buy = int(cash / current_price)
            shares += shares_to_buy
            cash -= shares_to_buy * current_price
            entry_price = current_price
            positions.append(f"Buy {shares_to_buy} shares at {current_price:.2f}")

        elif shares > 0:
            if current_price <= entry_price * (1 - stop_loss) or current_price >= entry_price * (1 + take_profit) or macd < macd_signal:
                cash += shares * current_price
                positions.append(f"Sell {shares} shares at {current_price:.2f}")
                shares = 0

    return positions, cash

# Main Execution
if __name__ == "__main__":
    ticker = "AAPL"
    X, y, data, scaler = fetch_and_prepare_data(ticker)

    # LSTM
    lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    lstm_preds = lstm_model.predict(X)
    lstm_trades, lstm_cash = trading_strategy(lstm_preds, data, scaler)
    print("LSTM Results:")
    print("Trades:", lstm_trades)
    print("Final Cash:", lstm_cash)

    # Transformer
    transformer_model = build_transformer_model((X.shape[1], X.shape[2]))
    transformer_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    transformer_preds = transformer_model.predict(X)
    transformer_trades, transformer_cash = trading_strategy(transformer_preds, data, scaler)
    print("\nTransformer Results:")
    print("Trades:", transformer_trades)
    print("Final Cash:", transformer_cash)

    # Quantum
    quantum_weights = np.random.random(n_qubits)
    quantum_preds = quantum_predict(X, quantum_weights)
    quantum_preds_scaled = (quantum_preds + 1) / 2
    quantum_trades, quantum_cash = trading_strategy(quantum_preds_scaled, data, scaler)
    print("\nQuantum Results:")
    print("Trades:", quantum_trades)
    print("Final Cash:", quantum_cash)
