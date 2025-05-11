import warnings
import pennylane as qml
import pennylane.numpy as qnp
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, Add, Conv1D, MaxPooling1D
)
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Model
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
import os
import numpy as np
from ta.momentum import RSIIndicator


def moving_average(data, window=3):
    return np.convolve(data, np.ones(window) / window, mode='valid')


def compute_rsi_from_series(close_series, period=14):
    rsi_indicator = RSIIndicator(close_series, window=period)
    return rsi_indicator.rsi().dropna().values


def trading_strategy(predictions, last_date, future_dates, rsi_series=None, rsi_threshold=(30, 70), verbose=False):
    positions = []
    cash = 10000
    initial_cash = cash
    shares = 0
    stop_loss = 0.05
    take_profit = 0.1
    window = 3
    entry_price = 0
    num_trades = 0
    wins = 0
    losses = 0

    def debug_log(msg):
        print(msg)
        with open("logs/trade_debug.txt", "a") as f:
            f.write(msg + "\n")

    def moving_average(data, window=3):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    ma_predictions = moving_average(predictions, window)
    ma_indices = range(window - 1, len(predictions))

    for i in ma_indices:
        pred_price = predictions[i]
        date_str = future_dates[i].strftime("%m%d%Y")
        ma_current = ma_predictions[i - (window - 1)]
        ma_previous = ma_predictions[i - window] if i > window - 1 else ma_predictions[0]

        rsi_ok = True
        if rsi_series is not None and i < len(rsi_series):
            rsi = rsi_series[i]
            rsi_ok = rsi < rsi_threshold[1]  # Avoid overbought

        if shares == 0 and ma_current > ma_previous and rsi_ok and cash > pred_price > 0:
            shares_to_buy = max(1, int(cash / pred_price))
            shares += shares_to_buy
            cash -= shares_to_buy * pred_price
            entry_price = pred_price
            num_trades += 1
            positions.append(f"{date_str} Buy {shares_to_buy} @ ${pred_price:.2f}")
            if verbose:
                os.makedirs("logs", exist_ok=True)
                debug_log(f"BUY on {date_str} at {pred_price:.2f}")

        elif shares > 0:
            # Stop loss or take profit
            if pred_price <= entry_price * (1 - stop_loss) or pred_price >= entry_price * (1 + take_profit):
                proceeds = shares * pred_price
                cash += proceeds
                profit = proceeds - (entry_price * shares)
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                positions.append(f"{date_str} Sell {shares} @ ${pred_price:.2f} | {'Win' if profit > 0 else 'Loss'}")
                if verbose:
                    os.makedirs("logs", exist_ok=True)
                    debug_log(f"BUY on {date_str} at {pred_price:.2f}")
                shares = 0

    if shares > 0:
        final_price = predictions[-1]
        proceeds = shares * final_price
        cash += proceeds
        profit = proceeds - (entry_price * shares)
        if profit > 0:
            wins += 1
        else:
            losses += 1
        positions.append(
            f"{future_dates[-1].strftime('%m%d%Y')} Final Sell {shares} @ ${final_price:.2f} | {'Win' if profit > 0 else 'Loss'}")
        if verbose:
            os.makedirs("logs", exist_ok=True)
            debug_log(f"FINAL SELL at {final_price:.2f}")

    return_pct = ((cash - initial_cash) / initial_cash) * 100
    go_no_go = "Go" if cash > initial_cash else "No Go"

    stats = {
        "Return %": f"{return_pct:.2f}%",
        "Trades": num_trades,
        "Wins": wins,
        "Losses": losses
    }

    return positions, f"${cash:.2f}", go_no_go, stats


def compute_trade_stats(positions, initial_cash, final_cash):
    total_return = final_cash / initial_cash - 1
    trades = [p for p in positions if "Buy" in p or "Sell" in p]
    buy_prices = [float(p.split('@ $')[1]) for p in positions if "Buy" in p]
    sell_prices = [float(p.split('@ $')[1]) for p in positions if "Sell" in p]

    wins = sum(1 for i in range(min(len(buy_prices), len(sell_prices)))
               if sell_prices[i] > buy_prices[i])
    losses = sum(1 for i in range(min(len(buy_prices), len(sell_prices)))
                 if sell_prices[i] < buy_prices[i])
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades else 0.0

    return {
        "Return %": f"{total_return:.2%}",
        "Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate": f"{win_rate:.1%}"
    }


def add_noise_debug_overlay(ax, predictions, dates, label="Quantum Volatility", fill=False):
    diff = np.abs(np.diff(predictions))
    smoothed = np.convolve(diff, np.ones(5) / 5, mode="same")
    ax2 = ax.twinx()
    ax2.plot(dates[1:], smoothed, color="gray", alpha=0.3, label=label, linestyle="dotted")
    if fill:
        ax2.fill_between(dates[1:], smoothed, alpha=0.1, color="gray")
    ax2.set_ylabel("Prediction Volatility", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)


def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape')
        ],
        run_eagerly=False
    )
    return model


def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn_output = Dense(128, activation="relu")(x)
    ffn_output = Dense(input_shape[-1])(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = x[:, -1, :]
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape')
        ],
        run_eagerly=False
    )
    return model


def build_gru_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(64, return_sequences=True)(x)
    x = GRU(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape')
        ],
        run_eagerly=False
    )
    return model


n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)

    for layer in range(4):
        for i in range(n_qubits):
            qml.RX(weights[layer * n_qubits + i], wires=i)
            qml.RZ(weights[layer * n_qubits + i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))


def normalize_input(x):

    if len(x.shape) == 1:
        raw = x[:n_qubits]
    else:
        raw = np.mean(x, axis=0)[:n_qubits]

    min_val = np.min(raw)
    max_val = np.max(raw)
    scaled = (raw - min_val) / (max_val - min_val + 1e-6)
    print(f"Normalized input (first {n_qubits}):", scaled)
    return (scaled * 2 * np.pi) - np.pi


def smooth_predictions(preds, window=3):
    return np.convolve(preds, np.ones(window) / window, mode="same")


def optimize_quantum_weights(X, y, iterations=300, verbose=False):
    weights = qnp.random.random(4 * n_qubits, requires_grad=True)  # Adjusted for 4 layers
    opt = qml.AdamOptimizer(stepsize=0.1)  # Reverted to AdamOptimizer
    y = qnp.array(y, requires_grad=False)
    for step in range(iterations):
        def cost(w):
            preds = []
            for x in X[:, -1]:
                x_scaled = normalize_input(x)
                pred = quantum_circuit(x_scaled, w)
                preds.append(pred)
            preds = qnp.array(preds)
            if verbose and step % 10 == 0:
                print(f"[Step {step}] Sample preds:", preds[:5])
            return qnp.mean((preds - y) ** 2)

        weights = opt.step(cost, weights)
    return weights


def quantum_predict_future(last_sequence, weights, scaler, look_back, future_days, horizon="Short"):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        x_scaled = normalize_input(current_sequence)
        pred = quantum_circuit(x_scaled, weights)
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred

    predictions = np.array(predictions)
    print(f"[Horizon: {horizon}] Raw predictions (first 10):", predictions[:10])

    predictions = (predictions + 1) / 2  # Normalize to [0, 1]
    print(f"[Horizon: {horizon}] Normalized predictions (first 10):", predictions[:10])

    # Removed smooth_predictions to allow more variability
    print(f"[Horizon: {horizon}] Predictions after normalization (first 10):", predictions[:10])

    padded = np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), 6))), axis=1)
    final_predictions = np.maximum(scaler.inverse_transform(padded)[:, 0], 0)
    print(f"[Horizon: {horizon}] Final predictions (first 10):", final_predictions[:10])

    return final_predictions
