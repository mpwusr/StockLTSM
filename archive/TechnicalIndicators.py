import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMA, MACD
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Assume your data loading and LSTM setup exists
# Example: Fetching data
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-04-05")
data = data[['Close', 'Volume']]

# Add technical indicators
data['SMA_20'] = SMA(data['Close'], window=20)
data['SMA_50'] = SMA(data['Close'], window=50)
data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
macd = MACD(data['Close'])
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

# Prepare data for LSTM (your existing preprocessing)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Volume']].dropna())
X, y = [], []
look_back = 60
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i])
    y.append(scaled_data[i, 0])  # Predicting 'Close'
X, y = np.array(X), np.array(y)

# Your LSTM model (simplified example)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# Trading strategy
def trading_strategy(predictions, data):
    positions = []
    cash = 10000  # Starting cash
    shares = 0
    stop_loss = 0.05  # 5% stop-loss
    take_profit = 0.1  # 10% take-profit

    for i in range(len(predictions)):
        pred_price = predictions[i]
        current_price = data['Close'].iloc[i + look_back]
        sma_20 = data['SMA_20'].iloc[i + look_back]
        sma_50 = data['SMA_50'].iloc[i + look_back]
        rsi = data['RSI'].iloc[i + look_back]
        macd = data['MACD'].iloc[i + look_back]
        macd_signal = data['MACD_Signal'].iloc[i + look_back]
        volatility = data['Close'].pct_change().rolling(20).std().iloc[i + look_back]

        # Entry: Prediction > current price, SMA_20 > SMA_50, RSI < 70, MACD bullish
        if pred_price > current_price and sma_20 > sma_50 and rsi < 70 and macd > macd_signal and cash > current_price:
            shares_to_buy = int(cash / current_price)
            shares += shares_to_buy
            cash -= shares_to_buy * current_price
            entry_price = current_price
            positions.append(f"Buy {shares_to_buy} shares at {current_price}")

        # Exit: Stop-loss, take-profit, or bearish signals
        elif shares > 0:
            if current_price <= entry_price * (1 - stop_loss) or current_price >= entry_price * (1 + take_profit) or macd < macd_signal:
                cash += shares * current_price
                positions.append(f"Sell {shares} shares at {current_price}")
                shares = 0

    return positions, cash

# Predict and trade
predictions = model.predict(X)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((len(predictions), X.shape[2]-1))), axis=1))[:, 0]
trades, final_cash = trading_strategy(predictions, data)
print("Trades:", trades)
print("Final Cash:", final_cash)
