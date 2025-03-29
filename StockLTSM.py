import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import Input
from sklearn.preprocessing import MinMaxScaler

# Download historical price data from Yahoo Finance
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2024-01-01"
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# Extract and reshape the closing prices
prices = data[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Parameters
sequence_length = 60  # Using 60 days of previous data to predict next day

# Create sequences
X, y = create_sequences(prices_scaled, sequence_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    Input(shape=(sequence_length, 1)),  # Explicitly define the input shape here
    LSTM(units=50, return_sequences=True),  # No need for input_shape now
    LSTM(units=50),
    Dense(units=1)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with validation data
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_test, y_test), verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Denormalize the predictions
train_predictions = scaler.inverse_transform(train_predictions)
y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Plot training and validation loss
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 2. Plot actual vs predicted prices
plt.subplot(2, 1, 2)
train_dates = data.index[sequence_length:train_size + sequence_length]
test_dates = data.index[train_size + sequence_length:len(prices)]

plt.plot(train_dates, y_train_unscaled, label='Actual Train Prices')
plt.plot(train_dates, train_predictions, label='Predicted Train Prices')
plt.plot(test_dates, y_test_unscaled, label='Actual Test Prices')
plt.plot(test_dates, test_predictions, label='Predicted Test Prices')
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()

# Print model performance
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss:.6f}')
print(f'Test Loss: {test_loss:.6f}')