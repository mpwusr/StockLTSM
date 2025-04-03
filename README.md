# Stock Price Prediction with LSTM

This repository contains a Python script that uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data from Yahoo Finance. The script downloads data, preprocesses it, trains an LSTM model, and visualizes the results.

## Dataset
- **Source**: Yahoo Finance via `yfinance`
- **Ticker**: AAPL (Apple Inc.) - configurable
- **Time Period**: January 1, 2010, to January 1, 2024
- **Data**: Daily closing prices

## Features
- **Data Processing**
  - Downloads historical stock prices
  - Normalizes data using MinMaxScaler
  - Creates sequences (60-day windows) for time series prediction
- **Model**
  - LSTM architecture with two layers (50 units each)
  - Dense output layer for single-value prediction
- **Training**
  - 70/30 train-test split
  - 50 epochs with Adam optimizer and MSE loss
- **Visualization**
  - Training and validation loss over epochs
  - Actual vs. predicted prices for train and test sets

## Requirements
```bash
pip install matplotlib numpy yfinance keras sklearn

## Usage
1. Ensure an internet connection for data download
2. Run the script:
```bash
python stock_price_lstm.py
```
3. View outputs:
   - Console: Train and test loss values
   - Plots: Loss curves and price predictions

## Script Structure
- **Data Retrieval**: Downloads stock data using `yfinance`
- **Preprocessing**: Normalizes and sequences data
- **Model Definition**: Builds LSTM model with Keras
- **Training**: Fits model with validation
- **Prediction**: Generates and denormalizes predictions
- **Visualization**: Plots results with Matplotlib

## Configuration
Key parameters in the script:
- `ticker`: Stock symbol (default: "AAPL")
- `start_date`: Start date (default: "2010-01-01")
- `end_date`: End date (default: "2024-01-01")
- `sequence_length`: Days of historical data per prediction (default: 60)
- `epochs`: Training iterations (default: 50)
- `batch_size`: Training batch size (default: 32)

## Output
- **Console**:
  ```
  Train Loss: 0.000123
  Test Loss: 0.000456
  ```
- **Plots**:
  - Top: Training and validation loss over epochs
  - Bottom: Actual vs. predicted stock prices over time

## Example Output
![Sample Plot](example_output.png) *(Add this file manually if desired)*

## Notes
- Adjust `ticker`, `start_date`, and `end_date` to analyze different stocks or periods
- Model performance may vary with different sequence lengths or epochs
- Requires stable internet for `yfinance` data retrieval
- Uses MinMaxScaler; other scalers (e.g., StandardScaler) could be substituted

## License
This project is open-source and available under the MIT License.
```
