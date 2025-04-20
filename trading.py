import os


def moving_average(data, window=3):
    return np.convolve(data, np.ones(window) / window, mode='valid')


def compute_rsi_from_series(close_series, period=14):
    rsi_indicator = RSIIndicator(close_series, window=period)
    return rsi_indicator.rsi().dropna().values


import numpy as np
from ta.momentum import RSIIndicator

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
        return np.convolve(data, np.ones(window)/window, mode='valid')

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

    # Final sell if still holding
    if shares > 0:
        final_price = predictions[-1]
        proceeds = shares * final_price
        cash += proceeds
        profit = proceeds - (entry_price * shares)
        if profit > 0:
            wins += 1
        else:
            losses += 1
        positions.append(f"{future_dates[-1].strftime('%m%d%Y')} Final Sell {shares} @ ${final_price:.2f} | {'Win' if profit > 0 else 'Loss'}")
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
