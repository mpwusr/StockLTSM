import csv
import os

from dotenv import set_key, load_dotenv


def extract_tickers_from_csv(file_path):
    symbols = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for cell in row:
                ticker = cell.strip().upper()
                if ticker and ticker.isalpha():
                    symbols.append(ticker)
    return symbols


def write_tickers_to_env(symbols, env_path=".env"):
    if not os.path.exists(env_path):
        open(env_path, "a").close()
    load_dotenv(env_path)
    tickers_str = ",".join(sorted(set(symbols)))
    set_key(env_path, "TICKERS", tickers_str)
    print(f" TICKERS updated in {env_path}:")
    print(tickers_str)


if __name__ == "__main__":
    csv_path = input("Enter path to picklist CSV file: ").strip()
    if not os.path.isfile(csv_path):
        print(" File not found.")
    else:
        tickers = extract_tickers_from_csv(csv_path)
        write_tickers_to_env(tickers)
