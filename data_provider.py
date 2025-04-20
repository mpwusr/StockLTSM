import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def log(msg):
    print(f"[data_provider] {msg}")


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
