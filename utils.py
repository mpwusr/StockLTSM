import os
from dotenv import load_dotenv

def load_tickers_from_env():
    load_dotenv()
    return os.getenv("TICKERS", "").split(",")
