import csv
import os
import requests
import yfinance as yf
from dotenv import set_key, load_dotenv


def download_from_gdrive(gdrive_url, output_path="temp_tickers.csv"):
    if "spreadsheets/d/" in gdrive_url:
        file_id = gdrive_url.split("spreadsheets/d/")[-1].split("/")[0]
        download_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    elif "id=" in gdrive_url:
        file_id = gdrive_url.split("id=")[-1].split("&")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    elif "file/d/" in gdrive_url:
        file_id = gdrive_url.split("file/d/")[-1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    else:
        raise ValueError("❌ Invalid Google Drive or Google Sheets URL format.")

    response = requests.get(download_url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ File downloaded to {output_path}")
        return output_path
    else:
        raise RuntimeError(f"❌ Download failed. Status code: {response.status_code}")


def extract_first_column_tickers(file_path):
    symbols = set()
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].strip().isalpha():
                symbols.add(row[0].strip().upper())
    return sorted(symbols)


def validate_tickers_yfinance(tickers):
    valid = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if "shortName" in info and info["shortName"]:
                valid.append(ticker)
        except Exception:
            pass
    return valid


def write_tickers_to_env(tickers, env_path=".env"):
    if not os.path.exists(env_path):
        open(env_path, "a").close()
    load_dotenv(env_path)
    tickers_str = ",".join(sorted(set(tickers)))
    set_key(env_path, "TICKERS", tickers_str)
    print(f"\n✅ TICKERS updated in {env_path}:")
    print(tickers_str)


if __name__ == "__main__":
    url = input("📎 Paste the Google Drive CSV shareable link: ").strip()
    try:
        local_csv = download_from_gdrive(url)
        raw_tickers = extract_first_column_tickers(local_csv)
        print(f"🔍 Found {len(raw_tickers)} raw tickers. Validating...")
        clean_tickers = validate_tickers_yfinance(raw_tickers)
        print(f"✅ {len(clean_tickers)} tickers validated.")
        write_tickers_to_env(clean_tickers)
        os.remove(local_csv)
    except Exception as e:
        print(f"❌ Error: {e}")
