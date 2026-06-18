import os
import pandas as pd
import yfinance as yf
from datasets import load_dataset


def create_directory_structure():
    dirs = ['data', 'src', 'notebooks']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def fetch_fx_data(ticker="EURUSD=X", period="2y", interval="1h"):
    print(f"Downloading market price action for {ticker}...")
    fx_data = yf.download(tickers=ticker, period=period, interval=interval)
    if isinstance(fx_data.columns, pd.MultiIndex):
        fx_data.columns = fx_data.columns.get_level_values(0)
    output_path = os.path.join('data', 'fx_prices.csv')
    fx_data.to_csv(output_path)
    return fx_data


def fetch_speech_corpus():
    print("Streaming speech texts from 'istat-ai/ECB-FED-speeches'...")
    dataset = load_dataset("istat-ai/ECB-FED-speeches", split="train", streaming=True)

    speeches_list = []
    for i, record in enumerate(dataset):
        if i >= 5000:
            break
        speeches_list.append({
            'date': record.get('date'),
            'author': record.get('author'),
            'country': record.get('country'),
            'text': record.get('text'),
            'clean_text': record.get('clean_text')
        })

    df_speeches = pd.DataFrame(speeches_list)
    output_path = os.path.join('data', 'speeches.csv')
    df_speeches.to_csv(output_path, index=False)
    print(f"Successfully saved texts to {output_path}.")
    return df_speeches


if __name__ == "__main__":
    create_directory_structure()
    fetch_fx_data()
    fetch_speech_corpus()
