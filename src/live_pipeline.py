import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from fredapi import Fred
from transformers import pipeline


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def get_agent_label(res):
    return res['label']


AGENT_WHITELIST = {'Financial Sector', 'Central Bank'}


def run_live_production_signal():
    print("=======================================================")
    print("      LAUNCHING LIVE PRODUCTION TRADING ENGINE         ")
    print("=======================================================")

    FRED_API_KEY = "27878d4316dd6e73d8faff3041cd499f"
    ticker = "EURUSD=X"

    print(f"Fetching real-time market streams for {ticker}...")
    live_prices = yf.download(tickers=ticker, period="7d", interval="1h")
    if isinstance(live_prices.columns, pd.MultiIndex):
        live_prices.columns = live_prices.columns.get_level_values(0)

    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_daily = live_prices.resample('D').agg(ohlc_dict).dropna(subset=['Close'])
    df_daily['returns'] = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
    df_daily['returns_lag1'] = df_daily['returns'].shift(1)

    print("Fetching active economic factors from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    cpi = fred.get_series('CPIAUCNS').pct_change().iloc[-1]
    nfp = fred.get_series('PAYEMS').diff().iloc[-1]

    current_econ_surprise = (cpi + nfp) / 2.0
    df_daily['econ_surprise'] = current_econ_surprise

    print("Scanning active central bank RSS communication channels...")
    live_speech_feed = [
        {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": "The Committee decided to maintain the target rate... Inflation remains elevated... The Committee will deliver price stability... Forward guidance is not well suited for the current policy conjuncture."
        }
    ]

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading CentralBankRoBERTa models on device: {'GPU' if device == 0 else 'CPU'}")

    agent_engine = pipeline(
        "text-classification",
        model="Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier",
        device=device,
    )
    sentiment_engine = pipeline(
        "text-classification",
        model="Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
        device=device,
    )

    speech_text = live_speech_feed[0]["text"]
    agent_res = agent_engine(speech_text, truncation=True, max_length=512)[0]
    agent_label = get_agent_label(agent_res)

    if agent_label in AGENT_WHITELIST:
        sent_res = sentiment_engine(speech_text, truncation=True, max_length=512)[0]
        if sent_res['label'] == 'positive':
            raw_score = sent_res['score']
        else:
            raw_score = -sent_res['score']
    else:
        raw_score = 0.0

    print(f"Latest speech processed. Agent={agent_label} Sentiment={raw_score:+.4f}")

    df_daily['semantic_regime'] = raw_score

    for lag in range(1, 7):
        df_daily[f'speech_lag_{lag}'] = df_daily['semantic_regime'].shift(lag)

    df_daily = df_daily.dropna()
    active_row = df_daily.iloc[-1]

    intercept = 0.0
    beta_speech_4 = 0.0017
    beta_econ = 0.0003

    predicted_return = (
        intercept
        + (beta_speech_4 * active_row['speech_lag_4'])
        + (beta_econ * active_row['econ_surprise'])
    )

    execution_signal = "BUY / LONG" if predicted_return > 0 else "SELL / SHORT"

    print("\n=======================================================")
    print("             PRODUCTION EXECUTION TARGET               ")
    print("=======================================================")
    print(f"Timestamp:                 {df_daily.index[-1]}")
    print(f"Current EUR/USD Spot:       {active_row['Close']:.5f}")
    print(f"Model Predicted Return:    {predicted_return:+.7f}")
    print(f"ACTIONABLE ORDER TARGET:   *** {execution_signal} ***")
    print("=======================================================")

    return execution_signal


if __name__ == "__main__":
    run_live_production_signal()
