import os
import random
import numpy as np
import pandas as pd
import torch
import yfinance as yf


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def construct_distributed_lag_matrix(df, max_lags=6):
    df = df.copy()
    for lag in range(1, max_lags + 1):
        df[f'speech_lag_{lag}'] = df.groupby('pair_id')['semantic_regime'].shift(lag)

    df['returns_lag1'] = df.groupby('pair_id')['returns'].shift(1)

    lag_cols = [f'speech_lag_{lag}' for lag in range(1, max_lags + 1)]
    df = df.dropna(subset=lag_cols + ['returns', 'returns_lag1'])

    lag_vals = [df[f'speech_lag_{i}'] for i in range(1, max_lags + 1)]
    df['almon_term_1'] = sum(lag_vals)
    df['almon_term_2'] = sum((i + 1) * lag_vals[i - 1] for i in range(max_lags))
    return df


def align_and_merge_multi_currency():
    """Compiles a high-density, multi-currency panel matrix across major USD crosses."""
    pairs = {
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "GBPUSD": "GBPUSD=X"
    }

    print("Downloading global multi-currency market data...")
    resampled_pairs = []

    for name, ticker in pairs.items():
        data = yf.download(tickers=ticker, period="2y", interval="1h")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df_pair = data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        df_pair['returns'] = np.log(df_pair['Close'] / df_pair['Close'].shift(1))
        df_pair['pair_id'] = name
        resampled_pairs.append(df_pair)

    df_fx = pd.concat(resampled_pairs).sort_index()
    if df_fx.index.tz is not None:
        df_fx.index = df_fx.index.tz_localize(None)

    speech_path = os.path.join('data', 'speeches_scored.csv')
    df_speeches = pd.read_csv(speech_path, parse_dates=['date']).set_index('date')
    if df_speeches.index.tz is not None:
        df_speeches.index = df_speeches.index.tz_localize(None)
    df_speech_agg = df_speeches.groupby(df_speeches.index)['semantic_score'].mean().to_frame()
    df_speech_agg.index = df_speech_agg.index.ceil('4h')

    df_panel = df_fx.join(df_speech_agg, how='left')
    df_panel['semantic_score'] = df_panel['semantic_score'].fillna(0.0)

    df_panel['semantic_regime'] = df_panel['semantic_score'].replace(0.0, np.nan)
    df_panel['semantic_regime'] = df_panel['semantic_regime'].ewm(span=6, adjust=False).mean().fillna(0.0)

    fred_path = os.path.join('data', 'fred_shocks.csv')
    df_fred = pd.read_csv(fred_path, parse_dates=['date']).set_index('date')
    if df_fred.index.tz is not None:
        df_fred.index = df_fred.index.tz_localize(None)
    df_panel = df_panel.join(df_fred[['econ_surprise']], how='left')
    df_panel['econ_surprise'] = df_panel['econ_surprise'].ffill().fillna(0.0)

    df_panel = construct_distributed_lag_matrix(df_panel, max_lags=6)

    output_path = os.path.join('data', 'currency_panel_h4.csv')
    df_panel.to_csv(output_path)
    print(f"Global currency panel compiled to {output_path} ({len(df_panel)} rows, {df_panel['pair_id'].nunique()} pairs).")
    return df_panel


if __name__ == "__main__":
    align_and_merge_multi_currency()
