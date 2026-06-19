import os
import random
import numpy as np
import pandas as pd
import torch


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
        df[f'speech_lag_{lag}'] = df['semantic_regime'].shift(lag)
        df[f'strict_lag_{lag}'] = df['strict_regime'].shift(lag)

    df['returns_lag1'] = df['returns'].shift(1)

    all_lags = [f'speech_lag_{lag}' for lag in range(1, max_lags + 1)]
    all_lags += [f'strict_lag_{lag}' for lag in range(1, max_lags + 1)]
    df = df.dropna(subset=all_lags + ['returns', 'returns_lag1'])

    broad_vals = [df[f'speech_lag_{i}'] for i in range(1, max_lags + 1)]
    strict_vals = [df[f'strict_lag_{i}'] for i in range(1, max_lags + 1)]
    df['almon_term_1'] = sum(broad_vals)
    df['almon_term_2'] = sum((i + 1) * broad_vals[i - 1] for i in range(max_lags))
    df['strict_almon_1'] = sum(strict_vals)
    df['strict_almon_2'] = sum((i + 1) * strict_vals[i - 1] for i in range(max_lags))
    return df


def align_and_merge_datasets():
    price_path = os.path.join('data', 'fx_prices.csv')
    speech_path = os.path.join('data', 'speeches_scored.csv')
    fred_path = os.path.join('data', 'fred_shocks.csv')
    output_path = os.path.join('data', 'merged_h4.csv')

    df_prices = pd.read_csv(price_path)
    time_col = 'Datetime' if 'Datetime' in df_prices.columns else 'Date'
    df_prices[time_col] = pd.to_datetime(df_prices[time_col], errors='coerce', utc=True).dt.tz_localize(None)
    df_prices = df_prices.dropna(subset=[time_col]).set_index(time_col)

    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_h4 = df_prices.resample('4h').agg(ohlc_dict).dropna(subset=['Close'])
    df_h4['returns'] = np.log(df_h4['Close'] / df_h4['Close'].shift(1))

    df_speeches = pd.read_csv(speech_path, parse_dates=['date'])
    df_speeches['date'] = pd.to_datetime(df_speeches['date'], errors='coerce', utc=True).dt.tz_localize(None)
    df_speech_agg = df_speeches.groupby('date')[['semantic_score', 'strict_score']].mean()

    df_merged = df_h4.join(df_speech_agg, how='left')
    df_merged['semantic_score'] = df_merged['semantic_score'].fillna(0.0)
    df_merged['strict_score'] = df_merged['strict_score'].fillna(0.0)

    df_merged['semantic_regime'] = df_merged['semantic_score'].replace(0.0, np.nan)
    df_merged['semantic_regime'] = df_merged['semantic_regime'].ewm(span=6, adjust=False).mean().fillna(0.0)

    df_merged['strict_regime'] = df_merged['strict_score'].replace(0.0, np.nan)
    df_merged['strict_regime'] = df_merged['strict_regime'].ewm(span=6, adjust=False).mean().fillna(0.0)

    df_fred = pd.read_csv(fred_path, parse_dates=['date'])
    df_fred['date'] = pd.to_datetime(df_fred['date'], utc=True).dt.tz_localize(None)
    df_fred = df_fred.set_index('date')
    df_merged = df_merged.join(df_fred[['econ_surprise']], how='left')
    df_merged['econ_surprise'] = df_merged['econ_surprise'].ffill().fillna(0.0)

    df_merged = construct_distributed_lag_matrix(df_merged, max_lags=6)

    df_merged.to_csv(output_path)
    print(f"Phase 2 multi-lag baseline compiled to {output_path} ({len(df_merged)} entries).")
    return df_merged


if __name__ == "__main__":
    align_and_merge_datasets()
