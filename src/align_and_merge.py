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


def build_country_lags(df, prefix, max_lags=6):
    """Create lags + Almon PDL terms for a country-specific score column."""
    for lag in range(1, max_lags + 1):
        df[f'{prefix}_lag_{lag}'] = df[f'{prefix}_score'].shift(lag)
    vals = [df[f'{prefix}_lag_{i}'] for i in range(1, max_lags + 1)]
    df[f'{prefix}_almon_1'] = sum(vals)
    df[f'{prefix}_almon_2'] = sum((i + 1) * vals[i - 1] for i in range(max_lags))
    df[f'{prefix}_macro_interact'] = df[f'{prefix}_score'] * df['econ_surprise']
    return df


def align_and_merge_datasets():
    price_path = os.path.join('data', 'fx_prices.csv')
    speech_path = os.path.join('data', 'speeches_scored.csv')
    fred_path = os.path.join('data', 'fred_shocks.csv')
    output_path = os.path.join('data', 'merged_daily.csv')

    # Daily price data (already daily from yfinance)
    df_prices = pd.read_csv(price_path)
    time_col = 'Datetime' if 'Datetime' in df_prices.columns else 'Date'
    df_prices[time_col] = pd.to_datetime(df_prices[time_col], errors='coerce', utc=True).dt.tz_localize(None)
    df_prices = df_prices.dropna(subset=[time_col]).set_index(time_col)

    # For daily source data, just use as-is; no resampling needed
    df_daily = df_prices.copy()
    df_daily['returns'] = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
    df_daily['hv_20'] = df_daily['returns'].rolling(20).std()

    # Load scored speeches and split by country (non-filled)
    df_speeches = pd.read_csv(speech_path, parse_dates=['date'])
    df_speeches['date'] = pd.to_datetime(df_speeches['date'], errors='coerce', utc=True).dt.tz_localize(None)

    fed = df_speeches[df_speeches['country'] == 'United States'].copy()
    ecb = df_speeches[df_speeches['country'] == 'Euro area'].copy()

    # Use strict_score (requires >=3 policy keywords per speech)
    fed['score'] = fed['strict_score']
    ecb['score'] = ecb['strict_score']

    fed_agg = fed.groupby('date')['score'].mean().rename('fed_score')
    ecb_agg = ecb.groupby('date')['score'].mean().rename('ecb_score')

    # Non-filled join — zeros on days without speeches
    df_merged = df_daily.join(fed_agg, how='left')
    df_merged = df_merged.join(ecb_agg, how='left')
    df_merged['fed_score'] = df_merged['fed_score'].fillna(0.0)
    df_merged['ecb_score'] = df_merged['ecb_score'].fillna(0.0)

    # FRED macro shocks
    df_fred = pd.read_csv(fred_path, parse_dates=['date'])
    df_fred['date'] = pd.to_datetime(df_fred['date'], utc=True).dt.tz_localize(None)
    df_fred = df_fred.set_index('date')
    df_merged = df_merged.join(df_fred[['econ_surprise']], how='left')
    df_merged['econ_surprise'] = df_merged['econ_surprise'].ffill().fillna(0.0)

    # Build country-specific lags + Almon PDLs
    df_merged = build_country_lags(df_merged, 'fed', max_lags=6)
    df_merged = build_country_lags(df_merged, 'ecb', max_lags=6)

    df_merged['returns_lag1'] = df_merged['returns'].shift(1)

    # Drop rows missing any required feature
    lag_cols = [f'{p}_lag_{i}' for p in ['fed', 'ecb'] for i in range(1, 7)]
    required = lag_cols + ['returns', 'returns_lag1', 'hv_20']
    df_merged = df_merged.dropna(subset=required)

    df_merged.to_csv(output_path)
    print(f"Daily merged dataset saved to {output_path} ({len(df_merged)} entries).")
    return df_merged


if __name__ == "__main__":
    align_and_merge_datasets()
