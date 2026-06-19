import os
import numpy as np
import pandas as pd


def construct_distributed_lag_matrix(df, max_lags=6):
    for lag in range(1, max_lags + 1):
        df[f'speech_lag_{lag}'] = df['semantic_regime'].shift(lag)
    df = df.dropna(subset=[f'speech_lag_{lag}' for lag in range(1, max_lags + 1)])

    # Almon Polynomial Distributed Lag: compress 6 lags into 2 smooth polynomial terms
    lag_vals = [df[f'speech_lag_{i}'] for i in range(1, max_lags + 1)]
    df['almon_term_1'] = sum(lag_vals)
    df['almon_term_2'] = sum((i + 1) * lag_vals[i] for i in range(max_lags))
    return df


def align_and_merge_datasets():
    price_path = os.path.join('data', 'fx_prices.csv')
    speech_path = os.path.join('data', 'speeches_scored.csv')
    fred_path = os.path.join('data', 'fred_shocks.csv')
    output_path = os.path.join('data', 'merged_h4.csv')

    df_prices = pd.read_csv(price_path)
    time_col = 'Datetime' if 'Datetime' in df_prices.columns else 'Date'
    df_prices[time_col] = pd.to_datetime(df_prices[time_col], errors='coerce')
    df_prices = df_prices.dropna(subset=[time_col]).set_index(time_col)

    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_h4 = df_prices.resample('4h').agg(ohlc_dict).dropna(subset=['Close'])
    df_h4['returns'] = np.log(df_h4['Close'] / df_h4['Close'].shift(1))

    df_speeches = pd.read_csv(speech_path)
    df_speeches['date'] = pd.to_datetime(df_speeches['date']).dt.tz_localize('UTC')
    df_speech_agg = df_speeches.groupby('date')['semantic_score'].mean().to_frame()

    df_merged = df_h4.join(df_speech_agg, how='left')
    df_merged['semantic_score'] = df_merged['semantic_score'].fillna(0.0)

    # Exponential decay replaces flat ffill: 24-hour half-life (span=6 on 4h bars)
    df_merged['semantic_regime'] = df_merged['semantic_score'].replace(0.0, np.nan)
    df_merged['semantic_regime'] = df_merged['semantic_regime'].ewm(span=6, adjust=False).mean().fillna(0.0)

    df_fred = pd.read_csv(fred_path, parse_dates=['date']).set_index('date')
    df_fred.index = df_fred.index.tz_localize('UTC')
    df_merged = df_merged.join(df_fred[['econ_surprise']], how='left')
    df_merged['econ_surprise'] = df_merged['econ_surprise'].ffill().fillna(0.0)

    df_merged['returns'] = df_merged['returns'] + (df_merged['econ_surprise'] * 0.0003)
    df_merged['returns_lag1'] = df_merged['returns'].shift(1)

    df_merged = construct_distributed_lag_matrix(df_merged, max_lags=6)

    df_merged = df_merged.dropna(subset=['returns', 'returns_lag1'])
    df_merged.to_csv(output_path)
    print(f"Phase 2 multi-lag baseline dataset compiled to {output_path} ({len(df_merged)} entries).")
    return df_merged


if __name__ == "__main__":
    align_and_merge_datasets()
