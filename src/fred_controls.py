import os
import random
import numpy as np
import pandas as pd
import torch
from fredapi import Fred


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def fetch_and_save_fred_shocks():
    """Fetches real macro variables from FRED or executes safe simulation fallback."""
    output_path = os.path.join('data', 'fred_shocks.csv')
    api_key = "27878d4316dd6e73d8faff3041cd499f"

    if not os.path.exists('data'):
        os.makedirs('data')

    try:
        print("Connecting to FRED API interface...")
        fred = Fred(api_key=api_key)

        cpi = fred.get_series('CPIAUCNS')
        nfp = fred.get_series('PAYEMS')

        cpi_shock = cpi.pct_change().dropna()
        nfp_shock = nfp.diff().dropna()

        df_cpi = pd.DataFrame({'cpi_surprise': cpi_shock})
        df_nfp = pd.DataFrame({'nfp_surprise': nfp_shock})

        df_fred = df_cpi.join(df_nfp, how='outer').fillna(0.0)

        df_fred['cpi_z'] = (
            df_fred['cpi_surprise'] - df_fred['cpi_surprise'].rolling(12, min_periods=1).mean()
        ) / df_fred['cpi_surprise'].rolling(12, min_periods=1).std().fillna(1.0)

        df_fred['nfp_z'] = (
            df_fred['nfp_surprise'] - df_fred['nfp_surprise'].rolling(12, min_periods=1).mean()
        ) / df_fred['nfp_surprise'].rolling(12, min_periods=1).std().fillna(1.0)

        df_fred['econ_surprise'] = (df_fred['cpi_z'] + df_fred['nfp_z']) / 2.0
        df_fred = df_fred.dropna()

        df_fred.index.name = 'date'
        df_fred[['cpi_surprise', 'nfp_surprise', 'econ_surprise']].to_csv(output_path)
        print(f"Real FRED macro economic shocks saved to {output_path}.")

    except Exception as e:
        print(f"FRED API data extraction failed: {e}. Executing fallback simulation engine...")
        dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="ME")
        enforce_strict_reproducibility()

        df_fallback = pd.DataFrame(index=dates)
        df_fallback.index.name = 'date'
        df_fallback['cpi_surprise'] = np.random.normal(0, 1, len(df_fallback))
        df_fallback['nfp_surprise'] = np.random.normal(0, 1, len(df_fallback))
        df_fallback['econ_surprise'] = (df_fallback['cpi_surprise'] + df_fallback['nfp_surprise']) / 2.0
        df_fallback.to_csv(output_path)
        print(f"Fallback simulated framework saved safely to {output_path}.")


if __name__ == "__main__":
    fetch_and_save_fred_shocks()
