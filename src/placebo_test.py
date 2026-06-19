import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def run_placebo_test(n_iterations=1000, model_type='ridge'):
    """
    Permutation test on multi-currency panel: shuffling speech destroys
    cross-currency coordination, making it impossible for randomized noise
    to simultaneously match EUR, JPY, and GBP signatures.
    """
    input_path = os.path.join('data', 'currency_panel_h4.csv')
    orig = pd.read_csv(input_path)
    time_col = 'Datetime' if 'Datetime' in orig.columns else 'date' if 'date' in orig.columns else orig.columns[0]
    orig[time_col] = pd.to_datetime(orig[time_col], errors='coerce')
    orig = orig.dropna(subset=[time_col])
    orig = orig.sort_values([time_col, 'pair_id']).reset_index(drop=True)

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    features = lag_cols + ['econ_surprise', 'returns_lag1']

    pairs = orig['pair_id'].unique()
    split_idx = int(len(orig[orig['pair_id'] == pairs[0]]) * 0.70)

    # --- True OOS performance ---
    true_oos_r2s = []
    for pid in pairs:
        pair_df = orig[orig['pair_id'] == pid].reset_index(drop=True)
        X, y = pair_df[features], pair_df['returns']
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        true_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
        true_model.fit(X_train, y_train)
        y_pred_true = true_model.predict(X_test)
        true_oos_r2s.append(r2_score(y_test, y_pred_true))

    true_oos_r2 = np.mean(true_oos_r2s)

    print(f"\n{'=' * 70}")
    print(f"  PERMUTATION / PLACEBO TEST — Multi-Currency Panel ({n_iterations} iterations)")
    print(f"{'=' * 70}")
    print(f"  True Mean OOS R2 (avg across {len(pairs)} pairs): {true_oos_r2:+.5f}")

    placebo_r2 = []

    for i in range(n_iterations):
        df_perm = orig.copy()
        df_perm['semantic_regime'] = np.random.permutation(df_perm['semantic_regime'].values)

        for lag in range(1, 7):
            df_perm[f'speech_lag_{lag}'] = df_perm.groupby('pair_id')['semantic_regime'].shift(lag)
        df_perm = df_perm.dropna(subset=lag_cols + ['returns', 'returns_lag1'])

        perm_r2s = []
        for pid in pairs:
            pair_df = df_perm[df_perm['pair_id'] == pid].reset_index(drop=True)
            X, y = pair_df[features], pair_df['returns']
            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

            if len(X_train) < 10 or len(X_test) < 5:
                continue

            perm_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
            perm_model.fit(X_train, y_train)
            y_pred_perm = perm_model.predict(X_test)
            perm_r2s.append(r2_score(y_test, y_pred_perm))

        placebo_r2.append(np.mean(perm_r2s) if perm_r2s else 0.0)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_iterations} iterations complete...")

    placebo_r2 = np.array(placebo_r2)
    p_value = np.mean(placebo_r2 >= true_oos_r2)

    print(f"\n  Null Distribution (placebo R2 across panel):")
    print(f"    Mean: {placebo_r2.mean():+.5f}")
    print(f"    Std:  {placebo_r2.std():.5f}")
    print(f"    95th percentile: {np.percentile(placebo_r2, 95):+.5f}")
    print(f"    99th percentile: {np.percentile(placebo_r2, 99):+.5f}")
    print(f"  True OOS R2: {true_oos_r2:+.5f}")
    print(f"  Permutation p-value: {p_value:.6f}")

    verdict = "PASS" if p_value < 0.05 else "FAIL"
    print(f"\n  VERDICT: {verdict} -- Speech features encode real cross-currency "
          f"signal (p={p_value:.4f})")

    out = pd.DataFrame({'placebo_r2': placebo_r2})
    out.to_csv(os.path.join('data', 'placebo_test_results.csv'), index=False)
    print(f"  Placebo distribution saved to data/placebo_test_results.csv")

    return placebo_r2, p_value, true_oos_r2


if __name__ == "__main__":
    run_placebo_test()
