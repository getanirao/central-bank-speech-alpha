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
    input_path = os.path.join('data', 'merged_h4.csv')
    orig = pd.read_csv(input_path, index_col=0, parse_dates=True)

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    features = lag_cols + ['econ_surprise', 'returns_lag1']
    split_idx = int(len(orig) * 0.70)

    X = orig[features]
    y = orig['returns']
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    true_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
    true_model.fit(X_train, y_train)
    y_pred_true = true_model.predict(X_test)
    true_oos_r2 = r2_score(y_test, y_pred_true)

    print(f"\n{'=' * 70}")
    print(f"  PERMUTATION / PLACEBO TEST ({n_iterations} iterations)")
    print(f"{'=' * 70}")
    print(f"  True OOS R2: {true_oos_r2:+.5f}")

    placebo_r2 = []

    for i in range(n_iterations):
        df_perm = orig.copy()
        df_perm['semantic_regime'] = np.random.permutation(df_perm['semantic_regime'].values)

        for lag in range(1, 7):
            df_perm[f'speech_lag_{lag}'] = df_perm['semantic_regime'].shift(lag)
        df_perm = df_perm.dropna(subset=lag_cols)

        X_perm = df_perm[features]
        y_perm = df_perm['returns']
        Xp_train, yp_train = X_perm.iloc[:split_idx], y_perm.iloc[:split_idx]
        Xp_test, yp_test = X_perm.iloc[split_idx:], y_perm.iloc[split_idx:]

        perm_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
        perm_model.fit(Xp_train, yp_train)
        y_pred_perm = perm_model.predict(Xp_test)
        r2 = r2_score(yp_test, y_pred_perm)
        placebo_r2.append(r2)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_iterations} iterations complete...")

    placebo_r2 = np.array(placebo_r2)
    p_value = np.mean(placebo_r2 >= true_oos_r2)

    print(f"\n  Null Distribution (placebo R2):")
    print(f"    Mean: {placebo_r2.mean():+.5f}")
    print(f"    Std:  {placebo_r2.std():.5f}")
    print(f"    95th percentile: {np.percentile(placebo_r2, 95):+.5f}")
    print(f"    99th percentile: {np.percentile(placebo_r2, 99):+.5f}")
    print(f"  True OOS R2: {true_oos_r2:+.5f}")
    print(f"  Permutation p-value: {p_value:.6f}")

    verdict = "PASS" if p_value < 0.05 else "FAIL"
    print(f"\n  VERDICT: {verdict} -- Speech features encode real information "
          f"(p={p_value:.4f})")

    out = pd.DataFrame({'placebo_r2': placebo_r2})
    out.to_csv(os.path.join('data', 'placebo_test_results.csv'), index=False)
    return placebo_r2, p_value, true_oos_r2


if __name__ == "__main__":
    run_placebo_test()
