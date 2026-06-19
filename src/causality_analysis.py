import os
import random
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def execute_statistical_tests():
    input_path = os.path.join('data', 'merged_h4.csv')
    plot_path = os.path.join('notebooks', 'shape_analysis.png')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    print("\n" + "=" * 70)
    print("      PHASE 2 ECONOMETRIC MATRIX SOUNDNESS CHECK")
    print("=" * 70)
    adf_returns = adfuller(df['returns'])[1]
    print(f"Returns Stationarity (ADF p-value): {adf_returns:.6f}")

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)] + [f'strict_lag_{i}' for i in range(1, 7)]
    features = lag_cols + ['econ_surprise', 'returns_lag1']
    X = df[features]
    X_const = sm.add_constant(X)
    y = df['returns']

    # --- OLS ---
    ols_model = sm.OLS(y, X_const).fit()

    # --- Ridge ---
    ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
    ridge_cv.fit(X, y)
    ridge_coefs = np.concatenate([[ridge_cv.intercept_], ridge_cv.coef_])

    # --- XGBoost (replaces Random Forest) ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.1,
        reg_lambda=0.1,
        reg_alpha=0.01,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X, y)
    xgb_importances = xgb_model.feature_importances_

    print("\n" + "=" * 70)
    print("      OLS vs RIDGE vs XGBoOST")
    print("=" * 70)
    header = f"{'Variable':<20} {'OLS Coef':>10} {'OLS p':>8} {'Ridge Coef':>10} {'XGB Imp':>10}"
    print(header)
    print("-" * 60)
    var_names = ['const'] + features
    for i, name in enumerate(var_names):
        ols_c = ols_model.params.iloc[i]
        ols_p = ols_model.pvalues.iloc[i]
        ridge_c = ridge_coefs[i]
        xgb_imp = xgb_importances[i - 1] if i > 0 else 0
        sig = " **" if ols_p < 0.05 else ""
        print(f"{name:<20} {ols_c:>+10.6f} {ols_p:>8.4f}{sig} {ridge_c:>+10.6f} {xgb_imp:>10.4f}")

    print(f"\nRidge CV selected alpha = {ridge_cv.alpha_:.4f}")
    print(f"OLS In-Sample R2 = {ols_model.rsquared:.5f}")

    y_pred_ridge = ridge_cv.predict(X)
    ss_res = np.sum((y - y_pred_ridge) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ridge_r2 = 1 - ss_res / ss_tot
    print(f"Ridge In-Sample R2 = {ridge_r2:.5f}")

    y_pred_xgb = xgb_model.predict(X)
    ss_res_xgb = np.sum((y - y_pred_xgb) ** 2)
    xgb_r2 = 1 - ss_res_xgb / ss_tot
    print(f"XGBoost In-Sample R2 = {xgb_r2:.5f}")

    # --- Almon PDL ---
    print("\n" + "=" * 70)
    print("      ALMON POLYNOMIAL DISTRIBUTED LAG (PDL)")
    print("=" * 70)
    if 'almon_term_1' in df.columns and 'almon_term_2' in df.columns:
        almon_features = ['almon_term_1', 'almon_term_2', 'econ_surprise', 'returns_lag1']
        X_almon = sm.add_constant(df[almon_features].dropna())
        y_almon = df.loc[X_almon.index, 'returns']
        almon_model = sm.OLS(y_almon, X_almon).fit()
        print(almon_model.summary().tables[1])
        print(f"Almon PDL In-Sample R2 = {almon_model.rsquared:.5f}")
    else:
        print("Almon terms not found.")

    # --- Granger ---
    print("\n" + "=" * 70)
    print("      MULTI-LAG GRANGER CAUSALITY TEST")
    print("=" * 70)
    df_granger = df[['returns', 'semantic_regime']]
    grangercausalitytests(df_granger, maxlag=6, verbose=True)

    # --- Four-panel plot ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    ax1.plot(df.index, df['cumulative_returns'], color='tab:blue', alpha=0.7, label='Returns Profile')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['semantic_regime'], color='tab:orange', linestyle='--', alpha=0.5, label='NLP Decay')
    ax1.set_title("EUR/USD Returns vs. Speech Semantic Decay")

    ax2.plot(ols_model.resid, color='purple', alpha=0.4, label='OLS Residuals')
    ridge_resid = y - y_pred_ridge
    ax2.plot(ridge_resid, color='green', alpha=0.2, label='Ridge Residuals')
    ax2.axhline(0, color='black', linestyle=':')
    ax2.set_title("Residual Error Distribution (purple=OLS, green=Ridge)")
    ax2.legend()

    ols_coefs = ols_model.params[lag_cols]
    ridge_coefs_lag = ridge_cv.coef_[:12]
    x = np.arange(len(lag_cols))
    width = 0.35
    ax3.bar(x - width / 2, ols_coefs, width, color='seagreen', alpha=0.8, edgecolor='black', label='OLS')
    ax3.bar(x + width / 2, ridge_coefs_lag, width, color='darkorange', alpha=0.8, edgecolor='black',
            label=f'Ridge (a={ridge_cv.alpha_:.2f})')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title("Distributed Lags: OLS vs Ridge")
    ax3.set_ylabel("Beta Strength")
    ax3.set_xticks(x)
    ax3.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
                         'S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
    ax3.legend()

    xgb_imps = xgb_importances[:12]
    ax4.bar(x, xgb_imps, width=0.5, color='crimson', alpha=0.7, edgecolor='black')
    ax4.set_title("XGBoost Feature Importances (Broad + Strict Lags)")
    ax4.set_ylabel("Importance")
    ax4.set_xticks(x)
    ax4.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5', 'L6',
                         'S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"\nDiagnostic Dashboard saved to: {plot_path}")


if __name__ == "__main__":
    execute_statistical_tests()
