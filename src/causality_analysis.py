import os
import random
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from linearmodels.panel import PooledOLS
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
    input_path = os.path.join('data', 'currency_panel_h4.csv')
    plot_path = os.path.join('notebooks', 'shape_analysis.png')
    df = pd.read_csv(input_path)

    time_col = 'Datetime' if 'Datetime' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])

    df = df.set_index(['pair_id', time_col]).sort_index()

    print("\n" + "=" * 70)
    print("      MULTI-CURRENCY PANEL ECONOMETRIC SOUNDNESS CHECK")
    print("=" * 70)

    all_returns = df['returns'].values
    adf_panel = adfuller(all_returns)[1]
    print(f"Panel Returns Stationarity (ADF p-value): {adf_panel:.6f}")

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    features = lag_cols + ['econ_surprise', 'returns_lag1']
    X = sm.add_constant(df[features])
    y = df['returns']

    # --- Pooled OLS Panel Regression ---
    print("\n" + "=" * 70)
    print("      GLOBAL MULTI-CURRENCY PANEL REGRESSION (Pooled OLS)")
    print("=" * 70)
    panel_model = PooledOLS(y, X).fit()
    print(panel_model)

    # --- Ridge ---
    ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
    ridge_cv.fit(df[features], y)
    ridge_coefs = np.concatenate([[ridge_cv.intercept_], ridge_cv.coef_])

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(df[features], y)
    rf_importances = rf.feature_importances_

    print("\n" + "=" * 70)
    print("      OLS vs RIDGE vs RANDOM FOREST (Panel)")
    print("=" * 70)
    header = f"{'Variable':<20} {'OLS Coef':>10} {'OLS p':>8} {'Ridge Coef':>10} {'RF Import':>10}"
    print(header)
    print("-" * 60)
    var_names = ['const'] + features
    for i, name in enumerate(var_names):
        ols_c = panel_model.params.iloc[i]
        ols_p = panel_model.pvalues.iloc[i]
        ridge_c = ridge_coefs[i]
        rf_imp = rf_importances[i - 1] if i > 0 else 0
        sig = " **" if ols_p < 0.05 else ""
        print(f"{name:<20} {ols_c:>+10.6f} {ols_p:>8.4f}{sig} {ridge_c:>+10.6f} {rf_imp:>10.4f}")

    print(f"\nRidge CV selected alpha = {ridge_cv.alpha_:.4f}")
    print(f"Pooled OLS In-Sample R2 = {panel_model.rsquared:.5f}")

    y_pred_ridge = ridge_cv.predict(df[features])
    ss_res = np.sum((y - y_pred_ridge) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ridge_r2 = 1 - ss_res / ss_tot
    print(f"Ridge In-Sample R2 = {ridge_r2:.5f}")

    y_pred_rf = rf.predict(df[features])
    ss_res_rf = np.sum((y - y_pred_rf) ** 2)
    rf_r2 = 1 - ss_res_rf / ss_tot
    print(f"Random Forest In-Sample R2 = {rf_r2:.5f}")

    # --- Almon PDL ---
    print("\n" + "=" * 70)
    print("      ALMON POLYNOMIAL DISTRIBUTED LAG (PDL) — Panel")
    print("=" * 70)
    if 'almon_term_1' in df.columns and 'almon_term_2' in df.columns:
        almon_features = ['almon_term_1', 'almon_term_2', 'econ_surprise', 'returns_lag1']
        X_almon = sm.add_constant(df[almon_features].dropna())
        y_almon = df.loc[X_almon.index, 'returns']
        almon_model = PooledOLS(y_almon, X_almon).fit()
        print(almon_model)
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

    for pid in df.index.get_level_values(0).unique():
        sub = df.xs(pid, level=0)
        cum_ret = (1 + sub['returns']).cumprod() - 1
        ax1.plot(sub.index, cum_ret.values, alpha=0.5, label=pid)
    ax1.set_title("Cumulative Returns by Currency Pair")
    ax1.legend()

    ax2.plot(range(len(panel_model.resids)), panel_model.resids.values, color='purple', alpha=0.4, label='Pooled OLS Residuals')
    ax2.axhline(0, color='black', linestyle=':')
    ax2.set_title("Residual Error Distribution")
    ax2.legend()

    pair_list = df.index.get_level_values(0).unique()
    x = np.arange(len(lag_cols))
    width = 0.25
    for j, pid in enumerate(pair_list):
        sub = df.xs(pid, level=0)
        X_pid = sm.add_constant(sub[features])
        ols_pid = sm.OLS(sub['returns'], X_pid).fit()
        ax3.bar(x + width * (j - 1), ols_pid.params[lag_cols], width, alpha=0.7, label=pid)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title("Lag Coefficients by Currency Pair (Per-Pair OLS)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(['L1\n(4h)', 'L2\n(8h)', 'L3\n(12h)', 'L4\n(16h)', 'L5\n(20h)', 'L6\n(24h)'])
    ax3.legend()

    rf_imps = rf_importances[:6]
    ax4.bar(x, rf_imps, width=0.5, color='crimson', alpha=0.7, edgecolor='black')
    ax4.set_title("Random Forest Feature Importances (Panel)")
    ax4.set_ylabel("Importance")
    ax4.set_xticks(x)
    ax4.set_xticklabels(['L1\n(4h)', 'L2\n(8h)', 'L3\n(12h)', 'L4\n(16h)', 'L5\n(20h)', 'L6\n(24h)'])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"\nDiagnostic Dashboard saved to: {plot_path}")


if __name__ == "__main__":
    execute_statistical_tests()
