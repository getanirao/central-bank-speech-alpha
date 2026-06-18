import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import matplotlib.pyplot as plt


def execute_statistical_tests():
    input_path = os.path.join('data', 'merged_h4.csv')
    plot_path = os.path.join('notebooks', 'shape_analysis.png')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    print("\n=======================================================")
    print("      PHASE 2 ECONOMETRIC MATRIX SOUNDNESS CHECK      ")
    print("=======================================================")
    adf_returns = adfuller(df['returns'])[1]
    print(f"Returns Line Stationarity ADF p-value: {adf_returns:.6f}")

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    X = df[lag_cols + ['econ_surprise', 'returns_lag1']]
    X = sm.add_constant(X)
    y = df['returns']

    ols_model = sm.OLS(y, X).fit()
    print(ols_model.summary())

    print("\n=======================================================")
    print("      PHASE 2 MULTI-LAG GRANGER CAUSALITY TEST         ")
    print("=======================================================")
    df_granger = df[['returns', 'semantic_regime']]
    grangercausalitytests(df_granger, maxlag=6, verbose=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    ax1.plot(df.index, df['cumulative_returns'], color='tab:blue', alpha=0.7, label='Returns Profile')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['semantic_regime'], color='tab:orange', linestyle='--', alpha=0.5, label='NLP Regime')
    ax1.set_title("EUR/USD Returns vs. Speech Semantic Adjustments")

    ax2.plot(ols_model.resid, color='purple', alpha=0.4, label='Residual Noise Matrix')
    ax2.axhline(0, color='black', linestyle=':')
    ax2.set_title("Econometric Residual Error Distribution Profile")
    ax2.legend()

    coef_vals = ols_model.params[lag_cols]
    coef_errors = ols_model.bse[lag_cols]

    ax3.bar(lag_cols, coef_vals, yerr=coef_errors * 1.96, color='seagreen', capsize=5, alpha=0.8, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title("Almon Distributed Coefficient Impact Layout (95% Confidence Boundaries)")
    ax3.set_ylabel("Beta Strength Multiplier")
    ax3.set_xticklabels(['Lag 1 (4h)', 'Lag 2 (8h)', 'Lag 3 (12h)', 'Lag 4 (16h)', 'Lag 5 (20h)', 'Lag 6 (24h)'])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"\nPhase 3 Diagnostic Dashboard compiled successfully to: {plot_path}")


if __name__ == "__main__":
    execute_statistical_tests()
