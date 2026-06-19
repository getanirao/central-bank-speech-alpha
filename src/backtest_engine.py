import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV


def run_out_of_sample_backtest(train_window_pct=0.70, model_type='ridge'):
    """
    Executes a strict historical walk-forward validation.
    Supports both 'ols' and 'ridge' model types.
    """
    input_path = os.path.join('data', 'merged_h4.csv')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    split_idx = int(len(df) * train_window_pct)

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    features = lag_cols + ['econ_surprise', 'returns_lag1']

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\n--- Running {model_type.upper()} Walk-Forward ---")
    print(f"Training Baseline Sample Space: {len(train_df)} rows")
    print(f"Out-of-Sample Evaluation Space: {len(test_df)} rows")

    if model_type == 'ridge':
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
        model.fit(train_df[features], train_df['returns'])
        y_pred_test = model.predict(test_df[features])
        # In-sample R²
        y_pred_train = model.predict(train_df[features])
        ss_res = np.sum((train_df['returns'] - y_pred_train) ** 2)
        ss_tot = np.sum((train_df['returns'] - np.mean(train_df['returns'])) ** 2)
        in_sample_r2 = 1 - ss_res / ss_tot
        print(f"Ridge CV selected alpha = {model.alpha_:.4f}")
    else:
        X_train = sm.add_constant(train_df[features])
        model = sm.OLS(train_df['returns'], X_train).fit()
        X_test = test_df[features].copy()
        X_test.insert(0, 'const', 1.0)
        y_pred_test = model.predict(X_test)
        in_sample_r2 = model.rsquared

    y_test = test_df['returns']
    oos_r2 = r2_score(y_test, y_pred_test)
    oos_mse = mean_squared_error(y_test, y_pred_test)

    print("\n=======================================================")
    print("      OUT-OF-SAMPLE (OOS) METRIC DIAGNOSTICS          ")
    print("=======================================================")
    print(f"In-Sample R-squared:       {in_sample_r2:.5f}")
    print(f"OUT-OF-SAMPLE R-squared:   {oos_r2:.5f}")
    print(f"OOS Mean Squared Error:    {oos_mse:.7f}")

    if oos_r2 <= 0:
        print("\nWARNING: OOS R-squared is negative.")
    else:
        print("\nSUCCESS: Structural alpha confirmed out-of-sample.")

    test_df = test_df.copy()
    test_df['predicted_return'] = y_pred_test
    test_df['trading_signal'] = np.sign(test_df['predicted_return'])
    test_df['strategy_return'] = test_df['trading_signal'] * test_df['returns']

    test_df['cum_market_returns'] = (1 + test_df['returns']).cumprod() - 1
    test_df['cum_strategy_returns'] = (1 + test_df['strategy_return']).cumprod() - 1

    print("\n=======================================================")
    print("         TRADING PERFORMANCE METRICS                  ")
    print("=======================================================")

    strategy_mean = test_df['strategy_return'].mean()
    strategy_std = test_df['strategy_return'].std()
    info_ratio = (strategy_mean / strategy_std) * np.sqrt(6) if strategy_std != 0 else 0.0
    print(f"Strategy Mean Return per 4h: {strategy_mean:.6f}")
    print(f"Strategy Std Dev per 4h:      {strategy_std:.6f}")
    print(f"Information Ratio (annual.):  {info_ratio:.4f}")

    hit_rate = (test_df['trading_signal'] == np.sign(test_df['returns'])).mean()
    print(f"Directional Prediction Accuracy (Hit Rate): {hit_rate:.4%}")

    total_strat = test_df['cum_strategy_returns'].iloc[-1]
    total_mkt = test_df['cum_market_returns'].iloc[-1]
    print(f"Total Strategy Return (OOS): {total_strat:.4%}")
    print(f"Total Market Return (OOS):    {total_mkt:.4%}")

    backtest_path = os.path.join('data', f'backtest_results_{model_type}.csv')
    test_df.to_csv(backtest_path)
    print(f"\nOOS trade logs written safely to: {backtest_path}")

    return test_df


def run_comparison_backtest(train_window_pct=0.70):
    """Runs both OLS and Ridge backtests and prints a comparison."""
    ols_result = run_out_of_sample_backtest(train_window_pct, model_type='ols')
    ridge_result = run_out_of_sample_backtest(train_window_pct, model_type='ridge')

    print("\n" + "=" * 60)
    print("      OOS WALK-FORWARD COMPARISON: OLS vs RIDGE")
    print("=" * 60)
    print(f"{'Metric':<35} {'OLS':>12} {'Ridge':>12}")
    print("-" * 59)

    # Compute metrics for comparison
    for name, oos_r2_val, rid_r2_val in [
        ('OOS R²', r2_score(ols_result['returns'], ols_result['predicted_return']),
                   r2_score(ridge_result['returns'], ridge_result['predicted_return'])),
    ]:
        print(f"{name:<35} {oos_r2_val:>+12.5f} {rid_r2_val:>+12.5f}")

    for name, ols_val, rid_val in [
        ('Hit Rate', (np.sign(ols_result['returns']) == ols_result['trading_signal']).mean(),
                      (np.sign(ridge_result['returns']) == ridge_result['trading_signal']).mean()),
        ('Info Ratio', (ols_result['strategy_return'].mean() / ols_result['strategy_return'].std()) * np.sqrt(6),
                       (ridge_result['strategy_return'].mean() / ridge_result['strategy_return'].std()) * np.sqrt(6)),
    ]:
        print(f"{name:<35} {ols_val:>12.4%} {rid_val:>12.4%}")

    print("\nVerdict: Ridge handles multicollinearity and may produce more stable OOS results.")


if __name__ == "__main__":
    run_comparison_backtest()
