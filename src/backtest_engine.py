import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score


def run_out_of_sample_backtest(train_window_pct=0.70):
    """
    Executes a strict historical walk-forward validation.
    Ensures zero future data leakage touches the predictive matrix.
    """
    input_path = os.path.join('data', 'merged_h4.csv')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    split_idx = int(len(df) * train_window_pct)

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    features = lag_cols + ['econ_surprise', 'returns_lag1']

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Training Baseline Sample Space: {len(train_df)} rows")
    print(f"Out-of-Sample Evaluation Space: {len(test_df)} rows")

    X_train = sm.add_constant(train_df[features])
    y_train = train_df['returns']
    model_fit = sm.OLS(y_train, X_train).fit()

    # Force constant column to ensure alignment with training params
    X_test = test_df[features].copy()
    X_test.insert(0, 'const', 1.0)
    y_test = test_df['returns']

    test_df = test_df.copy()
    test_df['predicted_return'] = model_fit.predict(X_test)

    oos_r2 = r2_score(y_test, test_df['predicted_return'])
    oos_mse = mean_squared_error(y_test, test_df['predicted_return'])

    print("\n=======================================================")
    print("      OUT-OF-SAMPLE (OOS) METRIC DIAGNOSTICS          ")
    print("=======================================================")
    print(f"In-Sample R-squared:   {model_fit.rsquared:.5f}")
    print(f"OUT-OF-SAMPLE R-squared: {oos_r2:.5f}")
    print(f"OOS Mean Squared Error:  {oos_mse:.7f}")

    if oos_r2 <= 0:
        print("\nWARNING: OOS R-squared is negative. The model is overfitted to historical anomalies.")
    else:
        print("\nSUCCESS: Structural alpha confirmed out-of-sample.")

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

    backtest_path = os.path.join('data', 'backtest_results.csv')
    test_df.to_csv(backtest_path)
    print(f"\nOOS trade logs written safely to: {backtest_path}")

    return test_df


if __name__ == "__main__":
    run_out_of_sample_backtest()
