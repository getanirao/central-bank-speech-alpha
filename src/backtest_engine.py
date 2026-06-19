import os
import random
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def _load_data():
    input_path = os.path.join('data', 'merged_h4.csv')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    return df


def _compute_metrics(test_df):
    r2 = r2_score(test_df['returns'], test_df['predicted_return'])
    hit_rate = (test_df['trading_signal'] == np.sign(test_df['returns'])).mean()
    ir = (test_df['strategy_return'].mean() / test_df['strategy_return'].std()) * np.sqrt(6) if test_df['strategy_return'].std() != 0 else 0.0
    total_strat = test_df['cum_strategy_returns'].iloc[-1]
    total_mkt = test_df['cum_market_returns'].iloc[-1]
    return r2, hit_rate, ir, total_strat, total_mkt


def _apply_trading_costs(test_df, pip_cost=0.00005):
    test_df = test_df.copy()
    test_df['trading_signal'] = np.sign(test_df['predicted_return'])
    test_df['trade_change'] = test_df['trading_signal'].diff().abs().fillna(0)
    test_df['strategy_return'] = (test_df['trading_signal'] * test_df['returns']) - (test_df['trade_change'] * pip_cost)
    test_df['cum_market_returns'] = (1 + test_df['returns']).cumprod() - 1
    test_df['cum_strategy_returns'] = (1 + test_df['strategy_return']).cumprod() - 1
    return test_df


def get_features():
    broad = [f'speech_lag_{i}' for i in range(1, 7)]
    strict = [f'strict_lag_{i}' for i in range(1, 7)]
    return broad + strict + ['econ_surprise', 'returns_lag1']


def run_static_backtest(model_type='ridge', train_window_pct=0.70, pip_cost=0.00005):
    df = _load_data()
    split_idx = int(len(df) * train_window_pct)
    features = get_features()

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\n--- Static {model_type.upper()} Backtest ({pip_cost * 10000:.1f} pip cost) ---")
    print(f"Training: {len(train_df)} rows | Testing: {len(test_df)} rows")

    if model_type == 'ridge':
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
        model.fit(train_df[features], train_df['returns'])
        y_pred_test = model.predict(test_df[features])
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

    test_df = test_df.copy()
    test_df['predicted_return'] = y_pred_test
    test_df = _apply_trading_costs(test_df, pip_cost)

    r2, hit_rate, ir, total_strat, total_mkt = _compute_metrics(test_df)

    print(f"\nIn-Sample R2:       {in_sample_r2:.5f}")
    print(f"OOS R2:             {oos_r2:.5f}")
    print(f"Info Ratio (ann.):  {ir:.4f}")
    print(f"Hit Rate:           {hit_rate:.4%}")
    print(f"Total Strategy OOS: {total_strat:.4%}")
    print(f"Total Market OOS:   {total_mkt:.4%}")

    test_df.to_csv(os.path.join('data', f'backtest_results_{model_type}.csv'))
    return test_df


def run_xgboost_backtest(train_window_pct=0.70, pip_cost=0.00005):
    """XGBoost with early stopping and L1/L2 regularization."""
    df = _load_data()
    split_idx = int(len(df) * train_window_pct)
    features = get_features()

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df['returns']
    X_test, y_test = test_df[features], test_df['returns']

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.1,
        reg_lambda=0.1,
        reg_alpha=0.01,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    in_sample_r2 = 1 - ss_res / ss_tot
    oos_r2 = r2_score(y_test, y_pred_test)

    print(f"\n{'=' * 70}")
    print("      XGBoOST BACKTEST (Early Stopping + L1/L2)")
    print(f"{'=' * 70}")
    print(f"Training: {len(train_df)} rows | Testing: {len(test_df)} rows")

    importances = model.feature_importances_
    for feat, imp in zip(features, importances):
        print(f"  Feature Importance - {feat}: {imp:.4f}")

    test_df = test_df.copy()
    test_df['predicted_return'] = y_pred_test
    test_df = _apply_trading_costs(test_df, pip_cost)

    r2, hit_rate, ir, total_strat, total_mkt = _compute_metrics(test_df)

    print(f"\nIn-Sample R2:       {in_sample_r2:.5f}")
    print(f"OOS R2:             {oos_r2:.5f}")
    print(f"Info Ratio (ann.):  {ir:.4f}")
    print(f"Hit Rate:           {hit_rate:.4%}")
    print(f"Total Strategy OOS: {total_strat:.4%}")
    print(f"Total Market OOS:   {total_mkt:.4%}")

    test_df.to_csv(os.path.join('data', 'backtest_results_xgb.csv'))
    return test_df


def run_almon_pdl_backtest(train_window_pct=0.70, pip_cost=0.00005):
    df = _load_data()
    if 'almon_term_1' not in df.columns:
        print("Almon terms not found.")
        return None

    split_idx = int(len(df) * train_window_pct)
    features = ['almon_term_1', 'almon_term_2', 'econ_surprise', 'returns_lag1']

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = sm.add_constant(train_df[features])
    y_train = train_df['returns']
    model = sm.OLS(y_train, X_train).fit()

    X_test = sm.add_constant(test_df[features], has_constant='add')
    y_test = test_df['returns']
    y_pred = model.predict(X_test)

    print(f"\n{'=' * 70}")
    print("      ALMON POLYNOMIAL DISTRIBUTED LAG BACKTEST")
    print(f"{'=' * 70}")
    print(f"Training: {len(train_df)} rows | Testing: {len(test_df)} rows")
    print(model.summary().tables[1])

    test_df = test_df.copy()
    test_df['predicted_return'] = y_pred
    test_df = _apply_trading_costs(test_df, pip_cost)

    r2, hit_rate, ir, total_strat, total_mkt = _compute_metrics(test_df)

    print(f"\nOOS R2:             {r2:.5f}")
    print(f"Hit Rate:           {hit_rate:.4%}")
    print(f"Info Ratio (ann.):  {ir:.4f}")
    print(f"Total Strategy OOS: {total_strat:.4%}")
    print(f"Total Market OOS:   {total_mkt:.4%}")

    test_df.to_csv(os.path.join('data', 'backtest_results_almon.csv'))
    return test_df


def run_rolling_walk_forward(model_type='ridge', train_months=6, eval_months=2, pip_cost=0.00005):
    df = _load_data()
    features = get_features()

    start_date = df.index.min()
    end_date = df.index.max()
    train_delta = pd.Timedelta(days=30 * train_months)
    eval_delta = pd.Timedelta(days=30 * eval_months)

    windows = []
    window_start = start_date
    while window_start + train_delta + eval_delta <= end_date:
        train_end = window_start + train_delta
        test_end = train_end + eval_delta
        windows.append((window_start, train_end, test_end))
        window_start = train_end

    lag4_tracker = []
    oos_r2_tracker = []
    hit_rate_tracker = []
    total_strat_tracker = []

    print(f"\n{'=' * 70}")
    print(f"  Rolling Walk-Forward ({model_type.upper()})  Train {train_months}m / Eval {eval_months}m")
    print(f"{'=' * 70}")

    for i, (w_start, w_train_end, w_test_end) in enumerate(windows):
        train_df = df.loc[w_start:w_train_end]
        test_df = df.loc[w_train_end:w_test_end].iloc[1:]

        if len(train_df) < 10 or len(test_df) < 5:
            continue

        X_train = train_df[features].copy()
        y_train = train_df['returns'].copy()
        X_test = test_df[features].copy()
        y_test = test_df['returns'].copy()

        X_train = X_train.dropna(axis=1, how='all')
        X_test = X_test[X_train.columns]

        try:
            if model_type == 'ridge':
                model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                coefs_dict = dict(zip(X_train.columns, model.coef_))
            else:
                X_train_c = sm.add_constant(X_train)
                model = sm.OLS(y_train, X_train_c).fit()
                X_test_c = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_c)
                coefs_dict = dict(zip(X_train.columns, model.params[1:]))

            lag4_coef = coefs_dict.get('speech_lag_4', 0.0)
        except Exception as e:
            print(f"  Window {i + 1}: SKIPPED ({e})")
            continue

        lag4_tracker.append((w_test_end, lag4_coef))

        test_df = test_df.copy()
        test_df['predicted_return'] = y_pred
        test_df['trading_signal'] = np.sign(test_df['predicted_return'])
        test_df['trade_change'] = test_df['trading_signal'].diff().abs().fillna(0)
        test_df['strategy_return'] = (test_df['trading_signal'] * test_df['returns']) - (test_df['trade_change'] * pip_cost)

        oos_r2 = r2_score(y_test, y_pred)
        oos_r2_tracker.append((w_test_end, oos_r2))

        hit_rate = (test_df['trading_signal'] == np.sign(test_df['returns'])).mean()
        hit_rate_tracker.append((w_test_end, hit_rate))

        total_strat = (1 + test_df['strategy_return']).prod() - 1
        total_strat_tracker.append((w_test_end, total_strat))

        print(f"  Window {i + 1}: {w_start.date()}{w_test_end.date()} | "
              f"OOS R2={oos_r2:+.5f} | Hit={hit_rate:.2%} | Lag-4 b={lag4_coef:+.6f} | "
              f"Ret={total_strat:+.2%}")

    print(f"\n{'-' * 70}")
    print(f"  ROLLING WALK-FORWARD SUMMARY ({model_type.upper()})")
    print(f"{'-' * 70}")
    print(f"  Total windows:               {len(lag4_tracker)}")
    print(f"  Mean Lag-4 coefficient:      {np.mean([c for _, c in lag4_tracker]):+.6f}")
    print(f"  Std Lag-4 coefficient:       {np.std([c for _, c in lag4_tracker]):.6f}")
    print(f"  Mean OOS R2:                 {np.mean([r for _, r in oos_r2_tracker]):+.5f}")
    print(f"  Mean Hit Rate:               {np.mean([h for _, h in hit_rate_tracker]):.2%}")
    print(f"  Mean Window Return:          {np.mean([r for _, r in total_strat_tracker]):+.2%}")

    result = pd.DataFrame({
        'lag4_coef': [c for _, c in lag4_tracker],
        'oos_r2': [r for _, r in oos_r2_tracker],
        'hit_rate': [h for _, h in hit_rate_tracker],
        'window_return': [r for _, r in total_strat_tracker],
    }, index=pd.DatetimeIndex([d for d, _ in lag4_tracker]))

    result.to_csv(os.path.join('data', f'rolling_cv_{model_type}.csv'))
    return result


def run_comparison_backtest(train_window_pct=0.70, pip_cost=0.00005):
    ols_result = run_static_backtest('ols', train_window_pct, pip_cost)
    ridge_result = run_static_backtest('ridge', train_window_pct, pip_cost)

    print("\n" + "=" * 60)
    print("      LINEAR BACKTEST COMPARISON: OLS vs RIDGE")
    print("=" * 60)
    print(f"{'Metric':<35} {'OLS':>12} {'Ridge':>12}")
    print("-" * 59)

    for name, oos_r2_val, rid_r2_val in [
        ('OOS R2',
         r2_score(ols_result['returns'], ols_result['predicted_return']),
         r2_score(ridge_result['returns'], ridge_result['predicted_return'])),
    ]:
        print(f"{name:<35} {oos_r2_val:>+12.5f} {rid_r2_val:>+12.5f}")

    for name, ols_val, rid_val in [
        ('Hit Rate',
         (np.sign(ols_result['returns']) == ols_result['trading_signal']).mean(),
         (np.sign(ridge_result['returns']) == ridge_result['trading_signal']).mean()),
        ('Info Ratio',
         (ols_result['strategy_return'].mean() / ols_result['strategy_return'].std()) * np.sqrt(6),
         (ridge_result['strategy_return'].mean() / ridge_result['strategy_return'].std()) * np.sqrt(6)),
        ('Total Return',
         ols_result['cum_strategy_returns'].iloc[-1],
         ridge_result['cum_strategy_returns'].iloc[-1]),
    ]:
        print(f"{name:<35} {ols_val:>12.4%} {rid_val:>12.4%}")

    xgb_result = run_xgboost_backtest(train_window_pct, pip_cost)
    almon_result = run_almon_pdl_backtest(train_window_pct, pip_cost)

    all_results = {
        'OLS': ols_result,
        'Ridge': ridge_result,
        'XGBoost': xgb_result,
        'AlmonPDL': almon_result,
    }

    print("\n" + "=" * 70)
    print("      FOUR-MODEL OOS COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'OOS R2':>10} {'Hit Rate':>10} {'Info Ratio':>12} {'Total Ret':>10}")
    print("-" * 64)
    for name, res in all_results.items():
        if res is None:
            continue
        r2 = r2_score(res['returns'], res['predicted_return'])
        hr = (np.sign(res['returns']) == res['trading_signal']).mean()
        ir = (res['strategy_return'].mean() / res['strategy_return'].std()) * np.sqrt(6)
        tr = res['cum_strategy_returns'].iloc[-1]
        print(f"{name:<20} {r2:>+10.5f} {hr:>10.4%} {ir:>12.4f} {tr:>+10.4%}")

    print("\n" + "=" * 70)
    print("  Rolling Walk-Forward (6m train / 2m eval)")
    print("=" * 70)
    run_rolling_walk_forward('ols', train_months=6, eval_months=2)
    run_rolling_walk_forward('ridge', train_months=6, eval_months=2)

    print("\n" + "=" * 70)
    print("  ALL BACKTESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison_backtest()
