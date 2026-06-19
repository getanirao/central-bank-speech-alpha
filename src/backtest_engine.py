import os
import random
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def _load_panel():
    input_path = os.path.join('data', 'currency_panel_h4.csv')
    df = pd.read_csv(input_path)
    time_col = 'Datetime' if 'Datetime' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values([time_col, 'pair_id']).reset_index(drop=True)
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
    test_df['trade_change'] = test_df.groupby('pair_id')['trading_signal'].diff().abs().fillna(0)
    test_df['strategy_return'] = (test_df['trading_signal'] * test_df['returns']) - (test_df['trade_change'] * pip_cost)
    test_df['cum_market_returns'] = (1 + test_df['returns']).groupby(test_df['pair_id']).cumprod() - 1
    test_df['cum_strategy_returns'] = (1 + test_df['strategy_return']).groupby(test_df['pair_id']).cumprod() - 1
    return test_df


def run_static_backtest(model_type='ridge', train_window_pct=0.70, pip_cost=0.00005):
    df = _load_panel()

    pairs = df['pair_id'].unique()
    features = [f'speech_lag_{i}' for i in range(1, 7)] + ['econ_surprise', 'returns_lag1']

    all_results = []

    print(f"\n--- Panel {model_type.upper()} Backtest ({pip_cost * 10000:.1f} pip cost) ---")

    for pid in pairs:
        pair_df = df[df['pair_id'] == pid].reset_index(drop=True)
        split_idx = int(len(pair_df) * train_window_pct)

        train_df = pair_df.iloc[:split_idx]
        test_df = pair_df.iloc[split_idx:]

        if model_type == 'ridge':
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
            model.fit(train_df[features], train_df['returns'])
            y_pred_test = model.predict(test_df[features])
            y_pred_train = model.predict(train_df[features])
            ss_res = np.sum((train_df['returns'] - y_pred_train) ** 2)
            ss_tot = np.sum((train_df['returns'] - np.mean(train_df['returns'])) ** 2)
            in_sample_r2 = 1 - ss_res / ss_tot
        else:
            X_train = sm.add_constant(train_df[features])
            model = sm.OLS(train_df['returns'], X_train).fit()
            X_test = test_df[features].copy()
            X_test.insert(0, 'const', 1.0)
            y_pred_test = model.predict(X_test)
            in_sample_r2 = model.rsquared

        test_df = test_df.copy()
        test_df['predicted_return'] = y_pred_test
        test_df = _apply_trading_costs(test_df, pip_cost)

        r2, hit_rate, ir, total_strat, total_mkt = _compute_metrics(test_df)

        all_results.append({
            'pair': pid,
            'in_sample_r2': in_sample_r2,
            'oos_r2': r2,
            'hit_rate': hit_rate,
            'info_ratio': ir,
            'total_strat': total_strat,
            'total_mkt': total_mkt,
            'test_df': test_df,
        })

        print(f"  {pid}: OOS R2={r2:+.5f} | Hit={hit_rate:.2%} | IR={ir:.4f} | Ret={total_strat:+.2%}")

    avg_r2 = np.mean([r['oos_r2'] for r in all_results])
    avg_hr = np.mean([r['hit_rate'] for r in all_results])
    avg_ir = np.mean([r['info_ratio'] for r in all_results])
    avg_ret = np.mean([r['total_strat'] for r in all_results])
    avg_is_r2 = np.mean([r['in_sample_r2'] for r in all_results])

    print(f"  Panel Average: OOS R2={avg_r2:+.5f} | Hit={avg_hr:.2%} | IR={avg_ir:.4f} | Ret={avg_ret:+.2%}")

    combined = pd.concat([r['test_df'] for r in all_results])
    combined.to_csv(os.path.join('data', f'backtest_results_{model_type}.csv'), index=False)

    return all_results, avg_r2, avg_hr, avg_ir, avg_ret, avg_is_r2


def run_nonlinear_oos_backtest(train_window_pct=0.70, pip_cost=0.00005):
    """Random Forest backtest on multi-currency panel, run per-pair."""
    df = _load_panel()
    pairs = df['pair_id'].unique()
    features = [f'speech_lag_{i}' for i in range(1, 7)] + ['econ_surprise', 'returns_lag1']

    all_results = []

    print(f"\n{'=' * 70}")
    print("      NON-LINEAR RANDOM FOREST BACKTEST (Panel)")
    print(f"{'=' * 70}")

    for pid in pairs:
        pair_df = df[df['pair_id'] == pid].reset_index(drop=True)
        split_idx = int(len(pair_df) * train_window_pct)

        train_df = pair_df.iloc[:split_idx]
        test_df = pair_df.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(train_df[features], train_df['returns'])
        y_pred = model.predict(test_df[features])

        if pid == pairs[0]:
            importances = model.feature_importances_
            for feat, imp in zip(features, importances):
                print(f"  Feature Importance - {feat}: {imp:.4f}")

        test_df = test_df.copy()
        test_df['predicted_return'] = y_pred
        test_df = _apply_trading_costs(test_df, pip_cost)

        r2, hit_rate, ir, total_strat, total_mkt = _compute_metrics(test_df)

        all_results.append(test_df)
        print(f"  {pid}: OOS R2={r2:+.5f} | Hit={hit_rate:.2%} | IR={ir:.4f} | Ret={total_strat:+.2%}")

    combined = pd.concat(all_results)
    oos_r2 = r2_score(combined['returns'], combined['predicted_return'])
    hit_rate = (combined['trading_signal'] == np.sign(combined['returns'])).mean()
    ir = (combined['strategy_return'].mean() / combined['strategy_return'].std()) * np.sqrt(6)
    total_strat = combined['cum_strategy_returns'].iloc[-1]
    total_mkt = combined['cum_market_returns'].iloc[-1]

    print(f"\n  Panel Aggregate: OOS R2={oos_r2:+.5f} | Hit={hit_rate:.2%} | IR={ir:.4f} | Ret={total_strat:+.2%}")

    combined.to_csv(os.path.join('data', 'backtest_results_rf.csv'), index=False)
    return combined


def run_almon_pdl_backtest(train_window_pct=0.70, pip_cost=0.00005):
    """Almon PDL backtest on multi-currency panel, run per-pair."""
    df = _load_panel()

    if 'almon_term_1' not in df.columns:
        print("Almon terms not found.")
        return None

    pairs = df['pair_id'].unique()
    features = ['almon_term_1', 'almon_term_2', 'econ_surprise', 'returns_lag1']

    all_results = []

    print(f"\n{'=' * 70}")
    print("      ALMON POLYNOMIAL DISTRIBUTED LAG BACKTEST (Panel)")
    print(f"{'=' * 70}")

    for pid in pairs:
        pair_df = df[df['pair_id'] == pid].reset_index(drop=True)
        split_idx = int(len(pair_df) * train_window_pct)

        train_df = pair_df.iloc[:split_idx]
        test_df = pair_df.iloc[split_idx:]

        X_train = sm.add_constant(train_df[features])
        y_train = train_df['returns']
        model = sm.OLS(y_train, X_train).fit()

        X_test = sm.add_constant(test_df[features], has_constant='add')
        y_test = test_df['returns']
        y_pred = model.predict(X_test)

        test_df = test_df.copy()
        test_df['predicted_return'] = y_pred
        test_df = _apply_trading_costs(test_df, pip_cost)

        r2, hit_rate, ir, total_strat, total_mkt = _compute_metrics(test_df)

        all_results.append(test_df)
        print(f"  {pid}: OOS R2={r2:+.5f} | Hit={hit_rate:.2%} | IR={ir:.4f} | Ret={total_strat:+.2%}")

    combined = pd.concat(all_results)
    oos_r2 = r2_score(combined['returns'], combined['predicted_return'])
    hit_rate = (combined['trading_signal'] == np.sign(combined['returns'])).mean()
    ir = (combined['strategy_return'].mean() / combined['strategy_return'].std()) * np.sqrt(6)
    total_strat = combined['cum_strategy_returns'].iloc[-1]
    total_mkt = combined['cum_market_returns'].iloc[-1]

    print(f"\n  Panel Aggregate: OOS R2={oos_r2:+.5f} | Hit={hit_rate:.2%} | IR={ir:.4f} | Ret={total_strat:+.2%}")

    combined.to_csv(os.path.join('data', 'backtest_results_almon.csv'), index=False)
    return combined


def run_rolling_walk_forward(model_type='ridge', train_months=6, eval_months=2, pip_cost=0.00005):
    df = _load_panel()
    pairs = df['pair_id'].unique()
    features = [f'speech_lag_{i}' for i in range(1, 7)] + ['econ_surprise', 'returns_lag1']

    results = []

    print(f"\n{'=' * 70}")
    print(f"  Rolling Walk-Forward ({model_type.upper()}) Panel — {train_months}m train / {eval_months}m eval")
    print(f"{'=' * 70}")

    for pid in pairs:
        pair_df = df[df['pair_id'] == pid].sort_values('Datetime' if 'Datetime' in df.columns else 'date').reset_index(drop=True)
        dates = pd.to_datetime(pair_df.columns[0] if 'Datetime' not in pair_df.columns and 'date' not in pair_df.columns else None)
        start_date = pair_df['Datetime' if 'Datetime' in pair_df.columns else 'date'].iloc[0]
        end_date = pair_df['Datetime' if 'Datetime' in pair_df.columns else 'date'].iloc[-1]

        train_delta = pd.Timedelta(days=30 * train_months)
        eval_delta = pd.Timedelta(days=30 * eval_months)

        window_start = start_date
        window_idx = 0
        while window_start + train_delta + eval_delta <= end_date:
            train_end = window_start + train_delta
            test_end = train_end + eval_delta

            train_df = pair_df[(pair_df['Datetime' if 'Datetime' in pair_df.columns else 'date'] >= window_start) &
                               (pair_df['Datetime' if 'Datetime' in pair_df.columns else 'date'] < train_end)]
            test_df = pair_df[(pair_df['Datetime' if 'Datetime' in pair_df.columns else 'date'] >= train_end) &
                              (pair_df['Datetime' if 'Datetime' in pair_df.columns else 'date'] < test_end)]

            if len(train_df) < 10 or len(test_df) < 5:
                window_start = train_end
                continue

            X_train, y_train = train_df[features], train_df['returns']
            X_test, y_test = test_df[features], test_df['returns']

            try:
                if model_type == 'ridge':
                    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    X_train_c = sm.add_constant(X_train)
                    model = sm.OLS(y_train, X_train_c).fit()
                    X_test_c = sm.add_constant(X_test, has_constant='add')
                    y_pred = model.predict(X_test_c)
            except Exception as e:
                window_start = train_end
                continue

            window_idx += 1
            test_df = test_df.copy()
            test_df['predicted_return'] = y_pred
            test_df['trading_signal'] = np.sign(test_df['predicted_return'])
            test_df['trade_change'] = test_df['trading_signal'].diff().abs().fillna(0)
            test_df['strategy_return'] = (test_df['trading_signal'] * test_df['returns']) - (test_df['trade_change'] * pip_cost)
            oos_r2 = r2_score(y_test, y_pred)
            hit_rate = (test_df['trading_signal'] == np.sign(test_df['returns'])).mean()
            total_strat = (1 + test_df['strategy_return']).prod() - 1

            results.append({
                'pair': pid,
                'window': window_idx,
                'train_end': train_end,
                'test_end': test_end,
                'oos_r2': oos_r2,
                'hit_rate': hit_rate,
                'return': total_strat,
            })

            window_start = train_end

    if results:
        res_df = pd.DataFrame(results)
        summary = res_df.groupby('window').agg({'oos_r2': 'mean', 'hit_rate': 'mean', 'return': 'mean'}).reset_index()
        print(f"\n{'─' * 70}")
        print(f"  ROLLING PANEL SUMMARY ({model_type.upper()})")
        print(f"{'─' * 70}")
        for _, row in summary.iterrows():
            print(f"  Window {int(row['window'])}: OOS R2={row['oos_r2']:+.5f} | Hit={row['hit_rate']:.2%} | Ret={row['return']:+.2%}")
        print(f"\n  Mean OOS R2: {summary['oos_r2'].mean():+.5f}")
        print(f"  Mean Hit Rate: {summary['hit_rate'].mean():.2%}")
        print(f"  Mean Window Return: {summary['return'].mean():+.2%}")

        res_df.to_csv(os.path.join('data', f'rolling_cv_{model_type}_panel.csv'), index=False)

    return results


def run_comparison_backtest(train_window_pct=0.70, pip_cost=0.00005):
    ols_result, ols_r2, ols_hr, ols_ir, ols_ret, ols_is = run_static_backtest('ols', train_window_pct, pip_cost)
    ridge_result, rid_r2, rid_hr, rid_ir, rid_ret, rid_is = run_static_backtest('ridge', train_window_pct, pip_cost)

    print("\n" + "=" * 60)
    print("      PANEL BACKTEST COMPARISON: OLS vs RIDGE")
    print("=" * 60)
    print(f"{'Metric':<35} {'OLS':>12} {'Ridge':>12}")
    print("-" * 59)
    print(f"{'Avg In-Sample R2':<35} {ols_is:>+12.5f} {rid_is:>+12.5f}")
    print(f"{'Avg OOS R2':<35} {ols_r2:>+12.5f} {rid_r2:>+12.5f}")
    print(f"{'Avg Hit Rate':<35} {ols_hr:>12.4%} {rid_hr:>12.4%}")
    print(f"{'Avg Info Ratio':<35} {ols_ir:>12.4f} {rid_ir:>12.4f}")
    print(f"{'Avg Total Return':<35} {ols_ret:>12.4%} {rid_ret:>12.4%}")

    rf_result = run_nonlinear_oos_backtest(train_window_pct, pip_cost)
    almon_result = run_almon_pdl_backtest(train_window_pct, pip_cost)

    all_results = {
        'OLS': pd.concat([r['test_df'] for r in ols_result]),
        'Ridge': pd.concat([r['test_df'] for r in ridge_result]),
        'RandomForest': rf_result,
        'AlmonPDL': almon_result,
    }

    print("\n" + "=" * 70)
    print("      FOUR-MODEL OOS COMPARISON (Multi-Currency Panel)")
    print("=" * 70)
    print(f"{'Model':<20} {'OOS R2':>10} {'Hit Rate':>10} {'Info Ratio':>12} {'Total Ret':>10}")
    print("-" * 64)
    for name, res in all_results.items():
        if res is None or len(res) == 0:
            continue
        r2 = r2_score(res['returns'], res['predicted_return'])
        hr = (np.sign(res['returns']) == res['trading_signal']).mean()
        ir = (res['strategy_return'].mean() / res['strategy_return'].std()) * np.sqrt(6)
        tr = res['cum_strategy_returns'].iloc[-1]
        print(f"{name:<20} {r2:>+10.5f} {hr:>10.4%} {ir:>12.4f} {tr:>+10.4%}")

    print("\n" + "=" * 70)
    print("  ALL BACKTESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison_backtest()
