import os
import random
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from purgedcv import PurgedKFold
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
    engineered = ['hv_20', 'speech_macro_interact', 'sentiment_momentum']
    return broad + strict + engineered + ['econ_surprise', 'returns_lag1']


def _get_purged_split(df, n_splits=5, embargo_bars=1):
    """Build PurgedKFold indices for H4 return prediction."""
    prediction_times = pd.Series(df.index, index=df.index)
    evaluation_times = prediction_times + pd.Timedelta(hours=4)
    pkf = PurgedKFold(
        n_splits=n_splits,
        prediction_times=prediction_times,
        evaluation_times=evaluation_times,
        purge_horizon='4h',
        embargo=f'{embargo_bars * 4}h',
    )
    return pkf


# ============================================================
# Phase 1: Purged Cross-Validation Backtest
# ============================================================
def run_purged_cv_backtest(model_type='ridge', n_splits=5, pip_cost=0.00005):
    df = _load_data()
    features = get_features()
    X, y = df[features], df['returns']
    pkf = _get_purged_split(df, n_splits=n_splits)

    fold_metrics = []

    print(f"\n{'=' * 70}")
    print(f"  PURGED CV BACKTEST ({model_type.upper()}) - {n_splits} folds, embargo=4h")
    print(f"{'=' * 70}")

    for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(df)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        if model_type == 'ridge':
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.1,
                reg_lambda=0.1, reg_alpha=0.01, random_state=42, verbosity=0,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            X_tr = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_tr).fit()
            X_te = X_test.copy()
            X_te.insert(0, 'const', 1.0)
            y_pred = model.predict(X_te)

        fold_df = df.iloc[test_idx].copy()
        fold_df['predicted_return'] = y_pred
        fold_df = _apply_trading_costs(fold_df, pip_cost)
        r2, hr, ir, ts, tm = _compute_metrics(fold_df)

        fold_metrics.append({
            'fold': fold_idx, 'oos_r2': r2, 'hit_rate': hr,
            'info_ratio': ir, 'total_strat': ts, 'total_mkt': tm,
            'train_size': len(X_train), 'test_size': len(X_test),
        })
        print(f"  Fold {fold_idx}: OOS R2={r2:+.5f} | Hit={hr:.2%} | IR={ir:.4f} | Ret={ts:+.2%} | "
              f"Train={len(X_train)} Test={len(X_test)}")

    avg_r2 = np.mean([m['oos_r2'] for m in fold_metrics])
    avg_hr = np.mean([m['hit_rate'] for m in fold_metrics])
    avg_ir = np.mean([m['info_ratio'] for m in fold_metrics])
    avg_ret = np.mean([m['total_strat'] for m in fold_metrics])

    print(f"\n  Purged CV Average: OOS R2={avg_r2:+.5f} | Hit={avg_hr:.2%} | IR={avg_ir:.4f} | Ret={avg_ret:+.2%}")
    return fold_metrics


# ============================================================
# Phase 2: Stacking Ensemble (Ridge + XGBoost + OLS -> ElasticNet)
# ============================================================
def run_stacking_ensemble_backtest(n_splits=5, pip_cost=0.00005):
    df = _load_data()
    features = get_features()
    X, y = df[features], df['returns']
    pkf = _get_purged_split(df, n_splits=n_splits)

    all_test_dfs = []

    print(f"\n{'=' * 70}")
    print("      STACKING ENSEMBLE: Ridge + XGBoost + OLS -> ElasticNet")
    print(f"{'=' * 70}")

    for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(df)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        # Level 0: Train base learners
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
        ridge.fit(X_train, y_train)

        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.1,
            reg_lambda=0.1, reg_alpha=0.01, random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)

        X_tr_c = sm.add_constant(X_train)
        ols = sm.OLS(y_train, X_tr_c).fit()
        X_te_c = X_test.copy()
        X_te_c.insert(0, 'const', 1.0)

        # Generate Level 1 features (out-of-fold predictions)
        # Inner CV within training fold for OOF predictions
        inner_pkf = _get_purged_split(df.iloc[train_idx], n_splits=3, embargo_bars=1)
        inner_X = X_train
        inner_y = y_train
        oof_preds = np.zeros((len(inner_X), 3))

        for inner_idx, (tr_in, val_in) in enumerate(inner_pkf.split(df.iloc[train_idx])):
            Xi_tr, yi_tr = inner_X.iloc[tr_in], inner_y.iloc[tr_in]
            Xi_val = inner_X.iloc[val_in]

            r_in = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
            r_in.fit(Xi_tr, yi_tr)
            oof_preds[val_in, 0] = r_in.predict(Xi_val)

            x_in = xgb.XGBRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.1,
                reg_lambda=0.1, reg_alpha=0.01, random_state=42, verbosity=0,
            )
            x_in.fit(Xi_tr, yi_tr)
            oof_preds[val_in, 1] = x_in.predict(Xi_val)

            Xi_tr_c = sm.add_constant(Xi_tr)
            o_in = sm.OLS(yi_tr, Xi_tr_c).fit()
            Xi_val_c = Xi_val.copy()
            Xi_val_c.insert(0, 'const', 1.0)
            oof_preds[val_in, 2] = o_in.predict(Xi_val_c)

        # Train ElasticNet meta-learner on OOF predictions
        meta = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
        meta.fit(oof_preds, inner_y)

        # Generate test predictions from base learners
        test_preds = np.column_stack([
            ridge.predict(X_test),
            xgb_model.predict(X_test),
            ols.predict(X_te_c),
        ])
        y_pred = meta.predict(test_preds)

        fold_df = df.iloc[test_idx].copy()
        fold_df['predicted_return'] = y_pred
        fold_df = _apply_trading_costs(fold_df, pip_cost)
        r2, hr, ir, ts, tm = _compute_metrics(fold_df)

        all_test_dfs.append(fold_df)
        print(f"  Fold {fold_idx}: OOS R2={r2:+.5f} | Hit={hr:.2%} | IR={ir:.4f} | Ret={ts:+.2%}")

    combined = pd.concat(all_test_dfs).sort_index()
    r2 = r2_score(combined['returns'], combined['predicted_return'])
    hr = (combined['trading_signal'] == np.sign(combined['returns'])).mean()
    ir = (combined['strategy_return'].mean() / combined['strategy_return'].std()) * np.sqrt(6)
    ts = combined['cum_strategy_returns'].iloc[-1]
    tm = combined['cum_market_returns'].iloc[-1]

    print(f"\n  Stacking Ensemble Aggregate: OOS R2={r2:+.5f} | Hit={hr:.2%} | IR={ir:.4f} | Ret={ts:+.2%}")
    combined.to_csv(os.path.join('data', 'backtest_results_stacking.csv'))
    return combined


# ============================================================
# Phase 4: XGBoost Hyperparameter Tuning with PurgedKFold
# ============================================================
def tune_xgboost(n_iter=30, n_splits=3, pip_cost=0.00005):
    df = _load_data()
    features = get_features()
    X, y = df[features], df['returns']
    pkf = _get_purged_split(df, n_splits=n_splits)

    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'reg_lambda': [0.01, 0.1, 1.0, 10.0],
        'reg_alpha': [0.0, 0.01, 0.1, 1.0, 5.0],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    # Build split indices for sklearn CV
    cv_indices = []
    for train_idx, test_idx in pkf.split(df):
        cv_indices.append((train_idx, test_idx))

    print(f"\n{'=' * 70}")
    print(f"  XGBoost HYPERPARAMETER TUNING (Purged CV, {n_iter} iterations)")
    print(f"{'=' * 70}")

    xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=n_iter,
        cv=cv_indices, scoring='neg_mean_squared_error',
        random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(X, y)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV MSE: {search.best_score_:.8f}")

    # Evaluate best model on full purged CV
    tuned_r2 = []
    for train_idx, test_idx in cv_indices:
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        best = xgb.XGBRegressor(**search.best_params_, n_estimators=200, random_state=42, verbosity=0)
        best.fit(X_tr, y_tr)
        yp = best.predict(X_te)
        tuned_r2.append(r2_score(y_te, yp))

    print(f"  Mean Purged CV OOS R2 with best params: {np.mean(tuned_r2):+.5f}")
    print(f"  Per-fold R2: {[f'{v:+.4f}' for v in tuned_r2]}")

    return search.best_params_


# ============================================================
# Static (non-purged) backtests for direct comparison
# ============================================================
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
    df = _load_data()
    split_idx = int(len(df) * train_window_pct)
    features = get_features()

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    X_train, y_train = train_df[features], train_df['returns']
    X_test, y_test = test_df[features], test_df['returns']

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.1,
        reg_lambda=0.1, reg_alpha=0.01, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    in_sample_r2 = 1 - ss_res / ss_tot
    oos_r2 = r2_score(y_test, y_pred_test)

    print(f"\n{'=' * 70}")
    print("      XGBoOST BACKTEST")
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


# ============================================================
# Phase 5: Regime-Adaptive Blending (volatility tercile models)
# ============================================================
def run_regime_backtest(n_splits=5, pip_cost=0.00005):
    df = _load_data()
    features = get_features()

    if 'hv_20' not in df.columns:
        print("hv_20 not found; skipping regime backtest.")
        return None

    # Define volatility regimes based on hv_20 terciles
    low_cut, high_cut = df['hv_20'].quantile([1/3, 2/3])
    df['vol_regime'] = pd.cut(
        df['hv_20'], bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=['low', 'mid', 'high'],
    )

    pkf = _get_purged_split(df, n_splits=n_splits)
    regime_metrics = {r: [] for r in ['low', 'mid', 'high']}
    all_fold_dfs = []

    print(f"\n{'=' * 70}")
    print("  PHASE 5: REGIME-ADAPTIVE BACKTEST (volatility tercile models)")
    print(f"{'=' * 70}")

    for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(df)):
        X_train = df[features].iloc[train_idx]
        y_train = df['returns'].iloc[train_idx]
        X_test = df[features].iloc[test_idx]
        y_test = df['returns'].iloc[test_idx]
        vol_train = df['vol_regime'].iloc[train_idx]
        vol_test = df['vol_regime'].iloc[test_idx]

        if len(X_train) < 30 or len(X_test) < 5:
            continue

        # Train regime-specific models (Ridge + XGBoost)
        regime_models = {}
        for regime in ['low', 'mid', 'high']:
            mask = vol_train == regime
            if mask.sum() < 10:
                continue
            Xr, yr = X_train[mask], y_train[mask]
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring='neg_mean_squared_error')
            ridge.fit(Xr, yr)
            xg = xgb.XGBRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.1,
                reg_lambda=0.1, reg_alpha=0.01, random_state=42, verbosity=0,
            )
            xg.fit(Xr, yr)
            regime_models[regime] = {'ridge': ridge, 'xgb': xg}

        # Predict using regime-specific models
        fold_df = df.iloc[test_idx].copy()
        fold_df['predicted_return'] = np.nan

        for regime in ['low', 'mid', 'high']:
            if regime not in regime_models:
                continue
            mask = vol_test == regime
            if mask.sum() == 0:
                continue
            idx = mask[mask].index
            X_sub = X_test.loc[idx]
            r_pred = regime_models[regime]['ridge'].predict(X_sub)
            x_pred = regime_models[regime]['xgb'].predict(X_sub)
            fold_df.loc[idx, 'predicted_return'] = 0.5 * r_pred + 0.5 * x_pred

        # Fill any remaining NaN with global XGBoost
        nan_idx = fold_df['predicted_return'].isna()
        if nan_idx.any():
            xgb_global = xgb.XGBRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.1,
                reg_lambda=0.1, reg_alpha=0.01, random_state=42, verbosity=0,
            )
            xgb_global.fit(X_train, y_train)
            fold_df.loc[nan_idx, 'predicted_return'] = xgb_global.predict(X_test.loc[nan_idx[nan_idx].index])

        fold_df = _apply_trading_costs(fold_df, pip_cost)
        r2, hr, ir, ts, tm = _compute_metrics(fold_df)
        all_fold_dfs.append(fold_df)

        # Per-regime metrics
        for regime in ['low', 'mid', 'high']:
            mask = vol_test == regime
            if mask.sum() < 3:
                continue
            sub = fold_df.loc[mask[mask].index]
            r2r = r2_score(sub['returns'], sub['predicted_return'])
            regime_metrics[regime].append(r2r)

        print(f"  Fold {fold_idx}: OOS R2={r2:+.5f} | Hit={hr:.2%} | IR={ir:.4f} | Ret={ts:+.2%}")

    # Summary by regime
    print(f"\n  Per-Regime OOS R2 (averaged across folds):")
    for regime in ['low', 'mid', 'high']:
        vals = regime_metrics[regime]
        if vals:
            print(f"    {regime.title()} Vol: {np.mean(vals):+.5f} (n={len(vals)} folds)")

    combined = pd.concat(all_fold_dfs).sort_index()
    r2 = r2_score(combined['returns'], combined['predicted_return'])
    hr = (combined['trading_signal'] == np.sign(combined['returns'])).mean()
    ir = (combined['strategy_return'].mean() / combined['strategy_return'].std()) * np.sqrt(6)
    ts = combined['cum_strategy_returns'].iloc[-1]
    tm = combined['cum_market_returns'].iloc[-1]

    print(f"\n  Regime-Adaptive Aggregate: OOS R2={r2:+.5f} | Hit={hr:.2%} | IR={ir:.4f} | Ret={ts:+.2%}")
    combined.to_csv(os.path.join('data', 'backtest_results_regime.csv'))

    # Save regime distribution
    print(f"  Regime distribution:\n{df['vol_regime'].value_counts().to_string()}")
    return combined


# ============================================================
# Orchestrator: runs everything
# ============================================================
def run_comparison_backtest(train_window_pct=0.70, pip_cost=0.00005):
    # Static (non-purged) comparison
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
        ('Hit Rate', (np.sign(ols_result['returns']) == ols_result['trading_signal']).mean(),
         (np.sign(ridge_result['returns']) == ridge_result['trading_signal']).mean()),
        ('Info Ratio', (ols_result['strategy_return'].mean() / ols_result['strategy_return'].std()) * np.sqrt(6),
         (ridge_result['strategy_return'].mean() / ridge_result['strategy_return'].std()) * np.sqrt(6)),
        ('Total Return', ols_result['cum_strategy_returns'].iloc[-1],
         ridge_result['cum_strategy_returns'].iloc[-1]),
    ]:
        print(f"{name:<35} {ols_val:>12.4%} {rid_val:>12.4%}")

    xgb_result = run_xgboost_backtest(train_window_pct, pip_cost)
    almon_result = run_almon_pdl_backtest(train_window_pct, pip_cost)

    # Phase 1: Purged CV backtests
    print("\n" + "=" * 70)
    print("  PHASE 1: PURGED CROSS-VALIDATION BACKTESTS")
    print("=" * 70)
    purged_ridge = run_purged_cv_backtest('ridge', n_splits=5, pip_cost=pip_cost)
    purged_xgb = run_purged_cv_backtest('xgboost', n_splits=5, pip_cost=pip_cost)

    # Phase 2: Stacking Ensemble
    stacking_result = run_stacking_ensemble_backtest(n_splits=5, pip_cost=pip_cost)

    # Phase 4: Hyperparameter tuning
    best_xgb_params = tune_xgboost(n_iter=30, n_splits=3, pip_cost=pip_cost)

    # Phase 5: Regime-adaptive blending
    regime_result = run_regime_backtest(n_splits=5, pip_cost=pip_cost)

    # --- Final comparison table ---
    all_results = {
        'OLS': ols_result,
        'Ridge': ridge_result,
        'XGBoost': xgb_result,
        'AlmonPDL': almon_result,
        'StackEns': stacking_result,
        'RegimeBlend': regime_result,
    }

    print("\n" + "=" * 70)
    print("      MODEL COMPARISON (Static 70/30 + Stacking)")
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
    print("  PURGED CV SUMMARY (honest time-series metrics)")
    print("=" * 70)
    for label, metrics in [('Ridge (Purged CV)', purged_ridge), ('XGBoost (Purged CV)', purged_xgb)]:
        r2s = [m['oos_r2'] for m in metrics]
        hrs = [m['hit_rate'] for m in metrics]
        irs = [m['info_ratio'] for m in metrics]
        rets = [m['total_strat'] for m in metrics]
        print(f"  {label}: OOS R2={np.mean(r2s):+.5f} | Hit={np.mean(hrs):.2%} | "
              f"IR={np.mean(irs):.4f} | Ret={np.mean(rets):+.2%}")

    print(f"\n  Best XGBoost params from tuning: {best_xgb_params}")

    # Rolling walk-forward
    run_rolling_walk_forward('ols', train_months=6, eval_months=2)
    run_rolling_walk_forward('ridge', train_months=6, eval_months=2)

    print("\n" + "=" * 70)
    print("  ALL BACKTESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison_backtest()
