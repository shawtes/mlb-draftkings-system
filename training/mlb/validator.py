"""
validator.py — Walk-forward temporal validation, per-player evaluation,
                and Conformalized Quantile Regression (CQR).

Part of the MLB DFS Training Pipeline v2.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================================
# Walk-Forward Temporal Validator
# ============================================================================

class WalkForwardValidator:
    """
    Time-series walk-forward cross-validation.

    - Splits by date, not by index position
    - Configurable embargo gap between train end and test start
    - Growing training window (expanding)
    - Returns per-fold metrics + all out-of-sample predictions
    """

    def __init__(self, n_splits=5, gap_days=7):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(self, df, date_col='date'):
        """
        Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain `date_col` as a datetime column.
        date_col : str
            Name of the date column.

        Yields
        ------
        (train_idx, test_idx) : tuple of np.ndarray
        """
        dates = pd.to_datetime(df[date_col])
        min_date = dates.min()
        max_date = dates.max()
        total_days = (max_date - min_date).days

        # Reserve first 40% of data for minimum training window
        min_train_days = int(total_days * 0.4)
        remaining_days = total_days - min_train_days
        test_block_size = remaining_days // self.n_splits

        if test_block_size < 1:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Total days={total_days}, min_train_days={min_train_days}"
            )

        for fold in range(self.n_splits):
            train_end_date = min_date + timedelta(days=min_train_days + fold * test_block_size)
            test_start_date = train_end_date + timedelta(days=self.gap_days)
            test_end_date = train_end_date + timedelta(days=(fold + 1) * test_block_size)

            # Cap test end at max date
            test_end_date = min(test_end_date, max_date)

            train_mask = dates <= train_end_date
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"  Fold {fold+1}: skipped (empty train or test set)")
                continue

            yield train_idx, test_idx

    def validate(self, df, X, y, build_and_fit_fn, date_col='date'):
        """
        Run walk-forward validation.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataframe (used for date splitting and metadata).
        X : array-like
            Preprocessed feature matrix (same row order as df).
        y : array-like
            Target vector.
        build_and_fit_fn : callable(X_train, y_train) -> model
            Function that builds, fits, and returns a model.
        date_col : str
            Date column name in df.

        Returns
        -------
        metrics_df : pd.DataFrame
            Per-fold metrics (fold, n_train, n_test, mae, rmse, r2).
        all_oos_preds : pd.DataFrame
            All out-of-sample predictions with Name, Date, Actual, Predicted.
        """
        y_arr = np.asarray(y)
        fold_metrics = []
        all_oos = []

        for fold_num, (train_idx, test_idx) in enumerate(self.split(df, date_col)):
            fold = fold_num + 1
            print(f"\n--- Fold {fold}/{self.n_splits} ---")
            print(f"  Train: {len(train_idx)} rows, Test: {len(test_idx)} rows")

            if hasattr(X, 'iloc'):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]

            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Build + fit model
            model = build_and_fit_fn(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

            fold_metrics.append({
                'fold': fold,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
            })

            # Collect OOS predictions
            oos_df = pd.DataFrame({
                'Name': df.iloc[test_idx]['Name'].values if 'Name' in df.columns else 'unknown',
                'Date': df.iloc[test_idx][date_col].values,
                'Actual': y_test,
                'Predicted': y_pred,
                'Fold': fold,
            })
            all_oos.append(oos_df)

        metrics_df = pd.DataFrame(fold_metrics)
        all_oos_df = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()

        # Summary
        if not metrics_df.empty:
            print(f"\n=== Walk-Forward CV Summary ({self.n_splits} folds) ===")
            print(f"  Mean MAE:  {metrics_df['mae'].mean():.4f} +/- {metrics_df['mae'].std():.4f}")
            print(f"  Mean RMSE: {metrics_df['rmse'].mean():.4f} +/- {metrics_df['rmse'].std():.4f}")
            print(f"  Mean R2:   {metrics_df['r2'].mean():.4f} +/- {metrics_df['r2'].std():.4f}")

        return metrics_df, all_oos_df

    def validate_pregame(self, df, features_raw, y, build_preprocess_fn,
                         build_model_fn, date_col='date', n_features=100):
        """
        Walk-forward validation with in-fold preprocessing.

        Unlike validate(), this fits the preprocessor and feature selector
        INSIDE each fold (on train data only), preventing data leakage from
        the scaler/selector seeing test data.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataframe (used for date splitting).
        features_raw : pd.DataFrame
            Raw feature DataFrame (numeric + categorical, before preprocessing).
        y : array-like
            Target vector.
        build_preprocess_fn : callable(numeric_features, categorical_features)
            Returns a fitted-ready ColumnTransformer preprocessor.
        build_model_fn : callable(X_train, y_train) -> model
            Builds, fits, and returns a model.
        date_col : str
            Date column name in df.
        n_features : int
            Number of features to select per fold.

        Returns
        -------
        metrics_df, all_oos_preds, last_fold_artifacts : tuple
            last_fold_artifacts = (preprocessor, selector) from final fold
            for use in final model training.
        """
        y_arr = np.asarray(y)
        fold_metrics = []
        all_oos = []
        last_preprocessor = None
        last_selector = None

        # Determine numeric vs categorical columns
        numeric_cols = features_raw.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features_raw.select_dtypes(include=['object', 'category']).columns.tolist()

        for fold_num, (train_idx, test_idx) in enumerate(self.split(df, date_col)):
            fold = fold_num + 1
            print(f"\n--- Fold {fold}/{self.n_splits} ---")
            print(f"  Train: {len(train_idx)} rows, Test: {len(test_idx)} rows")

            # Split raw features
            X_train_raw = features_raw.iloc[train_idx]
            X_test_raw = features_raw.iloc[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Print date ranges
            train_dates = pd.to_datetime(df.iloc[train_idx][date_col])
            test_dates = pd.to_datetime(df.iloc[test_idx][date_col])
            print(f"  Train dates: {train_dates.min().date()} to {train_dates.max().date()}")
            print(f"  Test dates:  {test_dates.min().date()} to {test_dates.max().date()}")
            print(f"  Test y: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

            # Fit preprocessor on TRAIN ONLY
            preprocessor = build_preprocess_fn(numeric_cols, categorical_cols)
            X_train_proc = preprocessor.fit_transform(X_train_raw)
            X_test_proc = preprocessor.transform(X_test_raw)

            # Feature selection on TRAIN ONLY
            from model_builder import build_feature_selector, IndexSelector
            selector = build_feature_selector(
                X_train_proc, y_train, n_features=n_features,
            )
            X_train_sel = selector.transform(X_train_proc)
            X_test_sel = selector.transform(X_test_proc)

            print(f"  Preprocessed: {X_train_proc.shape[1]} -> Selected: {X_train_sel.shape[1]}")

            # Build + fit model
            model = build_model_fn(X_train_sel, y_train)

            # Predict
            y_pred = model.predict(X_test_sel)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Baseline comparison
            global_mean = y_train.mean()
            baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), global_mean))
            baseline_r2 = r2_score(y_test, np.full(len(y_test), global_mean))

            print(f"  Model:    MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
            print(f"  Baseline: MAE={baseline_mae:.4f}  R2={baseline_r2:.4f} (train mean)")

            fold_metrics.append({
                'fold': fold,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'baseline_mae': baseline_mae,
                'baseline_r2': baseline_r2,
            })

            # Collect OOS predictions
            oos_df = pd.DataFrame({
                'Name': df.iloc[test_idx]['Name'].values if 'Name' in df.columns else 'unknown',
                'Date': df.iloc[test_idx][date_col].values,
                'Actual': y_test,
                'Predicted': y_pred,
                'Fold': fold,
            })
            all_oos.append(oos_df)

            last_preprocessor = preprocessor
            last_selector = selector

        metrics_df = pd.DataFrame(fold_metrics)
        all_oos_df = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()

        # Summary
        if not metrics_df.empty:
            print(f"\n=== Walk-Forward CV Summary ({self.n_splits} folds, PREGAME) ===")
            print(f"  Mean MAE:  {metrics_df['mae'].mean():.4f} +/- {metrics_df['mae'].std():.4f}")
            print(f"  Mean RMSE: {metrics_df['rmse'].mean():.4f} +/- {metrics_df['rmse'].std():.4f}")
            print(f"  Mean R2:   {metrics_df['r2'].mean():.4f} +/- {metrics_df['r2'].std():.4f}")
            print(f"  Baseline MAE: {metrics_df['baseline_mae'].mean():.4f}")
            print(f"  Baseline R2:  {metrics_df['baseline_r2'].mean():.4f}")
            improvement = metrics_df['mae'].mean() - metrics_df['baseline_mae'].mean()
            print(f"  MAE improvement over baseline: {improvement:.4f}")

        return metrics_df, all_oos_df, (last_preprocessor, last_selector)


# ============================================================================
# Per-player evaluation
# ============================================================================

def per_player_evaluation(predictions_df, min_samples=10):
    """
    Compute MAE and R2 per player from a predictions DataFrame.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns: Name, Actual, Predicted.
    min_samples : int
        Minimum number of samples per player.

    Returns
    -------
    pd.DataFrame
        Columns: Name, MAE, R2, n_samples. Sorted by MAE ascending.
    """
    results = []
    for name, group in predictions_df.groupby('Name'):
        if len(group) < min_samples:
            continue
        actual = group['Actual'].values
        predicted = group['Predicted'].values
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted) if len(actual) > 1 else 0.0
        results.append({
            'Name': name,
            'MAE': mae,
            'R2': r2,
            'n_samples': len(group),
        })

    player_df = pd.DataFrame(results).sort_values('MAE', ascending=True)
    print(f"\nPer-player evaluation: {len(player_df)} players with >= {min_samples} samples")
    if not player_df.empty:
        print(f"  Best MAE:  {player_df.iloc[0]['Name']} ({player_df.iloc[0]['MAE']:.4f})")
        print(f"  Worst MAE: {player_df.iloc[-1]['Name']} ({player_df.iloc[-1]['MAE']:.4f})")
        print(f"  Median MAE: {player_df['MAE'].median():.4f}")
    return player_df


# ============================================================================
# Conformalized Quantile Regression (CQR)
# ============================================================================

def conformalized_quantile_regression(quantile_models, X_cal, y_cal, alpha=0.10):
    """
    Calibrate quantile prediction intervals using held-out calibration data.

    For quantile pair (q_lo, q_hi) with target coverage 1-alpha:
      1. Predict q_lo, q_hi on calibration set
      2. Compute conformity scores: max(q_lo - y, y - q_hi)
      3. Adjustment = quantile(scores, (1-alpha)(1 + 1/n))
      4. Calibrated interval = [q_lo - adj, q_hi + adj]

    Parameters
    ----------
    quantile_models : dict
        {quantile: fitted_model} from build_quantile_models.
    X_cal : array-like
        Calibration features.
    y_cal : array-like
        Calibration targets.
    alpha : float
        Miscoverage rate. Default 0.10 → 90% coverage target.

    Returns
    -------
    adjustment : float
        The conformity adjustment value.
    coverage : float
        Empirical coverage on calibration set after adjustment.
    """
    if quantile_models is None:
        print("No quantile models — skipping CQR calibration")
        return 0.0, 0.0

    y_cal = np.asarray(y_cal)

    # Find the lower and upper quantile models
    quantiles = sorted(quantile_models.keys())
    q_lo = quantiles[0]   # e.g. 0.10
    q_hi = quantiles[-1]  # e.g. 0.90

    print(f"CQR calibration: q_lo={q_lo}, q_hi={q_hi}, alpha={alpha}")

    # Predict bounds on calibration set
    pred_lo = quantile_models[q_lo].predict(X_cal)
    pred_hi = quantile_models[q_hi].predict(X_cal)

    # Conformity scores
    scores = np.maximum(pred_lo - y_cal, y_cal - pred_hi)

    # Quantile of scores for guaranteed coverage
    n = len(y_cal)
    quantile_level = min((1 - alpha) * (1 + 1 / n), 1.0)
    adjustment = np.quantile(scores, quantile_level)

    # Check empirical coverage
    calibrated_lo = pred_lo - adjustment
    calibrated_hi = pred_hi + adjustment
    covered = (y_cal >= calibrated_lo) & (y_cal <= calibrated_hi)
    coverage = covered.mean()

    print(f"  CQR adjustment: {adjustment:.4f}")
    print(f"  Empirical coverage on cal set: {coverage:.1%} (target: {1-alpha:.0%})")

    return adjustment, coverage


def predict_with_quantiles(quantile_models, X, cqr_adjustment=0.0,
                           calibrators=None):
    """
    Generate distributional predictions using quantile models + CQR adjustment.

    Returns DataFrame with columns for each quantile + calibrated bounds.
    If calibrators are provided, applies isotonic calibration to probabilities.
    """
    if quantile_models is None:
        return None

    results = {}
    quantiles = sorted(quantile_models.keys())
    for q in quantiles:
        col_name = f'q{int(q*100):02d}'
        results[col_name] = quantile_models[q].predict(X)

    df = pd.DataFrame(results)

    # Add calibrated prediction interval
    q_lo = quantiles[0]
    q_hi = quantiles[-1]
    df['prediction_lower_80'] = df[f'q{int(q_lo*100):02d}'] - cqr_adjustment
    df['prediction_upper_80'] = df[f'q{int(q_hi*100):02d}'] + cqr_adjustment
    df['prediction_std'] = (df['prediction_upper_80'] - df['prediction_lower_80']) / 3.29  # approx for 90% CI

    # Raw probability of exceeding thresholds (from quantile predictions)
    median_pred = df[f'q{int(quantiles[len(quantiles)//2]*100):02d}']
    std_est = df['prediction_std']
    from scipy.stats import norm
    for threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
        z = (threshold - median_pred) / std_est.clip(lower=0.1)
        raw_prob = 1 - norm.cdf(z)

        col = f'prob_over_{threshold}'
        if calibrators and threshold in calibrators:
            # Apply isotonic calibration
            raw_arr = np.asarray(raw_prob)
            df[col] = calibrators[threshold].predict(raw_arr)
            df[col] = df[col].clip(0, 1)
        else:
            df[col] = raw_prob

    return df


# ============================================================================
# Probability Calibration (Isotonic Regression)
# ============================================================================

def fit_probability_calibrators(quantile_models, X_cal, y_cal, cqr_adjustment=0.0,
                                thresholds=None):
    """
    Fit isotonic regression calibrators for each probability threshold.

    Uses a held-out calibration set to learn the mapping from raw
    (normal-approximation) probabilities to true empirical frequencies.

    Parameters
    ----------
    quantile_models : dict
        {quantile: fitted_model} from build_quantile_models.
    X_cal : array-like
        Calibration features (held-out from training).
    y_cal : array-like
        Calibration targets.
    cqr_adjustment : float
        CQR adjustment for prediction intervals.
    thresholds : list of int
        DK point thresholds to calibrate (default: [5,10,15,20,25,30,35,40]).

    Returns
    -------
    calibrators : dict
        {threshold: fitted IsotonicRegression} for each threshold.
    report : dict
        {threshold: {brier_before, brier_after, calibration_error_before,
                     calibration_error_after}} for logging.
    """
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import norm

    if thresholds is None:
        thresholds = [5, 10, 15, 20, 25, 30, 35, 40]

    y_cal = np.asarray(y_cal)

    # Generate raw quantile predictions on calibration set
    quantile_keys = sorted(quantile_models.keys())
    preds = {}
    for q in quantile_keys:
        preds[q] = quantile_models[q].predict(X_cal)

    mid_q = quantile_keys[len(quantile_keys) // 2]
    median_pred = preds[mid_q]
    q_lo_pred = preds[quantile_keys[0]] - cqr_adjustment
    q_hi_pred = preds[quantile_keys[-1]] + cqr_adjustment
    std_est = np.clip((q_hi_pred - q_lo_pred) / 3.29, 0.1, None)

    print("\n===== PROBABILITY CALIBRATION (Isotonic Regression) =====")

    calibrators = {}
    report = {}

    for threshold in thresholds:
        actual_binary = (y_cal >= threshold).astype(float)
        base_rate = actual_binary.mean()

        # Skip if too few positive examples
        if actual_binary.sum() < 20:
            print(f"  {threshold}+ DK: skipped (only {int(actual_binary.sum())} positives)")
            continue

        # Raw probabilities (same formula as predict_with_quantiles)
        z = (threshold - median_pred) / std_est
        raw_prob = 1 - norm.cdf(z)

        # Brier score before calibration
        brier_before = ((raw_prob - actual_binary) ** 2).mean()

        # Fit isotonic regression: raw_prob → actual_binary
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(raw_prob, actual_binary)

        # Calibrated probabilities on same set (in-sample check)
        cal_prob = iso.predict(raw_prob)
        brier_after = ((cal_prob - actual_binary) ** 2).mean()

        # Calibration error (binned)
        cal_err_before = _binned_calibration_error(raw_prob, actual_binary)
        cal_err_after = _binned_calibration_error(cal_prob, actual_binary)

        calibrators[threshold] = iso
        report[threshold] = {
            'brier_before': brier_before,
            'brier_after': brier_after,
            'cal_error_before': cal_err_before,
            'cal_error_after': cal_err_after,
            'base_rate': base_rate,
        }

        brier_improve = (1 - brier_after / brier_before) * 100 if brier_before > 0 else 0
        print(f"  {threshold}+ DK: Brier {brier_before:.4f} → {brier_after:.4f} "
              f"({brier_improve:+.1f}%) | CalErr {cal_err_before:.1%} → {cal_err_after:.1%} "
              f"| base rate {base_rate:.1%}")

    print(f"  Calibrators fitted: {len(calibrators)}/{len(thresholds)}")
    return calibrators, report


def _binned_calibration_error(predicted_prob, actual_binary, n_bins=10):
    """Compute weighted mean absolute calibration error across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    total_error = 0.0
    total_n = 0
    for i in range(n_bins):
        mask = (predicted_prob >= bins[i]) & (predicted_prob < bins[i + 1])
        n = mask.sum()
        if n >= 10:
            pred_mean = predicted_prob[mask].mean()
            actual_rate = actual_binary[mask].mean()
            total_error += abs(pred_mean - actual_rate) * n
            total_n += n
    return total_error / total_n if total_n > 0 else 0.0
