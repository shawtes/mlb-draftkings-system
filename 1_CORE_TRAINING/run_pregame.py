"""
run_pregame.py — Fast pregame pipeline runner.

Loads the already-engineered dataset (battersfinal_dataset_with_features.csv),
creates pregame features, and runs walk-forward validation with in-fold
preprocessing. Skips the expensive 3-engine feature engineering.

Usage:
  python 1_CORE_TRAINING/run_pregame.py [--n-splits 5] [--skip-hpo]
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='invalid value encountered')

# Add script dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    parse_args, HARDCODED_OPTIMAL_PARAMS,
    PREGAME_NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    MATCHUP_ALL_FEATURES,
)
from feature_engine import prepare_pregame_features
from model_builder import (
    build_preprocessor, build_ensemble, build_feature_selector,
    build_quantile_models, save_shap_importance, IndexSelector,
)
from validator import (
    WalkForwardValidator, per_player_evaluation,
    conformalized_quantile_regression, predict_with_quantiles,
    fit_probability_calibrators,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def main():
    args = parse_args()
    start_time = time.time()

    hw = args.hardware
    print(f"\n{'='*60}")
    print(f"PREGAME PIPELINE (leakage-free)")
    print(f"{'='*60}")
    print(f"  CPU cores: {hw['cpu_count']}  |  n_jobs: {args.n_jobs}")
    print(f"  n_splits: {args.n_splits}  |  gap_days: {args.gap_days}")
    print(f"{'='*60}\n")

    output_dir = args.output_dir or os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load pre-engineered dataset ----
    data_path = args.data_path
    if data_path is None:
        data_path = os.path.join(output_dir, 'battersfinal_dataset_with_features.csv')
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading pre-engineered dataset: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(['Name', 'date'], inplace=True)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # ---- 1b. Matchup engine (if --matchup) ----
    if args.matchup:
        from matchup_engine import prepare_matchup_dataset
        pitcher_csv = args.pitcher_csv
        if pitcher_csv is None:
            data_dir = os.path.dirname(data_path)
            candidate = os.path.join(data_dir, 'merged_fangraphs_data_pitchers.csv')
            if os.path.exists(candidate):
                pitcher_csv = candidate
        cache_dir = args.matchup_cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(output_dir, 'matchup_cache')
        df = prepare_matchup_dataset(
            df, pitcher_csv=pitcher_csv, cache_dir=cache_dir,
        )
        print(f"After matchup engine: {df.shape}")

    # ---- 2. Prepare pregame features ----
    df = prepare_pregame_features(df)

    # ---- 3. Build feature matrix ----
    numeric_features = [f for f in PREGAME_NUMERIC_FEATURES if f in df.columns]
    pgm_copula_pcs = [c for c in df.columns if c.startswith('pgm_copula_pc_')]
    numeric_features = list(set(numeric_features + pgm_copula_pcs))
    # Add matchup features if enabled
    if args.matchup:
        matchup_in_df = [f for f in MATCHUP_ALL_FEATURES if f in df.columns]
        dynamic_matchup = [c for c in df.columns
                           if (c.startswith('opp_') or c.startswith('matchup_'))
                           and c not in numeric_features]
        numeric_features = list(set(numeric_features + matchup_in_df + dynamic_matchup))
        print(f"  Added {len(matchup_in_df) + len(dynamic_matchup)} matchup features")

    categorical_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    target = df['calculated_dk_fpts'].copy()
    features = df[numeric_features + categorical_features].copy()
    date_series = df['date'].copy()

    # Only require target + some lagged data; let preprocessor impute the rest
    pgm_cols = [c for c in numeric_features if c.startswith('pgm_') or c.startswith('lag_')]
    if pgm_cols:
        valid_mask = target.notna() & features[pgm_cols].notna().any(axis=1)
    else:
        valid_mask = target.notna()

    n_dropped = (~valid_mask).sum()
    print(f"Dropped {n_dropped} rows (missing target or no lagged data)")

    df = df[valid_mask].reset_index(drop=True)
    target = target[valid_mask].reset_index(drop=True)
    features = features[valid_mask].reset_index(drop=True)
    date_series = date_series[valid_mask].reset_index(drop=True)
    print(f"Dataset after filtering: {len(df)} rows")

    # ---- 4. Walk-forward validation with in-fold preprocessing ----
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION (IN-FOLD PREPROCESSING)")
    print(f"{'='*60}")

    wf_validator = WalkForwardValidator(n_splits=args.n_splits, gap_days=args.gap_days)

    def build_and_fit_fn(X_train, y_train):
        ensemble = build_ensemble(HARDCODED_OPTIMAL_PARAMS,
                                  use_gpu=args.use_gpu, n_jobs=args.n_jobs)
        ensemble.fit(X_train, y_train)
        return ensemble

    metrics_df, oos_predictions_df, (last_preprocessor, last_selector) = \
        wf_validator.validate_pregame(
            df=df,
            features_raw=features,
            y=target.values,
            build_preprocess_fn=build_preprocessor,
            build_model_fn=build_and_fit_fn,
            date_col='date',
            n_features=args.n_features,
        )

    # Save OOS results
    oos_path = os.path.join(output_dir, 'oos_validation_results_pregame.csv')
    metrics_df.to_csv(oos_path, index=False)
    print(f"\nSaved {oos_path}")

    if not oos_predictions_df.empty:
        player_eval_df = per_player_evaluation(oos_predictions_df, min_samples=10)
        player_eval_path = os.path.join(output_dir, 'player_evaluation_pregame.csv')
        player_eval_df.to_csv(player_eval_path, index=False)
        print(f"Saved {player_eval_path}")

    # ---- 5. Train final model on all data ----
    print(f"\n{'='*60}")
    print(f"FINAL MODEL TRAINING")
    print(f"{'='*60}")

    preprocessor = last_preprocessor
    selector = last_selector
    features_preprocessed = preprocessor.transform(features)
    features_selected = selector.transform(features_preprocessed)

    # HPO or hard-coded
    if not args.skip_hpo:
        from model_builder import optuna_hpo
        splits = list(wf_validator.split(df, 'date'))
        if splits:
            last_train_idx, last_test_idx = splits[-1]
            X_train_raw = features.iloc[last_train_idx]
            X_test_raw = features.iloc[last_test_idx]
            hpo_preprocessor = build_preprocessor(
                features.select_dtypes(include=[np.number]).columns.tolist(),
                features.select_dtypes(include=['object', 'category']).columns.tolist(),
            )
            X_train_proc = hpo_preprocessor.fit_transform(X_train_raw)
            X_test_proc = hpo_preprocessor.transform(X_test_raw)
            hpo_selector = build_feature_selector(
                X_train_proc, target.values[last_train_idx],
                n_features=args.n_features,
            )
            X_hpo_train = hpo_selector.transform(X_train_proc)
            X_hpo_val = hpo_selector.transform(X_test_proc)
            best_params = optuna_hpo(
                X_hpo_train, target.values[last_train_idx],
                X_hpo_val, target.values[last_test_idx],
                n_trials=args.optuna_trials,
                use_gpu=args.use_gpu, n_jobs=args.n_jobs,
            )
        else:
            best_params = HARDCODED_OPTIMAL_PARAMS
    else:
        best_params = HARDCODED_OPTIMAL_PARAMS

    final_ensemble = build_ensemble(best_params, use_gpu=args.use_gpu, n_jobs=args.n_jobs)
    final_ensemble.fit(features_selected, target.values)
    print("Final ensemble trained on all data.")

    # Build pipeline
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_ensemble),
    ])

    # Predictions on full data
    all_predictions = final_ensemble.predict(features_selected)
    mae = mean_absolute_error(target, all_predictions)
    r2 = r2_score(target, all_predictions)
    print(f"Full-data MAE: {mae:.4f}  R2: {r2:.4f}")

    # ---- 6. Quantile models ----
    print("\n===== QUANTILE REGRESSION =====")
    quantile_models = build_quantile_models(features_selected, target.values,
                                            use_gpu=args.use_gpu, n_jobs=args.n_jobs)

    cqr_adjustment = 0.0
    calibrators = None
    if quantile_models is not None:
        splits = list(wf_validator.split(df, 'date'))
        if splits:
            _, cal_idx = splits[-1]
            X_cal = features_selected[cal_idx]
            y_cal = target.values[cal_idx]
            cqr_adjustment, _ = conformalized_quantile_regression(
                quantile_models, X_cal, y_cal, alpha=0.10,
            )

            # Fit isotonic calibrators for probability thresholds
            calibrators, cal_report = fit_probability_calibrators(
                quantile_models, X_cal, y_cal, cqr_adjustment,
            )

    # ---- 7. Save artifacts ----
    print("\n===== SAVING ARTIFACTS =====")

    # Final predictions
    final_results_df = pd.DataFrame({
        'Name': df['Name'].values,
        'Date': date_series.values,
        'Actual': target.values,
        'Predicted': all_predictions,
    })
    final_results_df.to_csv(os.path.join(output_dir, 'final_predictions_pregame.csv'), index=False)
    # Also save to canonical name (replaces leaky model predictions)
    final_results_df.to_csv(os.path.join(output_dir, 'final_predictions.csv'), index=False)
    print(f"Saved final_predictions.csv + final_predictions_pregame.csv")

    # Model pipeline — save to both pregame-specific and canonical names
    joblib.dump(complete_pipeline, os.path.join(output_dir, 'pregame_model_pipeline.pkl'))
    joblib.dump(complete_pipeline, os.path.join(output_dir, 'batters_final_ensemble_model_pipeline.pkl'))
    print(f"Saved batters_final_ensemble_model_pipeline.pkl (PREGAME — replaces leaky model)")

    # Quantile models + calibrators
    if quantile_models is not None:
        quant_artifact = {
            'models': quantile_models,
            'cqr_adjustment': cqr_adjustment,
            'calibrators': calibrators,
        }
        joblib.dump(quant_artifact, os.path.join(output_dir, 'pregame_quantile_models.pkl'))
        joblib.dump(quant_artifact, os.path.join(output_dir, 'quantile_models.pkl'))
        print(f"Saved quantile_models.pkl (PREGAME + calibrators)")

        quantile_preds_df = predict_with_quantiles(
            quantile_models, features_selected, cqr_adjustment,
            calibrators=calibrators,
        )
        if quantile_preds_df is not None:
            final_with_probs = pd.concat([final_results_df.reset_index(drop=True),
                                          quantile_preds_df.reset_index(drop=True)], axis=1)
            final_with_probs.to_csv(os.path.join(output_dir, 'final_predictions_pregame_with_probabilities.csv'),
                                    index=False)
            final_with_probs.to_csv(os.path.join(output_dir, 'final_predictions_with_probabilities.csv'),
                                    index=False)

    # OOS validation results to canonical name
    metrics_df.to_csv(os.path.join(output_dir, 'oos_validation_results.csv'), index=False)

    print("All artifacts saved.")

    # ---- Summary ----
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PREGAME PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Data: {len(df)} rows")
    print(f"Features selected: {features_selected.shape[1]}")
    print(f"Full-data MAE: {mae:.4f}  R2: {r2:.4f}")
    if not metrics_df.empty:
        print(f"OOS MAE (mean):  {metrics_df['mae'].mean():.4f} +/- {metrics_df['mae'].std():.4f}")
        print(f"OOS R2 (mean):   {metrics_df['r2'].mean():.4f} +/- {metrics_df['r2'].std():.4f}")
        print(f"Baseline MAE:    {metrics_df['baseline_mae'].mean():.4f}")
        print(f"\nREALITY CHECK:")
        print(f"  R2 ~0.02-0.15 = good honest model")
        print(f"  R2 > 0.50 = STILL LEAKING — investigate features")
        print(f"  MAE improvement over baseline = model adds value if negative")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
