"""
nba_training.py — NBA DFS Training Pipeline

Thin orchestrator that imports from:
  - nba_config.py          (CLI args, constants, feature lists)
  - nba_feature_engine.py  (3 feature classes, DK scoring, advanced stats)
  - model_builder.py       (ensemble, SHAP, Optuna, quantile models) — from 1_CORE_TRAINING
  - validator.py           (walk-forward CV, per-player eval, CQR)    — from 1_CORE_TRAINING

Usage:
  # First, scrape data:
  python 2_NBA_TRAINING/nba_scraper.py --output ./data/nba_game_logs.csv

  # Then train:
  python 2_NBA_TRAINING/nba_training.py --data-path ./data/nba_game_logs.csv --skip-hpo
  python 2_NBA_TRAINING/nba_training.py --data-path ./data/nba_game_logs.csv --n-splits 3
  python 2_NBA_TRAINING/nba_training.py --data-path ./data/nba_game_logs.csv --min-season 2014-15
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import joblib
import time
import os
import sys

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
warnings.filterwarnings('ignore', message='invalid value encountered')

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add this script's directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Add 1_CORE_TRAINING to path for model_builder and validator (sport-agnostic)
core_training_dir = os.path.join(os.path.dirname(script_dir), '1_CORE_TRAINING')
if core_training_dir not in sys.path:
    sys.path.insert(0, core_training_dir)

from nba_config import (
    parse_args, HARDCODED_OPTIMAL_PARAMS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    PREGAME_NUMERIC_FEATURES, COPULA_PREFIXES,
)
from nba_feature_engine import (
    compute_dk_fpts,
    compute_advanced_stats,
    engineer_features,
    reduce_copula_features,
    apply_marcel_shrinkage,
    prepare_pregame_features,
    NBAFinancialEngine,
    NBAProbabilisticEngine,
    NBACopulaEngine,
    set_feature_jobs,
)
from model_builder import (
    build_preprocessor,
    build_ensemble,
    build_feature_selector,
    build_quantile_models,
    optuna_hpo,
    save_shap_importance,
    IndexSelector,
)
from validator import (
    WalkForwardValidator,
    per_player_evaluation,
    conformalized_quantile_regression,
    predict_with_quantiles,
    fit_probability_calibrators,
)


def load_or_create_label_encoders(df, name_encoder_path, team_encoder_path):
    """Create and save label encoders for Name and Team columns."""
    le_name = LabelEncoder()
    le_name.fit(df['Name'])
    joblib.dump(le_name, name_encoder_path)

    le_team = LabelEncoder()
    le_team.fit(df['Team'])
    joblib.dump(le_team, team_encoder_path)

    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    return le_name, le_team


def main():
    # ---- 1. Parse CLI args ----
    args = parse_args()
    start_time = time.time()

    set_feature_jobs(args.feature_jobs)

    hw = args.hardware
    print(f"\n{'='*60}")
    print(f"NBA DFS TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"  CPU cores:      {hw['cpu_count']}")
    print(f"  RAM:            {hw['ram_gb']:.0f} GB" if hw['ram_gb'] else "  RAM:            unknown")
    print(f"  GPU:            {hw['gpu_name']} ({hw['gpu_mem_gb']:.1f} GB)" if hw['gpu_available'] else "  GPU:            none")
    print(f"  n_jobs:         {args.n_jobs}")
    print(f"  feature_jobs:   {args.feature_jobs}")
    print(f"  use_gpu:        {args.use_gpu}")
    print(f"  n_splits:       {args.n_splits}")
    print(f"  gap_days:       {args.gap_days}")
    print(f"{'='*60}\n")

    # ---- 2. Resolve paths ----
    output_dir = args.output_dir or os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    data_path = args.data_path
    if data_path is None:
        candidates = [
            os.path.join(os.path.dirname(script_dir), 'data', 'nba_game_logs.csv'),
            os.path.join(script_dir, 'nba_game_logs.csv'),
            os.path.join(output_dir, 'nba_game_logs.csv'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                data_path = candidate
                break
        if data_path is None:
            print("ERROR: No data path specified and no data file found.")
            print("Run: python 2_NBA_TRAINING/nba_scraper.py --output ./data/nba_game_logs.csv")
            print("Then: python 2_NBA_TRAINING/nba_training.py --data-path ./data/nba_game_logs.csv")
            sys.exit(1)

    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")

    # ---- 3. Load data ----
    print("\nLoading NBA game log dataset...")
    df = pd.read_csv(data_path, low_memory=False)

    # Standardize column names (nba_scraper.py already creates Name, Team, date)
    if 'PLAYER_NAME' in df.columns and 'Name' not in df.columns:
        df['Name'] = df['PLAYER_NAME']
    if 'TEAM_ABBREVIATION' in df.columns and 'Team' not in df.columns:
        df['Team'] = df['TEAM_ABBREVIATION']
    if 'GAME_DATE' in df.columns and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by=['Name', 'date'], inplace=True)

    # Coerce numeric columns
    numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA',
                    'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                    'OREB', 'DREB', 'PF', 'PLUS_MINUS', 'MIN', 'DD2', 'TD3']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse MIN if it's in "MM:SS" format
    if df['MIN'].dtype == object:
        def parse_minutes(val):
            if pd.isna(val):
                return 0.0
            val = str(val)
            if ':' in val:
                parts = val.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        df['MIN'] = df['MIN'].apply(parse_minutes)

    print(f"Loaded {len(df)} rows, date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Players: {df['Name'].nunique()}")
    print(f"Seasons: {df.get('SEASON_YEAR', pd.Series()).nunique() or 'N/A'}")

    # Season filter
    if args.min_season:
        if 'SEASON_YEAR' in df.columns:
            df = df[df['SEASON_YEAR'] >= args.min_season].reset_index(drop=True)
            print(f"After season filter (>= {args.min_season}): {len(df)} rows")

    # Filter out 0-minute games (DNPs)
    before = len(df)
    df = df[df['MIN'] > 0].reset_index(drop=True)
    print(f"Dropped {before - len(df)} zero-minute (DNP) rows → {len(df)} rows")

    # Filter players with too few games
    player_counts = df.groupby('Name')['date'].transform('count')
    before = len(df)
    df = df[player_counts >= args.min_games].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows from players with < {args.min_games} games → {len(df)} rows")

    # ---- 4. Compute DK fantasy points ----
    if 'calculated_dk_fpts' not in df.columns:
        print("Computing DraftKings NBA fantasy points...")
        df['calculated_dk_fpts'] = compute_dk_fpts(df)
    print(f"DK FPTS: mean={df['calculated_dk_fpts'].mean():.1f}, "
          f"median={df['calculated_dk_fpts'].median():.1f}, "
          f"max={df['calculated_dk_fpts'].max():.1f}")

    # ---- 5. Compute advanced stats ----
    print("\nComputing NBA advanced stats...")
    df = compute_advanced_stats(df)

    # Coerce object columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # ---- 6. Label encoders ----
    name_encoder_path = os.path.join(output_dir, 'label_encoder_name.pkl')
    team_encoder_path = os.path.join(output_dir, 'label_encoder_team.pkl')
    le_name, le_team = load_or_create_label_encoders(df, name_encoder_path, team_encoder_path)

    # ---- 7. Feature engineering (3 engines + advanced stats) ----
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING (3 engines)")
    print(f"{'='*60}")

    print("\nRunning financial-style feature engineering...")
    financial_engine = NBAFinancialEngine()
    df = financial_engine.calculate_features(df)

    print("\nRunning probabilistic feature engineering...")
    prob_engine = NBAProbabilisticEngine(lookback_window=30, min_observations=10)
    df = prob_engine.calculate_all_features(df)

    print("\nRunning advanced copula feature engineering...")
    copula_engine = NBACopulaEngine(lookback_window=30, min_observations=15)
    df = copula_engine.calculate_all_advanced_features(df)

    # Sabermetric-equivalent features (advanced stats + rolling DK points)
    df = engineer_features(df)

    # ---- 7b. Post-processing: copula PCA + Marcel shrinkage ----
    df = reduce_copula_features(df, n_components=8)
    df = apply_marcel_shrinkage(df, min_games=50)

    # Final cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    print(f"\nFeature engineering complete. Shape: {df.shape}")

    # ---- 8. Pre-game feature preparation (leakage-free) ----
    print(f"\n{'='*60}")
    print(f"PREGAME MODE: Using only pre-game features (leakage-free)")
    print(f"{'='*60}")
    df = prepare_pregame_features(df)

    numeric_features = [f for f in PREGAME_NUMERIC_FEATURES if f in df.columns]
    pgm_copula_pcs = [c for c in df.columns if c.startswith('pgm_copula_pc_')]
    numeric_features = list(set(numeric_features + pgm_copula_pcs))

    categorical_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    target = df['calculated_dk_fpts'].copy()
    features = df[numeric_features + categorical_features].copy()
    date_series = df['date'].copy()

    # Valid mask: require target + some lagged data
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

    # ---- 9. Walk-forward validation with in-fold preprocessing ----
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
    oos_path = os.path.join(output_dir, 'oos_validation_results.csv')
    metrics_df.to_csv(oos_path, index=False)
    print(f"\nSaved {oos_path}")

    if not oos_predictions_df.empty:
        player_eval_df = per_player_evaluation(oos_predictions_df, min_samples=10)
        player_eval_path = os.path.join(output_dir, 'player_evaluation.csv')
        player_eval_df.to_csv(player_eval_path, index=False)
        print(f"Saved {player_eval_path}")

    # ---- 10. Train final model on all data ----
    print(f"\n{'='*60}")
    print(f"FINAL MODEL TRAINING")
    print(f"{'='*60}")

    preprocessor = last_preprocessor
    selector = last_selector
    features_preprocessed = preprocessor.transform(features)
    features_selected = selector.transform(features_preprocessed)

    # HPO or hard-coded
    if not args.skip_hpo:
        print("\n===== OPTUNA HYPERPARAMETER OPTIMIZATION =====")
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
        print("Skipping HPO (--skip-hpo). Using hard-coded params.")
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

    # ---- 11. Quantile models ----
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
            calibrators, cal_report = fit_probability_calibrators(
                quantile_models, X_cal, y_cal, cqr_adjustment,
            )

    # ---- 12. Save artifacts ----
    print("\n===== SAVING ARTIFACTS =====")

    # Final predictions
    final_results_df = pd.DataFrame({
        'Name': df['Name'].values,
        'Date': date_series.values,
        'Actual': target.values,
        'Predicted': all_predictions,
    })
    final_results_df.to_csv(os.path.join(output_dir, 'final_predictions.csv'), index=False)
    print("Saved final_predictions.csv")

    # Model pipeline
    joblib.dump(complete_pipeline, os.path.join(output_dir, 'nba_model_pipeline.pkl'))
    print("Saved nba_model_pipeline.pkl")

    # Quantile models + calibrators
    if quantile_models is not None:
        quant_artifact = {
            'models': quantile_models,
            'cqr_adjustment': cqr_adjustment,
            'calibrators': calibrators,
        }
        joblib.dump(quant_artifact, os.path.join(output_dir, 'quantile_models.pkl'))
        print("Saved quantile_models.pkl")

        quantile_preds_df = predict_with_quantiles(
            quantile_models, features_selected, cqr_adjustment,
            calibrators=calibrators,
        )
        if quantile_preds_df is not None:
            final_with_probs = pd.concat([final_results_df.reset_index(drop=True),
                                          quantile_preds_df.reset_index(drop=True)], axis=1)
            final_with_probs.to_csv(
                os.path.join(output_dir, 'final_predictions_with_probabilities.csv'), index=False)
            print("Saved final_predictions_with_probabilities.csv")

    # OOS validation
    metrics_df.to_csv(os.path.join(output_dir, 'oos_validation_results.csv'), index=False)

    # Label encoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    print("Label encoders saved.")

    # Feature importances (SHAP-based)
    feature_importance_csv = os.path.join(output_dir, 'feature_importances.csv')
    feature_importance_plot = os.path.join(output_dir, 'feature_importances_plot.png')

    try:
        num_feat_names = numeric_features
        cat_feat_names = list(
            preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        )
        all_feat_names = num_feat_names + cat_feat_names
        selected_feat_names = [all_feat_names[i] for i in selector.selected_indices
                               if i < len(all_feat_names)]
    except Exception:
        selected_feat_names = [f'feature_{i}' for i in range(features_selected.shape[1])]

    try:
        base_model = final_ensemble.named_estimators_.get('lgbm') or \
                     final_ensemble.named_estimators_.get('gbr')
    except Exception:
        base_model = final_ensemble

    sample_size = min(5000, features_selected.shape[0])
    save_shap_importance(
        base_model,
        features_selected[:sample_size],
        selected_feat_names,
        feature_importance_csv,
        feature_importance_plot,
    )

    # Full dataset with features (for fast pregame re-training)
    final_dataset_path = os.path.join(output_dir, 'nba_dataset_with_features.csv')
    df.to_csv(final_dataset_path, index=False)
    print(f"Saved {final_dataset_path}")

    # Per-player quant profile
    QUANT_PROFILE_COLS = [
        'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
        'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
        'entropy', 'hurst_exponent', 'rolling_sharpe',
        'avg_player_correlation', 'correlation_volatility',
        'evt_return_level', 'exceedance_prob',
    ]
    available_quant_cols = [c for c in QUANT_PROFILE_COLS if c in df.columns]
    if available_quant_cols:
        quant_profile = (
            df.sort_values('date', ascending=False)
              .drop_duplicates('Name', keep='first')
              [['Name'] + available_quant_cols]
              .copy()
        )
        quant_profile_path = os.path.join(output_dir, 'player_quant_profile.csv')
        quant_profile.to_csv(quant_profile_path, index=False)
        print(f"Saved {quant_profile_path} ({len(quant_profile)} players, "
              f"{len(available_quant_cols)} quant columns)")

    # ---- Summary ----
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"NBA TRAINING PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Data: {len(df)} rows, {df['Name'].nunique()} players")
    print(f"Features selected: {features_selected.shape[1]}")
    print(f"Full-data MAE: {mae:.4f}  R2: {r2:.4f}")
    if not metrics_df.empty:
        print(f"OOS MAE (mean):  {metrics_df['mae'].mean():.4f} +/- {metrics_df['mae'].std():.4f}")
        print(f"OOS R2 (mean):   {metrics_df['r2'].mean():.4f} +/- {metrics_df['r2'].std():.4f}")
        print(f"Baseline MAE:    {metrics_df['baseline_mae'].mean():.4f}")
        print(f"\nREALITY CHECK:")
        print(f"  NBA DK FPTS have higher variance than MLB (~30 avg, 10-80 range)")
        print(f"  R2 ~0.05-0.20 = good honest model")
        print(f"  R2 > 0.50 = STILL LEAKING — investigate features")
        print(f"  MAE improvement over baseline = model adds value if negative")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
