"""
MLB DraftKings File Organization Script

This script organizes all MLB DraftKings files into a structured folder system.
Run this script to automatically move all files to their appropriate folders.
"""

import os
import shutil
from pathlib import Path

def organize_mlb_files():
    """
    Organize all MLB DraftKings files into structured folders
    """
    
    # Define source and destination paths
    source_dir = Path(r"c:\Users\smtes\Downloads\coinbase_ml_trader\app")
    dest_dir = Path(r"c:\Users\smtes\Downloads\coinbase_ml_trader\MLB_DRAFTKINGS_SYSTEM")
    
    # File organization mapping
    file_mapping = {
        # 1. CORE TRAINING FILES
        "1_CORE_TRAINING": [
            "training.py",
            "quick_param_search.py", 
            "quick_cv_search.py",
            "enhanced_auto_training.py",
            "fast_training.py",
            "optimized_training.py",
            "optimized_training_integration.py",
            "standalone_train_eval.py",
            "train_and_evaluate.py",
            "simple_train_eval.py",
            "quick_train_15.py",
            "quick_train_and_evaluate.py",
            "retrain_models.py",
            "training_monitor.py",
            "enhanced_features.py",
            "master_feature_engine.py",
            "efficient_preprocessing.py",
            "temporal_features.py",
            "cross_asset_features.py",
            "risk_features.py",
            "microstructure_features.py",
            "stacking_ml_engine.py",
            "ensemble_models.py",
            "ensemble_ml_integration.py",
            "ensemble_integration.py",
            "advanced_ensemble_ml.py",
            "advanced_ml_models.py",
            "advanced_time_series.py",
            "meta_classifier.py",
            "meta_integration.py",
            "meta_trading_classifier.py",
            "meta_trading_system.py",
            "stefan_jansen_improvements.py"
        ],
        
        # 2. PREDICTIONS
        "2_PREDICTIONS": [
            "predction.py",
            "predction01.py", 
            "predction01_walkforwardpredction.py",
            "predction_pyperpermatrization.py",
            "predict_day.py",
            "predict_interactive.py",
            "walkforward_predictor_refactored.py",
            "simple_walkforward_predictor.py",
            "quick_walk_forward_predictions.py",
            "synthetic_walk_forward_generator.py",
            "enhanced_ml_predictor.py",
            "optimized_prediction.py",
            "add_probability_predictions.py",
            "analyze_probabilities.py",
            "simple_probability_test.py",
            "calc_fpts.py",
            "calc_fpts_fixed.py",
            "improved_targets.py",
            "maybe.py",
            "maybe_backup.py",
            "maybe_fixed.py",
            "maybe.py.backup",
            "maybe.py.bak",
            "lk.py",
            "lk_improved.py",
            "lk_mock.py",
            "working.py"
        ],
        
        # 3. MODELS (trained models and encoders)
        "3_MODELS": [
            "batters_final_ensemble_model_pipeline.pkl",
            "batters_final_ensemble_model_pipeline01.pkl",
            "probability_predictor.pkl",
            "probability_predictor01.pkl",
            "trained_model.joblib",
            "quick_trained_model.joblib",
            "optimized_model.joblib",
            "model_metadata.joblib",
            "meta_trading_classifier.joblib",
            "processed_training_data.joblib",
            "label_encoder_name_sep2.pkl",
            "label_encoder_team_sep2.pkl",
            "scaler_sep2.pkl"
        ],
        
        # 4. DATA FILES
        "4_DATA": [
            "filtered_data.csv",
            "filtered_data1.CSV",
            "battersfinal_dataset_with_features.csv",
            "data_with_dk_entries_salaries.csv",
            "data_with_dk_entries_salaries.csv.backup_20250704_045724",
            "data_with_dk_entries_salaries_cleaned.csv",
            "data_with_dk_entries_salaries_cleaned_aggressive.csv",
            "data_with_dk_entries_salaries_final_cleaned.csv",
            "data_with_dk_entries_salaries_final_cleaned_with_pitchers.csv",
            "data_with_dk_entries_salaries_final_cleaned_with_pitchers_dk_fpts.csv",
            "data_with_realistic_salaries.csv",
            "merged_fangraphs_logs_with_fpts.csv",
            "merged_fangraphs_logs_with_fpts copy.csv",
            "merged_merged_fangraphs_logs_with_fpts_merged_fangraphs_data.csv",
            "merged_player_projections.csv",
            "merged_player_projections01.csv",
            "data_20210101_to_20250618.csv",
            "DD.csv",
            "x.csv",
            "asdf.csv",
            "data_retrival.py",
            "data_retrival_pitchers.py",
            "retrival.py",
            "create_merged_csv.py",
            "merge_fangraphs.py",
            "merge_projections.py",
            "simple_merge.py",
            "remove_injured_players.py",
            "simple_date_filter.py",
            "date_extractor.py",
            "check_columns.py",
            "validate_dk_entries_salaries.py"
        ],
        
        # 5. DRAFTKINGS ENTRIES
        "5_DRAFTKINGS_ENTRIES": [
            "DKEntries.csv",
            "DKEntries (1)_filled.csv",
            "DKEntries (1)_filled011.csv", 
            "DKEntries (1)_filled1111.csv",
            "DKEntries (1)_filled1236589.csv",
            "DKEntries (1)_filled369963.csv",
            "DKEntries (1)_filled455211337899446.csv",
            "DKEntries_FINAL_FILLED.csv",
            "test_alternative_dk.csv",
            "test_custom_dk.csv",
            "test_custom_entries.csv",
            "test_dk_entries.csv",
            "test_filled_entries_fixed.csv",
            "test_inconsistent_columns.csv",
            "test_mixed_dk.csv",
            "test_mixed_format.csv",
            "test_standard_dk.csv",
            "test_extra_commas.csv",
            "test_utf8_bom.csv",
            "test_with_headers.csv",
            "test_favorites_export.csv",
            "dk_entries_salary_generator.py",
            "dk_file_handler.py",
            "ppg_salary_generator.py",
            "realistic_salary_generator.py",
            "synthetic_salary_generator.py",
            "simple_dk_entries_demo.py",
            "enhanced_rl_demo_dk_entries.py",
            "test_dk_format.py"
        ],
        
        # 6. OPTIMIZATION
        "6_OPTIMIZATION": [
            "optimizer.py",
            "optimizer01.py",
            "optimizer01.backup.py",
            "optimizer_fixed.py",
            "optimizewr.backup.7-02-25.py",
            "backup_updated_7-04optimizer.py",
            "optimizer_input.csv",
            "optimizer_input_with_probabilities.csv",
            "pulp_lineup_optimizer.py",
            "practical_lineup_generator.py",
            "comprehensive_pulp_rl_demo.py",
            "rl_demo.py",
            "rl_team_selector.py",
            "enhanced_rl_demo.py",
            "realistic_rl_demo.py",
            "realistic_rl_system.py",
            "portfolio_optimization.py",
            "walkforward_rl_validator.py",
            "test_pulp_rl_integration.py",
            "run_rl_team_selector.py",
            "rl_config.py"
        ],
        
        # 7. ANALYSIS & EVALUATION
        "7_ANALYSIS": [
            "final_predictions.csv",
            "probability_predictions.csv",
            "feature_importances.csv",
            "feature_importances_plot.png",
            "evaluation_plots.png",
            "prediction_plots.png",
            "ml_evaluation_plots_20250610_173825.png",
            "evaluate_models.py",
            "evaluate_models_enhanced.py",
            "evaluate_predictions.py",
            "evaluate_ml_accuracy.py",
            "eval_existing_models.py",
            "eval_simple.py",
            "simple_evaluation.py",
            "simple_model_eval.py",
            "quick_evaluation_results_20250610_173549.csv",
            "quick_accuracy_check.py",
            "quick_eval.py",
            "quick_ensemble_test.py",
            "quick_model_test.py",
            "regression_eval.py",
            "model_evaluation_20250605_171748.csv",
            "model_evaluation_20250605_171748.json",
            "model_performance_comparison.py",
            "model_metrics_evaluation.py",
            "ensemble_model_evaluation.py",
            "ml_evaluation.py",
            "ml_evaluation_results_20250610_173824.csv",
            "existing_models_eval_20250605_173748.csv",
            "existing_models_eval_20250605_173936.csv",
            "existing_models_eval_20250605_174132.csv",
            "simple_eval_20250605_172302.csv",
            "trading_analysis_20250610_173607.csv",
            "simple_probability_test_results.csv",
            "analyze_data.py",
            "efficient_mae.py",
            "MAE.PY"
        ],
        
        # 8. DOCUMENTATION
        "8_DOCUMENTATION": [
            "TRAINING_INSTRUCTIONS.md",
            "COMPLETE_OPTIMIZATION_SUMMARY.md",
            "TIMESERIES_OPTIMIZATION_SUMMARY.md",
            "OPTIMIZATION_SUMMARY.md",
            "OPTIMIZATION_COMPLETE.md",
            "TRAINING_STATUS.md",
            "WALK_FORWARD_REFACTOR_SUMMARY.md",
            "PREDICTION_REFACTOR_SUMMARY.md",
            "DRAFTKINGS_FORMAT_FIX_SUMMARY.md",
            "PORTFOLIO_OPTIMIZATION_SOLUTION.md",
            "INTEGRATION_SUMMARY.md",
            "EVALUATION_GUIDE.md",
            "SIMPLE_BACKTESTING_GUIDE.md",
            "MLB_SEASON_DATES_IMPLEMENTATION.md",
            "optimal_parameters.txt",
            "optimal_cv_parameters.txt",
            "quick_model_features.txt",
            "ml_diagnosis_report.md",
            "ml_improvement_guide.py"
        ],
        
        # 9. BACKUP FILES
        "9_BACKUP": [
            "TRAINING.BACKUP.7-1-2025.PY",
            "training.backup.py",
            "training_backup_7-4.py",
            "TRAINING.O.PY",
            "TRAINING_DATABAASE_MLB.py",
            "optimization01.backup-07-03.py",
            "optimizer01.py.pre_fix_backup"
        ]
    }
    
    # Additional files with specific patterns
    prediction_patterns = [
        "batters_predictions_*.csv",
        "batters_probability_predictions_*.csv",
        "DFF_MLB_cheatsheet_*.csv",
        "predictions_*.csv",
        "synthetic_walk_forward_predictions_*.csv",
        "dfs_projections_with_probabilities.csv",
        "enhanced_projections_with_probabilities.csv"
    ]
    
    print("🏗️ ORGANIZING MLB DRAFTKINGS FILES")
    print("=" * 60)
    
    # Create all destination folders
    for folder in file_mapping.keys():
        folder_path = dest_dir / folder
        folder_path.mkdir(exist_ok=True)
        print(f"📁 Created/verified folder: {folder}")
    
    # Track moved files
    moved_files = 0
    missing_files = []
    
    # Move files based on mapping
    for folder, files in file_mapping.items():
        folder_path = dest_dir / folder
        print(f"\n📂 Moving files to {folder}...")
        
        for file in files:
            source_file = source_dir / file
            dest_file = folder_path / file
            
            if source_file.exists():
                try:
                    shutil.move(str(source_file), str(dest_file))
                    print(f"  ✅ Moved: {file}")
                    moved_files += 1
                except Exception as e:
                    print(f"  ❌ Error moving {file}: {e}")
            else:
                missing_files.append(file)
    
    # Move prediction files with patterns
    print(f"\n📊 Moving prediction files...")
    prediction_folder = dest_dir / "2_PREDICTIONS"
    
    for pattern in prediction_patterns:
        pattern_files = list(source_dir.glob(pattern))
        for file_path in pattern_files:
            dest_file = prediction_folder / file_path.name
            try:
                shutil.move(str(file_path), str(dest_file))
                print(f"  ✅ Moved: {file_path.name}")
                moved_files += 1
            except Exception as e:
                print(f"  ❌ Error moving {file_path.name}: {e}")
    
    # Summary
    print(f"\n🎉 ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Total files moved: {moved_files}")
    print(f"❌ Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  Missing files (already moved or not found):")
        for file in missing_files[:10]:  # Show first 10
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return moved_files, missing_files

if __name__ == "__main__":
    try:
        moved, missing = organize_mlb_files()
        
        print(f"\n📋 FOLDER STRUCTURE CREATED:")
        print("=" * 60)
        print("1_CORE_TRAINING     - Main training scripts and feature engineering")
        print("2_PREDICTIONS       - Prediction scripts and daily predictions")
        print("3_MODELS           - Trained models, encoders, and scalers")
        print("4_DATA             - Raw data files and data processing scripts")
        print("5_DRAFTKINGS_ENTRIES - DraftKings contest entries and generators")
        print("6_OPTIMIZATION     - Lineup optimizers and RL systems")
        print("7_ANALYSIS         - Model evaluation and analysis results")
        print("8_DOCUMENTATION    - All documentation and guides")
        print("9_BACKUP           - Backup files and old versions")
        
        print(f"\n🎯 NEXT STEPS:")
        print("=" * 60)
        print("1. Check the organized folders")
        print("2. Update any file paths in your scripts")
        print("3. Run training from 1_CORE_TRAINING/training.py")
        print("4. Generate predictions from 2_PREDICTIONS/")
        print("5. Check documentation in 8_DOCUMENTATION/")
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        print("Please check file permissions and paths")
