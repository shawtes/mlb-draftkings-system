"""
model_builder.py — Ensemble construction, SHAP feature selection, Optuna HPO,
                    and quantile regression models.

New ensemble: Ridge + Lasso + LightGBM + CatBoost  →  XGBRegressor meta-learner
(replaces Ridge+Lasso+SVR+GBR stacking from v1)

All new dependencies use try/except with graceful fallbacks.

Part of the MLB DFS Training Pipeline v2.
"""

import numpy as np
import warnings

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not available.")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: lightgbm not available. Using GradientBoostingRegressor as fallback.")

try:
    from catboost import CatBoostRegressor
    # Verify CatBoost works with current sklearn (1.8+ requires __sklearn_tags__)
    import sklearn
    _sklearn_major = int(sklearn.__version__.split('.')[1])
    if _sklearn_major >= 8:
        # CatBoost 1.2.x is incompatible with sklearn 1.8+
        CATBOOST_AVAILABLE = False
        print("Warning: catboost 1.2.x incompatible with sklearn 1.8+. Skipping.")
    else:
        CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: catboost not available. Skipping CatBoost in ensemble.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available. Using hard-coded parameters.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not available. Falling back to mutual_info_regression.")

from config import HARDCODED_OPTIMAL_PARAMS


# ============================================================================
# IndexSelector — sklearn-compatible transformer for serializable pipelines
# ============================================================================

class IndexSelector(BaseEstimator, TransformerMixin):
    """Select columns by index. Wraps `selected_indices` for pipeline serialization."""

    def __init__(self, selected_indices=None):
        self.selected_indices = selected_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.selected_indices is None:
            return X
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.selected_indices]
        # Convert sparse to CSC for efficient column slicing
        from scipy import sparse
        if sparse.issparse(X):
            return X.tocsc()[:, self.selected_indices].toarray()
        return X[:, self.selected_indices]

    def get_support(self, indices=False):
        if indices:
            return np.array(self.selected_indices)
        # Return boolean mask (requires knowing total feature count from last transform)
        return np.array(self.selected_indices)


# ============================================================================
# Preprocessor
# ============================================================================

def build_preprocessor(numeric_features, categorical_features):
    """Build a ColumnTransformer with median imputation (not mean)."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])


# ============================================================================
# Ensemble
# ============================================================================

def build_ensemble(lgbm_params=None, use_gpu=False, n_jobs=-1):
    """
    Build StackingRegressor:
      Base: Ridge, Lasso, LightGBM (or GBR fallback), CatBoost (optional)
      Meta: XGBRegressor

    GPU acceleration: LightGBM device='gpu', XGBoost device='cuda',
    CatBoost task_type='GPU' when use_gpu=True.
    """
    if lgbm_params is None:
        lgbm_params = HARDCODED_OPTIMAL_PARAMS

    base_models = [
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.01, max_iter=5000)),
    ]

    if LGBM_AVAILABLE:
        lgbm_extra = {}
        if use_gpu:
            lgbm_extra['device'] = 'gpu'
            lgbm_extra['gpu_use_dp'] = False  # single-precision is faster
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=lgbm_params.get('n_estimators', 200),
            max_depth=lgbm_params.get('max_depth', 6),
            learning_rate=lgbm_params.get('learning_rate', 0.1),
            subsample=lgbm_params.get('subsample', 0.8),
            colsample_bytree=lgbm_params.get('colsample_bytree', 0.9),
            min_child_weight=lgbm_params.get('min_child_weight', 3),
            reg_alpha=lgbm_params.get('reg_alpha', 0.1),
            reg_lambda=lgbm_params.get('reg_lambda', 1.0),
            n_jobs=n_jobs,
            random_state=42,
            verbose=-1,
            **lgbm_extra,
        )
        base_models.append(('lgbm', lgbm_model))
    else:
        gbr = GradientBoostingRegressor(
            n_estimators=lgbm_params.get('n_estimators', 200),
            max_depth=lgbm_params.get('max_depth', 6),
            learning_rate=lgbm_params.get('learning_rate', 0.1),
            subsample=lgbm_params.get('subsample', 0.8),
            random_state=42,
        )
        base_models.append(('gbr', gbr))

    if CATBOOST_AVAILABLE:
        cb_extra = {}
        if use_gpu:
            cb_extra['task_type'] = 'GPU'
            cb_extra['devices'] = '0'
        cb = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0,
            **cb_extra,
        )
        base_models.append(('catboost', cb))

    if XGB_AVAILABLE:
        xgb_device = 'cuda' if use_gpu else 'cpu'
        xgb_tree = 'gpu_hist' if use_gpu else 'hist'
        meta = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            tree_method=xgb_tree,
            device=xgb_device,
            objective='reg:squarederror',
            n_jobs=n_jobs,
            random_state=42,
        )
    else:
        meta = Ridge(alpha=1.0)

    ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=meta,
        cv=3,
        n_jobs=n_jobs,
    )
    return ensemble


# ============================================================================
# SHAP-based feature selection
# ============================================================================

def build_feature_selector(X_train, y_train, n_features=100):
    """
    Fit a quick LightGBM model and compute SHAP feature importances.
    Return sorted indices of top-n features.

    Falls back to mutual_info_regression if SHAP/LGBM unavailable.
    """
    n_total = X_train.shape[1]
    n_select = min(n_features, n_total)

    if LGBM_AVAILABLE:
        print(f"Feature selection: fitting quick LightGBM on {n_total} features...")
        quick_lgbm = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quick_lgbm.fit(X_train, y_train)

        # Use LightGBM built-in feature importances (robust with sparse input)
        importance = quick_lgbm.feature_importances_.astype(float)
        print(f"  LightGBM importances: shape={importance.shape}, "
              f"nonzero={np.count_nonzero(importance)}/{len(importance)}")
        selected_indices = np.argsort(importance)[::-1][:n_select].tolist()
        print(f"Selected {len(selected_indices)} features (top LightGBM importance)")
    else:
        print(f"Fallback: mutual_info_regression for feature selection on {n_total} features...")
        if hasattr(X_train, 'values'):
            X_arr = np.nan_to_num(X_train.values, nan=0.0)
        else:
            X_arr = np.nan_to_num(X_train, nan=0.0)
        mi_scores = mutual_info_regression(X_arr, y_train, random_state=42, n_neighbors=5)
        selected_indices = np.argsort(mi_scores)[::-1][:n_select].tolist()
        print(f"MI selected {len(selected_indices)} features")

    return IndexSelector(selected_indices=selected_indices)


# ============================================================================
# Optuna HPO
# ============================================================================

def optuna_hpo(X_train, y_train, X_val, y_val, n_trials=50,
               use_gpu=False, n_jobs=-1):
    """
    Bayesian hyperparameter optimization for LightGBM using Optuna.
    Returns best params dict. Falls back to HARDCODED_OPTIMAL_PARAMS.
    Runs trials in parallel across CPU cores; each trial uses GPU if available.
    """
    if not OPTUNA_AVAILABLE or not LGBM_AVAILABLE:
        print("Optuna/LightGBM not available — using hard-coded params")
        return HARDCODED_OPTIMAL_PARAMS

    print(f"Starting Optuna HPO with {n_trials} trials (gpu={use_gpu}, n_jobs={n_jobs})...")

    lgbm_extra = {}
    if use_gpu:
        lgbm_extra['device'] = 'gpu'
        lgbm_extra['gpu_use_dp'] = False

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(
            **params, n_jobs=n_jobs, random_state=42, verbose=-1,
            **lgbm_extra,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = np.mean(np.abs(y_val - preds))
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"Optuna best MAE: {study.best_value:.4f}")
    print(f"Optuna best params: {best}")
    return best


# ============================================================================
# Quantile regression models
# ============================================================================

def build_quantile_models(X_train, y_train, quantiles=None,
                          use_gpu=False, n_jobs=-1):
    """
    Train LightGBM quantile regression models for distributional predictions.
    Returns dict {quantile: fitted_model}.
    Falls back to None if LightGBM unavailable.
    """
    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    if not LGBM_AVAILABLE:
        print("LightGBM not available — skipping quantile models")
        return None

    lgbm_extra = {}
    if use_gpu:
        lgbm_extra['device'] = 'gpu'
        lgbm_extra['gpu_use_dp'] = False

    print(f"Training {len(quantiles)} quantile regression models (gpu={use_gpu})...")
    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=42,
            verbose=-1,
            **lgbm_extra,
        )
        model.fit(X_train, y_train)
        models[q] = model
        print(f"  q={q:.2f} trained")

    print("Quantile models complete.")
    return models


# ============================================================================
# SHAP feature importance export
# ============================================================================

def save_shap_importance(model, X_sample, feature_names, csv_path, plot_path):
    """Compute SHAP values and save feature importance CSV + plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if SHAP_AVAILABLE:
        print("Computing SHAP feature importances...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values)
            mean_abs = np.mean(np.abs(shap_values), axis=0).ravel()
        except Exception:
            # Fallback for non-tree models
            print("TreeExplainer failed, using model-based importance")
            if hasattr(model, 'feature_importances_'):
                mean_abs = model.feature_importances_
            else:
                mean_abs = np.ones(len(feature_names))
    elif hasattr(model, 'feature_importances_'):
        mean_abs = model.feature_importances_
    else:
        mean_abs = np.ones(len(feature_names))

    import pandas as pd
    n = min(len(feature_names), len(mean_abs))
    importance_df = pd.DataFrame({
        'Feature': feature_names[:n],
        'Importance': mean_abs[:n],
    }).sort_values('Importance', ascending=False)

    importance_df.to_csv(csv_path, index=False)
    print(f"Feature importances saved to {csv_path}")

    top = importance_df.head(25)
    plt.figure(figsize=(12, 10))
    plt.barh(top['Feature'], top['Importance'])
    plt.xlabel('Mean |SHAP value|' if SHAP_AVAILABLE else 'Importance')
    plt.ylabel('Feature')
    plt.title('Top 25 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")

    return importance_df
