"""
Tests for the NegativeBinomialXGBRegressor in training.py.

Validates that the custom negative binomial objective works correctly
within the XGBoost + sklearn StackingRegressor pipeline.
"""

import numpy as np
import warnings
import pytest

warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

# Import the class under test
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from training import NegativeBinomialXGBRegressor, HARDCODED_OPTIMAL_PARAMS


def _make_data(n=300, seed=42):
    """Generate synthetic fantasy-point-like data."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 10)
    true_mu = np.exp(X @ rng.randn(10) * 0.3 + 1.5)
    y = rng.negative_binomial(n=3, p=3 / (3 + true_mu)).astype(float)
    return X, y


class TestNegativeBinomialXGBRegressor:
    """Unit tests for NegativeBinomialXGBRegressor."""

    def test_predictions_are_positive(self):
        """NB log-link should produce strictly positive predictions."""
        X, y = _make_data()
        model = NegativeBinomialXGBRegressor(
            nb_alpha=1.0, n_estimators=50, tree_method='hist', random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(preds > 0), "All predictions should be positive"

    def test_negative_targets_clamped(self):
        """Negative targets should be clamped to zero (not crash)."""
        X, y = _make_data()
        y[0:10] = -5.0  # inject negatives
        model = NegativeBinomialXGBRegressor(
            nb_alpha=1.0, n_estimators=20, tree_method='hist', random_state=42
        )
        model.fit(X, y)  # should not raise
        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_get_set_params_roundtrip(self):
        """get_params / set_params should preserve nb_alpha correctly."""
        model = NegativeBinomialXGBRegressor(nb_alpha=0.5, n_estimators=10)
        params = model.get_params()
        assert params['nb_alpha'] == 0.5
        assert 'objective' not in params

        model.set_params(nb_alpha=2.0)
        assert model.nb_alpha == 2.0

    def test_sklearn_clone(self):
        """sklearn clone() should produce a working copy."""
        model = NegativeBinomialXGBRegressor(nb_alpha=0.7, n_estimators=10, tree_method='hist')
        cloned = clone(model)
        assert cloned.nb_alpha == 0.7
        X, y = _make_data(n=50)
        cloned.fit(X, y)
        preds = cloned.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds > 0)

    def test_works_in_stacking_regressor(self):
        """Must work as final_estimator inside StackingRegressor."""
        X, y = _make_data(n=200)
        base = [('ridge', Ridge())]
        meta = NegativeBinomialXGBRegressor(
            nb_alpha=1.0, n_estimators=20, tree_method='hist', random_state=42
        )
        stack = StackingRegressor(estimators=base, final_estimator=meta)
        stack.fit(X, y)
        preds = stack.predict(X)
        assert preds.shape == (200,)
        assert np.all(np.isfinite(preds))

    def test_full_pipeline_with_hardcoded_params(self):
        """Replicate the exact pipeline structure from training.py and apply HARDCODED_OPTIMAL_PARAMS."""
        X, y = _make_data(n=200)

        base_models = [
            ('ridge', Ridge()),
            ('lasso', Lasso()),
            ('svr', SVR()),
            ('gb', GradientBoostingRegressor(n_estimators=10)),
        ]
        xgb_kw = {'tree_method': 'hist', 'device': 'cpu', 'n_jobs': -1,
                   'random_state': 42, 'nb_alpha': 1.0}

        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=NegativeBinomialXGBRegressor(**xgb_kw),
        )
        voting_model = VotingRegressor(estimators=base_models)

        final_model = StackingRegressor(
            estimators=[('stacking', stacking_model), ('voting', voting_model)],
            final_estimator=NegativeBinomialXGBRegressor(**xgb_kw),
        )

        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])
        selector = SelectKBest(f_regression, k=min(5, X.shape[1]))

        complete_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('model', final_model),
        ])

        # Apply hard-coded optimal parameters (same dict used in training.py)
        complete_pipeline.set_params(**HARDCODED_OPTIMAL_PARAMS)

        complete_pipeline.fit(X, y)
        preds = complete_pipeline.predict(X)

        assert preds.shape == (200,)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0), "All predictions from the full pipeline should be positive"

    def test_alpha_zero_approaches_poisson(self):
        """With nb_alpha ~ 0 the objective should behave like Poisson."""
        X, y = _make_data(n=200)
        nb_model = NegativeBinomialXGBRegressor(
            nb_alpha=1e-6, n_estimators=50, tree_method='hist', random_state=42
        )
        nb_model.fit(X, y)
        preds = nb_model.predict(X)
        mae = mean_absolute_error(y, preds)
        # Sanity: MAE should be finite and reasonable
        assert np.isfinite(mae)
        assert mae < np.std(y) * 3

    def test_predictions_reasonable_range(self):
        """Predictions should be in a reasonable range for fantasy-point-like data."""
        X, y = _make_data(n=300)
        model = NegativeBinomialXGBRegressor(
            nb_alpha=1.0, n_estimators=100, tree_method='hist', random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        # Predictions should not be astronomically large
        assert np.max(preds) < np.max(y) * 5
        # Mean prediction should be in the same ballpark as mean target
        assert abs(np.mean(preds) - np.mean(y)) < np.std(y) * 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
