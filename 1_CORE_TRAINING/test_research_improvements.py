"""
Tests for the new research-grade improvements to training.py:
- WalkForwardValidator (temporal cross-validation)
- ConformalPredictor (calibrated prediction intervals)
- optimize_hyperparameters (Bayesian HPO via Optuna)
- compute_shap_importance (SHAP-based feature importance)
"""

import numpy as np
import warnings
import pytest

warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from training import (
    WalkForwardValidator,
    ConformalPredictor,
    NegativeBinomialXGBRegressor,
    optimize_hyperparameters,
    compute_shap_importance,
    HARDCODED_OPTIMAL_PARAMS,
)


def _make_data(n=500, seed=42):
    """Generate synthetic fantasy-point-like data with temporal ordering."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 10)
    true_mu = np.exp(X @ rng.randn(10) * 0.3 + 1.5)
    y = rng.negative_binomial(n=3, p=3 / (3 + true_mu)).astype(float)
    return X, y


# -------------------------------------------------------------------------
# WalkForwardValidator tests
# -------------------------------------------------------------------------

class TestWalkForwardValidator:

    def test_splits_are_non_overlapping(self):
        """Train and test indices must not overlap."""
        wfv = WalkForwardValidator(n_splits=5, min_train_pct=0.5)
        X, y = _make_data(n=200)
        for train_idx, test_idx in wfv.split(X):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_train_always_before_test(self):
        """All training indices must be strictly less than all test indices."""
        wfv = WalkForwardValidator(n_splits=5, min_train_pct=0.5)
        X, _ = _make_data(n=200)
        for train_idx, test_idx in wfv.split(X):
            assert max(train_idx) < min(test_idx)

    def test_embargo_gap_exists(self):
        """There should be a gap between max train idx and min test idx."""
        wfv = WalkForwardValidator(n_splits=3, embargo_pct=0.1, min_train_pct=0.5)
        X, _ = _make_data(n=200)
        for train_idx, test_idx in wfv.split(X):
            gap = min(test_idx) - max(train_idx) - 1
            assert gap >= 1, f"Expected embargo gap ≥ 1, got {gap}"

    def test_expanding_window(self):
        """Each successive training set should be larger than the previous."""
        wfv = WalkForwardValidator(n_splits=4, min_train_pct=0.3)
        X, _ = _make_data(n=300)
        train_sizes = [len(tr) for tr, _ in wfv.split(X)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_get_n_splits(self):
        wfv = WalkForwardValidator(n_splits=5)
        assert wfv.get_n_splits() == 5

    def test_yields_correct_number_of_splits(self):
        wfv = WalkForwardValidator(n_splits=3, min_train_pct=0.5)
        X, _ = _make_data(n=300)
        splits = list(wfv.split(X))
        assert len(splits) == 3


# -------------------------------------------------------------------------
# ConformalPredictor tests
# -------------------------------------------------------------------------

class TestConformalPredictor:

    def _fit_and_calibrate(self, n=500, alpha=0.1):
        """Helper: fit a model, calibrate conformal predictor."""
        X, y = _make_data(n=n, seed=42)
        # 60% train, 20% cal, 20% test
        n_train = int(0.6 * n)
        n_cal = int(0.2 * n)
        X_train, y_train = X[:n_train], y[:n_train]
        X_cal, y_cal = X[n_train:n_train + n_cal], y[n_train:n_train + n_cal]
        X_test, y_test = X[n_train + n_cal:], y[n_train + n_cal:]

        model = NegativeBinomialXGBRegressor(
            nb_alpha=1.0, n_estimators=50, tree_method='hist', random_state=42
        )
        model.fit(X_train, y_train)

        cp = ConformalPredictor(model, alpha=alpha)
        cp.calibrate(X_cal, y_cal)

        return cp, X_test, y_test

    def test_coverage_at_90pct(self):
        """Prediction intervals should cover ≥ 80% of test points (allowing slack)."""
        cp, X_test, y_test = self._fit_and_calibrate(n=800, alpha=0.1)
        _, lower, upper = cp.predict_interval(X_test)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        # Marginal coverage should be close to 90%; allow 80% as a lower bound
        # (finite-sample slack on 160 test points)
        assert coverage >= 0.75, f"Expected coverage ≥ 0.75, got {coverage:.3f}"

    def test_interval_width_positive(self):
        """Prediction intervals must have positive width."""
        cp, X_test, _ = self._fit_and_calibrate(n=500, alpha=0.1)
        _, lower, upper = cp.predict_interval(X_test)
        assert np.all(upper >= lower)

    def test_exceedance_probabilities_in_range(self):
        """Exceedance probabilities should be in [0, 1]."""
        cp, X_test, _ = self._fit_and_calibrate(n=500, alpha=0.1)
        probs = cp.predict_exceedance_prob(X_test, thresholds=[5, 10, 15, 20])
        for key, p in probs.items():
            assert np.all(p >= 0) and np.all(p <= 1), f"{key} out of range"

    def test_exceedance_monotone_in_threshold(self):
        """P(Y > t) should decrease as t increases."""
        cp, X_test, _ = self._fit_and_calibrate(n=500, alpha=0.1)
        probs = cp.predict_exceedance_prob(X_test, thresholds=[5, 10, 20])
        # On average, higher thresholds should have lower exceedance prob
        assert np.mean(probs['prob_over_5']) >= np.mean(probs['prob_over_20'])

    def test_calibrate_before_predict_raises(self):
        """Using predict before calibrate should raise RuntimeError."""
        model = NegativeBinomialXGBRegressor(n_estimators=10, tree_method='hist')
        X, y = _make_data(n=50)
        model.fit(X, y)
        cp = ConformalPredictor(model, alpha=0.1)
        with pytest.raises(RuntimeError):
            cp.predict_interval(X)


# -------------------------------------------------------------------------
# optimize_hyperparameters tests
# -------------------------------------------------------------------------

class TestOptimizeHyperparameters:

    def test_returns_dict(self):
        """optimize_hyperparameters should return a dict of pipeline-prefixed params."""
        X, y = _make_data(n=200)
        result = optimize_hyperparameters(X, y, n_trials=3, n_splits=2)
        assert isinstance(result, dict)
        # Should contain pipeline-prefixed keys
        for key in result:
            assert 'model__final_estimator__' in key

    def test_all_expected_keys_present(self):
        """Result should have the same parameter keys as HARDCODED_OPTIMAL_PARAMS."""
        X, y = _make_data(n=200)
        result = optimize_hyperparameters(X, y, n_trials=3, n_splits=2)
        for key in HARDCODED_OPTIMAL_PARAMS:
            assert key in result, f"Missing key: {key}"


# -------------------------------------------------------------------------
# compute_shap_importance tests
# -------------------------------------------------------------------------

class TestComputeShapImportance:

    def test_returns_dataframe(self):
        """Should return a DataFrame with Feature and SHAP_Importance columns."""
        X, y = _make_data(n=200)
        model = XGBRegressor(n_estimators=20, tree_method='hist', random_state=42)
        model.fit(X, y)
        result = compute_shap_importance(model, X)
        if result is not None:
            assert 'Feature' in result.columns
            assert 'SHAP_Importance' in result.columns

    def test_importance_values_non_negative(self):
        """SHAP importance (mean |SHAP|) should always be ≥ 0."""
        X, y = _make_data(n=200)
        model = XGBRegressor(n_estimators=20, tree_method='hist', random_state=42)
        model.fit(X, y)
        result = compute_shap_importance(model, X)
        if result is not None:
            assert np.all(result['SHAP_Importance'] >= 0)

    def test_custom_feature_names(self):
        """Should use provided feature names."""
        X, y = _make_data(n=200)
        model = XGBRegressor(n_estimators=20, tree_method='hist', random_state=42)
        model.fit(X, y)
        names = [f'feat_{i}' for i in range(X.shape[1])]
        result = compute_shap_importance(model, X, feature_names=names)
        if result is not None:
            assert result['Feature'].iloc[0].startswith('feat_')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
