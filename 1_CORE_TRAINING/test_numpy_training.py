"""
Test script for the numpy-based training.py implementation
This script validates that all the custom implementations work correctly
"""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom implementations
from training import (
    StandardScalerNumPy,
    LabelEncoderNumPy,
    OneHotEncoderNumPy,
    LinearRegressionNumPy,
    DecisionTreeRegressorNumPy,
    GradientBoostingRegressorNumPy,
    StackingRegressorNumPy,
    VotingRegressorNumPy,
    calculate_metrics,
    impute_missing_values
)

def test_standard_scaler():
    """Test StandardScalerNumPy"""
    print("Testing StandardScalerNumPy...")
    
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    
    scaler = StandardScalerNumPy()
    X_scaled = scaler.fit_transform(X)
    
    # Check that mean is approximately 0 and std is approximately 1
    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10), "Mean should be 0"
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10), "Std should be 1"
    
    # Test inverse transform
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inverse, atol=1e-10), "Inverse transform should recover original data"
    
    print("  ✓ StandardScalerNumPy passed all tests")


def test_label_encoder():
    """Test LabelEncoderNumPy"""
    print("Testing LabelEncoderNumPy...")
    
    y = np.array(['cat', 'dog', 'cat', 'bird', 'dog', 'bird'])
    
    encoder = LabelEncoderNumPy()
    y_encoded = encoder.fit_transform(y)
    
    # Check that encoding is consistent
    assert y_encoded[0] == y_encoded[2], "Same labels should have same encoding"
    assert y_encoded[1] == y_encoded[4], "Same labels should have same encoding"
    
    # Test inverse transform
    y_decoded = encoder.inverse_transform(y_encoded)
    assert np.array_equal(y, y_decoded), "Inverse transform should recover original labels"
    
    print("  ✓ LabelEncoderNumPy passed all tests")


def test_one_hot_encoder():
    """Test OneHotEncoderNumPy"""
    print("Testing OneHotEncoderNumPy...")
    
    X = np.array(['cat', 'dog', 'cat', 'bird'])
    
    encoder = OneHotEncoderNumPy()
    X_encoded = encoder.fit_transform(X)
    
    # Check shape
    assert X_encoded.shape == (4, 3), "Shape should be (4, 3) for 4 samples and 3 categories"
    
    # Check that each row sums to 1
    assert np.allclose(np.sum(X_encoded, axis=1), 1), "Each row should sum to 1"
    
    print("  ✓ OneHotEncoderNumPy passed all tests")


def test_linear_regression():
    """Test LinearRegressionNumPy"""
    print("Testing LinearRegressionNumPy...")
    
    # Create simple linear data: y = 2*x + 3 + noise
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.flatten() + 3 + np.random.randn(100) * 0.5
    
    model = LinearRegressionNumPy(alpha=0.1, max_iter=2000, learning_rate=0.1)
    model.fit(X, y)
    
    # Predictions should be close to actual
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    assert mse < 1.0, f"MSE should be small, got {mse}"
    
    print(f"  ✓ LinearRegressionNumPy passed (MSE: {mse:.4f})")


def test_decision_tree():
    """Test DecisionTreeRegressorNumPy"""
    print("Testing DecisionTreeRegressorNumPy...")
    
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(200, 2) * 10
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(200) * 0.5
    
    model = DecisionTreeRegressorNumPy(max_depth=5, min_samples_split=10)
    model.fit(X, y)
    
    # Predictions should be reasonable
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    assert mse < 5.0, f"MSE should be reasonable, got {mse}"
    
    print(f"  ✓ DecisionTreeRegressorNumPy passed (MSE: {mse:.4f})")


def test_gradient_boosting():
    """Test GradientBoostingRegressorNumPy"""
    print("Testing GradientBoostingRegressorNumPy...")
    
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(200, 2) * 10
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(200) * 0.5
    
    model = GradientBoostingRegressorNumPy(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10
    )
    model.fit(X, y)
    
    # Predictions should be reasonable
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    assert mse < 5.0, f"MSE should be reasonable, got {mse}"
    
    print(f"  ✓ GradientBoostingRegressorNumPy passed (MSE: {mse:.4f})")


def test_voting_regressor():
    """Test VotingRegressorNumPy"""
    print("Testing VotingRegressorNumPy...")
    
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.5
    
    # Create base models
    models = [
        ('model1', LinearRegressionNumPy(alpha=0.1, max_iter=1000, learning_rate=0.1)),
        ('model2', LinearRegressionNumPy(alpha=1.0, max_iter=1000, learning_rate=0.1)),
    ]
    
    ensemble = VotingRegressorNumPy(models)
    ensemble.fit(X, y)
    
    # Predictions should be reasonable
    y_pred = ensemble.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    assert mse < 2.0, f"MSE should be reasonable, got {mse}"
    
    print(f"  ✓ VotingRegressorNumPy passed (MSE: {mse:.4f})")


def test_stacking_regressor():
    """Test StackingRegressorNumPy"""
    print("Testing StackingRegressorNumPy...")
    
    # Create simple data
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.5
    
    # Create base models and meta model
    base_models = [
        ('model1', LinearRegressionNumPy(alpha=0.1, max_iter=1000, learning_rate=0.1)),
        ('model2', LinearRegressionNumPy(alpha=1.0, max_iter=1000, learning_rate=0.1)),
    ]
    meta_model = LinearRegressionNumPy(alpha=0.5, max_iter=500, learning_rate=0.1)
    
    ensemble = StackingRegressorNumPy(base_models, meta_model)
    ensemble.fit(X, y)
    
    # Predictions should be reasonable
    y_pred = ensemble.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    assert mse < 2.0, f"MSE should be reasonable, got {mse}"
    
    print(f"  ✓ StackingRegressorNumPy passed (MSE: {mse:.4f})")


def test_calculate_metrics():
    """Test calculate_metrics function"""
    print("Testing calculate_metrics...")
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'mae' in metrics, "Metrics should contain MAE"
    assert 'mse' in metrics, "Metrics should contain MSE"
    assert 'rmse' in metrics, "Metrics should contain RMSE"
    assert 'r2' in metrics, "Metrics should contain R²"
    assert 'mape' in metrics, "Metrics should contain MAPE"
    
    # Check that metrics are reasonable
    assert metrics['mae'] > 0, "MAE should be positive"
    assert metrics['r2'] > 0.8, "R² should be high for this simple case"
    
    print(f"  ✓ calculate_metrics passed (R²: {metrics['r2']:.4f})")


def test_impute_missing_values():
    """Test impute_missing_values function"""
    print("Testing impute_missing_values...")
    
    X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, np.nan], [10, 11, 12]])
    
    X_imputed = impute_missing_values(X, strategy='mean')
    
    # Check that there are no NaN values
    assert not np.any(np.isnan(X_imputed)), "Should not contain NaN values"
    
    # Check that non-NaN values are unchanged
    assert X_imputed[0, 0] == 1, "Non-NaN values should be unchanged"
    assert X_imputed[0, 1] == 2, "Non-NaN values should be unchanged"
    
    print("  ✓ impute_missing_values passed all tests")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RUNNING NUMPY TRAINING IMPLEMENTATION TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_standard_scaler,
        test_label_encoder,
        test_one_hot_encoder,
        test_linear_regression,
        test_decision_tree,
        test_gradient_boosting,
        test_voting_regressor,
        test_stacking_regressor,
        test_calculate_metrics,
        test_impute_missing_values,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
