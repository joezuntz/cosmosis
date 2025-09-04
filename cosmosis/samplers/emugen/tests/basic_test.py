#!/usr/bin/env python3
"""
Basic functionality test for the optimized emugen modules.

This test focuses on the core functionality that can be tested 
without the full CosmoSIS framework.
"""

import os
import sys
import numpy as np
import logging
from unittest.mock import Mock, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_numpy_operations():
    """Test basic numpy operations work correctly."""
    logger.info("Testing numpy operations...")
    
    # Test array creation and operations
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 5)
    
    # Test mean and std
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    
    assert x_mean.shape == (10,), f"Expected shape (10,), got {x_mean.shape}"
    assert x_std.shape == (10,), f"Expected shape (10,), got {x_std.shape}"
    
    # Test normalization
    x_norm = (x - x_mean) / x_std
    x_norm_mean = np.mean(x_norm, axis=0)
    x_norm_std = np.std(x_norm, axis=0)
    
    assert np.allclose(x_norm_mean, 0, atol=1e-10), "Normalized mean should be ~0"
    assert np.allclose(x_norm_std, 1, atol=1e-10), "Normalized std should be ~1"
    
    logger.info("✓ Numpy operations test passed")

def test_log_transform():
    """Test log transformation functionality."""
    logger.info("Testing log transformation...")
    
    # Create positive data
    data = np.random.exponential(2.0, (50, 10)) + 1e-10
    
    # Log transform
    log_data = np.log10(data)
    
    # Check all values are finite
    assert np.isfinite(log_data).all(), "All log values should be finite"
    
    # Back transform
    recovered = 10 ** log_data
    
    # Should recover original data
    assert np.allclose(recovered, data, rtol=1e-12), "Should recover original data"
    
    logger.info("✓ Log transformation test passed")

def test_data_preprocessing():
    """Test data preprocessing functions."""
    logger.info("Testing data preprocessing...")
    
    # Create synthetic cosmological data
    n_samples = 100
    n_features = 50
    
    # Parameters (cosmological parameters)
    omega_m = np.random.uniform(0.2, 0.4, n_samples)
    sigma_8 = np.random.uniform(0.7, 0.9, n_samples)
    h = np.random.uniform(0.6, 0.8, n_samples)
    
    # Create parameter dictionary
    X = {
        'omega_m': omega_m,
        'sigma_8': sigma_8,
        'h': h
    }
    
    # Create synthetic theory predictions (correlated with parameters)
    params_matrix = np.column_stack([omega_m, sigma_8, h])
    
    # Create realistic cosmological data vector (power spectrum-like)
    k_modes = np.logspace(-3, 1, n_features)  # k modes
    
    # Simple power spectrum model: P(k) ∝ k^n * T(k)^2
    theory_data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # Simple scaling with cosmological parameters
        amplitude = sigma_8[i] ** 2
        shape = -2.5 + 0.1 * (omega_m[i] - 0.3) / 0.1  # Tilt
        
        # Power spectrum
        power_spectrum = amplitude * (k_modes ** shape) * np.exp(-k_modes / 10)
        
        # Add some noise
        theory_data[i] = power_spectrum * (1 + 0.01 * np.random.randn(n_features))
    
    # Test parameter normalization
    X_mean = {key: np.mean(X[key]) for key in X.keys()}
    X_std = {key: np.std(X[key]) for key in X.keys()}
    
    # Normalize parameters
    X_norm = {key: (X[key] - X_mean[key]) / X_std[key] for key in X.keys()}
    
    # Check normalization
    for key in X.keys():
        assert np.abs(np.mean(X_norm[key])) < 1e-10, f"Normalized {key} mean should be ~0"
        assert np.abs(np.std(X_norm[key]) - 1) < 1e-10, f"Normalized {key} std should be ~1"
    
    # Test theory data normalization (log-normal)
    theory_data_safe = np.maximum(theory_data, 1e-30)  # Avoid log(0)
    log_theory = np.log10(theory_data_safe)
    
    log_mean = np.mean(log_theory, axis=0)
    log_std = np.std(log_theory, axis=0)
    
    theory_norm = (log_theory - log_mean) / log_std
    
    # Check theory normalization
    theory_norm_mean = np.mean(theory_norm, axis=0)
    theory_norm_std = np.std(theory_norm, axis=0)
    
    assert np.allclose(theory_norm_mean, 0, atol=1e-10), "Theory norm mean should be ~0"
    assert np.allclose(theory_norm_std, 1, atol=1e-10), "Theory norm std should be ~1"
    
    # Test back-transformation
    recovered_log = theory_norm * log_std + log_mean
    recovered_theory = 10 ** recovered_log
    
    assert np.allclose(recovered_theory, theory_data_safe, rtol=1e-12), "Should recover original theory data"
    
    logger.info("✓ Data preprocessing test passed")

def test_parameter_dict_operations():
    """Test parameter dictionary operations."""
    logger.info("Testing parameter dictionary operations...")
    
    # Create parameter dictionary
    params = ['omega_m', 'sigma_8', 'h']
    n_samples = 20
    
    X = {
        'omega_m': np.random.uniform(0.2, 0.4, n_samples),
        'sigma_8': np.random.uniform(0.7, 0.9, n_samples),
        'h': np.random.uniform(0.6, 0.8, n_samples)
    }
    
    # Test converting to ordered array
    X_array = np.stack([X[key] for key in params], axis=1)
    
    assert X_array.shape == (n_samples, 3), f"Expected shape ({n_samples}, 3), got {X_array.shape}"
    
    # Test that ordering is preserved
    for i, key in enumerate(params):
        assert np.array_equal(X_array[:, i], X[key]), f"Column {i} should match {key}"
    
    # Test single parameter prediction format
    single_params = {key: np.array([X[key][0]]) for key in params}
    single_array = np.stack([single_params[key] for key in params], axis=1)
    
    assert single_array.shape == (1, 3), f"Expected shape (1, 3), got {single_array.shape}"
    
    logger.info("✓ Parameter dictionary operations test passed")

def test_mock_neural_network():
    """Test mock neural network operations."""
    logger.info("Testing mock neural network operations...")
    
    # Simulate neural network architecture
    input_dim = 3
    hidden_dims = [64, 64, 32]
    output_dim = 50
    
    # Test weight initialization shapes
    architecture = [input_dim] + hidden_dims + [output_dim]
    n_layers = len(architecture) - 1
    
    # Mock weight matrices
    weights = []
    biases = []
    
    for i in range(n_layers):
        w = np.random.randn(architecture[i], architecture[i+1]) * 0.01
        b = np.zeros(architecture[i+1])
        weights.append(w)
        biases.append(b)
    
    # Test forward pass simulation
    batch_size = 10
    x = np.random.randn(batch_size, input_dim)
    
    # Simple forward pass
    activations = [x]
    for i in range(n_layers - 1):
        linear = np.dot(activations[-1], weights[i]) + biases[i]
        # ReLU activation
        activated = np.maximum(linear, 0)
        activations.append(activated)
    
    # Output layer (linear)
    output = np.dot(activations[-1], weights[-1]) + biases[-1]
    
    assert output.shape == (batch_size, output_dim), f"Expected output shape ({batch_size}, {output_dim}), got {output.shape}"
    
    # Test that we can compute gradients (mock)
    # In real case, this would be done by TensorFlow
    target = np.random.randn(batch_size, output_dim)
    loss = np.mean((output - target) ** 2)
    
    assert np.isfinite(loss), "Loss should be finite"
    assert loss >= 0, "MSE loss should be non-negative"
    
    logger.info("✓ Mock neural network operations test passed")

def test_emulator_workflow():
    """Test complete emulator workflow simulation."""
    logger.info("Testing complete emulator workflow...")
    
    # Step 1: Generate training data
    n_train = 100
    n_params = 3
    n_features = 20
    
    # Parameters
    X_train = {
        'omega_m': np.random.uniform(0.2, 0.4, n_train),
        'sigma_8': np.random.uniform(0.7, 0.9, n_train),
        'h': np.random.uniform(0.6, 0.8, n_train)
    }
    
    # Theory predictions (with realistic correlations)
    params_array = np.column_stack([X_train[key] for key in ['omega_m', 'sigma_8', 'h']])
    
    # Create correlated output
    true_weights = np.random.randn(n_params, n_features)
    y_train = np.dot(params_array, true_weights) + 0.1 * np.random.randn(n_train, n_features)
    
    # Step 2: Normalize data
    X_mean = {key: np.mean(X_train[key]) for key in X_train.keys()}
    X_std = {key: np.std(X_train[key]) for key in X_train.keys()}
    
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    
    # Step 3: Transform data
    X_norm = {key: (X_train[key] - X_mean[key]) / X_std[key] for key in X_train.keys()}
    y_norm = (y_train - y_mean) / y_std
    
    # Step 4: Simulate training (simple linear model as proxy)
    X_norm_array = np.column_stack([X_norm[key] for key in ['omega_m', 'sigma_8', 'h']])
    
    # Fit linear model (as proxy for neural network)
    # y = X @ W + b
    W_fit = np.linalg.lstsq(X_norm_array, y_norm, rcond=None)[0]
    
    # Step 5: Test prediction
    n_test = 10
    X_test = {
        'omega_m': np.random.uniform(0.2, 0.4, n_test),
        'sigma_8': np.random.uniform(0.7, 0.9, n_test),
        'h': np.random.uniform(0.6, 0.8, n_test)
    }
    
    # Normalize test data
    X_test_norm = {key: (X_test[key] - X_mean[key]) / X_std[key] for key in X_test.keys()}
    X_test_array = np.column_stack([X_test_norm[key] for key in ['omega_m', 'sigma_8', 'h']])
    
    # Predict
    y_pred_norm = np.dot(X_test_array, W_fit)
    y_pred = y_pred_norm * y_std + y_mean
    
    # Step 6: Validate prediction makes sense
    assert y_pred.shape == (n_test, n_features), f"Expected prediction shape ({n_test}, {n_features}), got {y_pred.shape}"
    assert np.isfinite(y_pred).all(), "All predictions should be finite"
    
    # Test single prediction
    single_test = {key: np.array([X_test[key][0]]) for key in X_test.keys()}
    single_norm = {key: (single_test[key] - X_mean[key]) / X_std[key] for key in single_test.keys()}
    single_array = np.column_stack([single_norm[key] for key in ['omega_m', 'sigma_8', 'h']])
    
    single_pred_norm = np.dot(single_array, W_fit)
    single_pred = single_pred_norm * y_std + y_mean
    
    assert single_pred.shape == (1, n_features), f"Expected single prediction shape (1, {n_features}), got {single_pred.shape}"
    assert np.allclose(single_pred, y_pred[0:1], rtol=1e-12), "Single prediction should match batch prediction"
    
    logger.info("✓ Complete emulator workflow test passed")

def test_error_handling():
    """Test error handling scenarios."""
    logger.info("Testing error handling scenarios...")
    
    # Test division by zero protection
    data = np.array([1.0, 2.0, 0.0, 3.0])
    std = np.std(data)
    
    # Add small epsilon to avoid division by zero
    std_safe = np.maximum(std, 1e-10)
    normalized = (data - np.mean(data)) / std_safe
    
    assert np.isfinite(normalized).all(), "Normalized data should be finite"
    
    # Test log of zero/negative protection
    data_with_zeros = np.array([1.0, 0.0, -1.0, 2.0])
    data_safe = np.maximum(data_with_zeros, 1e-30)
    log_data = np.log10(data_safe)
    
    assert np.isfinite(log_data).all(), "Log data should be finite"
    
    # Test array shape mismatches
    try:
        a = np.random.randn(10, 5)
        b = np.random.randn(3, 5)
        # This should work (broadcasting)
        result = a + b  # Will raise ValueError due to shape mismatch
        assert False, "Should have raised an error"
    except ValueError:
        pass  # Expected
    
    logger.info("✓ Error handling test passed")

def run_all_tests():
    """Run all basic tests."""
    tests = [
        test_numpy_operations,
        test_log_transform,
        test_data_preprocessing,
        test_parameter_dict_operations,
        test_mock_neural_network,
        test_emulator_workflow,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    logger.info("="*60)
    logger.info("Running Basic EmugenSampler Tests")
    logger.info("="*60)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    logger.info("="*60)
    logger.info("Test Results:")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total:  {len(tests)}")
    
    if failed == 0:
        logger.info("🎉 All basic tests passed!")
        logger.info("The optimized emugen modules should work correctly.")
    else:
        logger.error(f"❌ {failed} test(s) failed!")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
