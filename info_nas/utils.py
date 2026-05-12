"""Utility functions for data normalization and scoring."""

import numpy as np


def z_score_normalization(data):
    """Z-Score normalization with NaN/Inf guards.

    Args:
        data: 1-D array-like of numeric values.

    Returns:
        (normalized_data, mean, std) where normalized_data is a numpy array.
    """
    data_np = np.array(data, dtype=np.float64)

    if np.any(np.isnan(data_np)) or np.any(np.isinf(data_np)):
        return np.zeros_like(data_np), 0.0, 1.0

    mean = np.mean(data_np)
    std = np.std(data_np)

    if std == 0 or np.isnan(std):
        return np.zeros_like(data_np), mean, 1.0

    return (data_np - mean) / std, mean, std


def zero_score_combine(b_values, grad_values, s_values, weights):
    """Z-score normalize three component arrays and compute weighted sum.

    Args:
        b_values: B (bottleneck) component scores.
        grad_values: G (gradient) component scores.
        s_values: S (stability) component scores.
        weights: Tuple/list of (b1_power, g1_power, s1_power).

    Returns:
        List of combined scores.
    """
    b_norm, _, _ = z_score_normalization(b_values)
    g_norm, _, _ = z_score_normalization(grad_values)
    s_norm, _, _ = z_score_normalization(s_values)

    b_w, g_w, s_w = weights
    return [
        b_w * b + g_w * g + s_w * s
        for b, g, s in zip(b_norm, g_norm, s_norm)
    ]
