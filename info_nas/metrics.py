"""Correlation metrics for ranking evaluation."""

from scipy.stats import kendalltau, pearsonr, spearmanr


def calculate_correlation(scores, performances):
    """Kendall's Tau-b correlation with p-value.

    Args:
        scores: Predicted scores.
        performances: Ground-truth performance values.

    Returns:
        (tau, p_value)
    """
    tau, p_value = kendalltau(scores, performances)
    return tau, p_value


def calculate_spearman_correlation(scores, performances):
    """Spearman rank correlation with p-value.

    Args:
        scores: Predicted scores.
        performances: Ground-truth performance values.

    Returns:
        (correlation, p_value)
    """
    correlation, p_value = spearmanr(scores, performances)
    return correlation, p_value


def calculate_pearson_correlation(scores, performances):
    """Pearson correlation with p-value.

    Args:
        scores: Predicted scores.
        performances: Ground-truth performance values.

    Returns:
        (correlation, p_value)
    """
    correlation, p_value = pearsonr(scores, performances)
    return correlation, p_value
