"""Info-NAS: A zero-cost proxy for Vision Transformer architecture search."""

__version__ = "1.0.0"

from info_nas.calculator import calculate_information_value_ratio
from info_nas.config import DEFAULT_PARAMS, PARAM_KEYS, param_list_to_dict
from info_nas.metrics import (
    calculate_correlation,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
)
from info_nas.utils import zero_score_combine, z_score_normalization

__all__ = [
    "calculate_information_value_ratio",
    "calculate_correlation",
    "calculate_spearman_correlation",
    "calculate_pearson_correlation",
    "z_score_normalization",
    "zero_score_combine",
    "PARAM_KEYS",
    "DEFAULT_PARAMS",
]
