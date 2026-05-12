"""Shared configuration constants for Info-NAS."""

# Names of the 10 trainable hyperparameters
PARAM_KEYS = [
    "head_power",
    "mlp_power",
    "dim_power",
    "head_power_1",
    "head_power_2",
    "bottleneck_power_1",
    "bottleneck_power_2",
    "b1_power",
    "g1_power",
    "s1_power",
]

# Default parameters (used for both tiny and small search spaces)
DEFAULT_PARAMS = [
    1.144643514073649,
    0.5606274602095567,
    2.1458925400545126,
    3.0298981004448753,
    4.405264203339544,
    0.10875665504668165,
    -27.196547689121708,
    10.5544027118415,
    1.211460571390166,
    -3.1005067530389736,
]


def param_list_to_dict(param_list):
    """Convert a parameter list to a dictionary using PARAM_KEYS."""
    return dict(zip(PARAM_KEYS, param_list))
