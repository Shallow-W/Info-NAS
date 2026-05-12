"""Parameter optimization for Info-NAS using scipy.optimize."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize

from info_nas import (
    DEFAULT_PARAMS,
    PARAM_KEYS,
    calculate_correlation,
    calculate_information_value_ratio,
    z_score_normalization,
)
from info_nas.config import param_list_to_dict
from info_nas.evaluator import Evaluator


def calculate_scores(archs, param, weights):
    """Compute Info-NAS scores for a list of architectures."""
    b_values, grad_values, s_values = [], [], []
    for arch in archs:
        try:
            _, b_val, g_val, s_val = calculate_information_value_ratio(
                arch["layer_num"], arch["mlp_ratio"],
                arch["num_heads"], arch["embed_dim"], param
            )
            if np.isnan(b_val) or np.isinf(b_val):
                b_val = 0
            if np.isnan(g_val) or np.isinf(g_val):
                g_val = 0
            if np.isnan(s_val) or np.isinf(s_val):
                s_val = 0
        except Exception:
            b_val, g_val, s_val = 0, 0, 0
        b_values.append(b_val)
        grad_values.append(g_val)
        s_values.append(s_val)

    b_values, _, _ = z_score_normalization(b_values)
    grad_values, _, _ = z_score_normalization(grad_values)
    s_values, _, _ = z_score_normalization(s_values)

    b_w, g_w, s_w = weights
    scores = [b_w * b + g_w * g + s_w * s
              for b, g, s in zip(b_values, grad_values, s_values)]
    return scores


def loss_function(param_values, param_keys, archs, true_values):
    """Loss function: minimize negative Kendall's tau."""
    if np.any(np.isnan(param_values)) or np.any(np.isinf(param_values)):
        return 1.0

    param = dict(zip(param_keys, param_values))
    weights = param_values[-3:]

    try:
        preds = calculate_scores(archs, param, weights)
        preds = np.array(preds)
        true_arr = np.array(true_values)

        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            return 1.0
        if np.std(preds) < 1e-10:
            return 1.0

        tau, _ = calculate_correlation(preds.tolist(), true_arr.tolist())
        if np.isnan(tau):
            return 1.0
        return -tau
    except Exception:
        return 1.0


def train(dataset="Tiny"):
    """Fit optimal hyperparameters by maximizing Kendall's tau."""
    evaluator = Evaluator(is_zero_score=True)
    data_all = evaluator.load_ood_data(dataset)
    if data_all is None:
        print("Data loading failed, exiting")
        return None
    data_all = list(data_all.values())

    train_archs = [s["net_setting"] for s in data_all]
    train_performances = [s["performance"]["Imagenet"]["clean"] for s in data_all]
    print(f"Dataset: {dataset}, {len(train_archs)} samples")

    init_params = np.array(DEFAULT_PARAMS, dtype=np.float64)

    init_loss = loss_function(init_params, PARAM_KEYS, train_archs, train_performances)
    print(f"Initial loss: {init_loss:.6f} (Kendall tau: {-init_loss:.6f})")

    print("Optimizing...")
    res = minimize(
        loss_function, init_params,
        args=(PARAM_KEYS, train_archs, train_performances),
        method="Powell",
        options={"disp": True, "xtol": 1e-6, "ftol": 1e-6, "maxiter": 1000},
    )

    print("\n" + "=" * 50)
    print("Optimization complete!")
    print(f"Optimal params: {res.x.tolist()}")
    print(f"Max Kendall tau: {-res.fun:.6f}")
    print("=" * 50)

    return res.x.tolist()


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "Tiny"
    train(dataset=dataset)
