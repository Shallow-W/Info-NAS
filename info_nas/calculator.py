"""Core Info-NAS algorithm: information value ratio calculation."""

import numpy as np


def calculate_information_value_ratio(layer_num, mlp_ratios, num_heads, embed_dims, param):
    """Compute the information value ratio for a ViT architecture.

    This is the core zero-cost proxy. It calculates a performance score
    directly from architectural parameters (layer count, MLP ratio, head
    count, embedding dimension) using an information-theoretic formula,
    without any forward/backward pass.

    Args:
        layer_num: Number of transformer layers.
        mlp_ratios: MLP ratio per layer (list of length layer_num).
        num_heads: Number of attention heads per layer.
        embed_dims: Embedding dimension per layer.
        param: Dict mapping parameter names (from PARAM_KEYS) to float values.

    Returns:
        (final_ratio, b_ratio, grad_ratio, s_ratio) where:
            - final_ratio: combined score (higher = better predicted performance)
            - b_ratio: bottleneck component score
            - grad_ratio: gradient component score
            - s_ratio: stability component score

    Raises:
        ValueError: If input list lengths do not match layer_num.
    """
    if not (len(mlp_ratios) == layer_num and len(num_heads) == layer_num
            and len(embed_dims) == layer_num):
        raise ValueError("Input list lengths must equal layer_num")

    # --- Algorithm hyperparameters ---
    head_power = param.get("head_power", 5.53433166e+00)
    mlp_power = param.get("mlp_power", 1.50000000e+00)
    dim_power = param.get("dim_power", 2.08480912e+00)
    head_power_1 = param.get("head_power_1", 5.00000000e-01)
    head_power_2 = param.get("head_power_2", 4.93162294e+00)
    bottleneck_power_1 = param.get("bottleneck_power_1", 2.00000000e+01)
    bottleneck_power_2 = param.get("bottleneck_power_2", -3.01178552e+01)
    final_power = param.get("final_power", 1)
    grad_power = param.get("grad_power", 3.00000000e+09)
    s_power = param.get("s_power", 1.00000000e-02)

    total_info_value = []
    layer_score = []

    # Normalize dim_per_head
    all_dim_per_head = num_heads * 64
    max_dim_per_head = max(all_dim_per_head) if all_dim_per_head else 1

    for i in range(layer_num):
        # Original information amount
        denominator = 17 - i
        if denominator <= 0:
            denominator = 1
        original_info = np.power(2, 12 - (2 / denominator))
        layer_score.append(original_info)

        dim_per_head = all_dim_per_head[i]
        sequence_length = (224 * 224 * 3) / embed_dims[i]

        # Head dimension factor (normalized)
        normalized_dim_per_head = dim_per_head / max_dim_per_head
        head_dim_abs_factor = (1 + np.tanh(
            head_power_2 * (normalized_dim_per_head - head_power_1)
        ))
        head_dim_abs_factor = 1 + head_power_2 * np.log(num_heads[i] + head_power_1)

        # Bottleneck factor: penalizes when head dim << sequence length
        bottleneck_ratio = dim_per_head / sequence_length
        bottleneck_penalty = 1 / (
            1 + np.exp(bottleneck_power_2 * (bottleneck_ratio - bottleneck_power_1))
        )

        base = head_dim_abs_factor * bottleneck_penalty
        if base <= 0:
            base = 1e-6
        head_efficiency = base ** head_power

        # MLP efficiency
        mlp_efficiency = mlp_ratios[i] ** mlp_power

        # Embedding dimension factor
        if embed_dims[i] <= 0:
            embed_dims[i] = 1e-6
        dim_factor = np.log(embed_dims[i]) ** dim_power

        # Combined information utilization rate
        info_utilization_rate = dim_factor * (head_efficiency * mlp_efficiency)

        # Final information rate
        final_info_rate = original_info * info_utilization_rate * final_power
        total_info_value.append(final_info_rate)

    # --- Component scores ---
    b_ratio = np.sum(total_info_value)
    base_ratio = np.sum(total_info_value)

    # Gradient component (layer-to-layer information flow)
    grad_value = []
    for i in range(1, len(total_info_value) - 1):
        k1 = layer_score[i + 1]
        k2 = layer_score[i - 1]
        x = np.abs(k1 * total_info_value[i + 1] - k2 * total_info_value[i - 1])
        grad_value.append(x)
    grad_ratio = np.sum(grad_value)
    base_ratio += grad_power * grad_ratio

    # Stability component (cross-layer information consistency)
    s = []
    for i in range(len(total_info_value)):
        k1 = layer_score[i]
        for j in range(i + 1, len(total_info_value)):
            k2 = layer_score[j]
            score = np.abs(k1 * total_info_value[i] - k2 * total_info_value[j])
            s.append(score)
    s_ratio = np.sum(s)
    base_ratio += s_power * s_ratio

    final_ratio = base_ratio
    return final_ratio, b_ratio, grad_ratio, s_ratio
