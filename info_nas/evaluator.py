"""Evaluator: data loading, scoring, and correlation analysis for Info-NAS."""

import json
import time

import numpy as np

from info_nas.calculator import calculate_information_value_ratio
from info_nas.config import DEFAULT_PARAMS, PARAM_KEYS, param_list_to_dict
from info_nas.metrics import (
    calculate_correlation,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
)
from info_nas.utils import z_score_normalization, zero_score_combine


class Evaluator:
    """Load architecture datasets, score them with Info-NAS, and compute correlations."""

    def __init__(self, param=None, is_zero_score=False):
        """Initialize the evaluator.

        Args:
            param: List of 10 float hyperparameters. If None, uses DEFAULT_PARAMS.
            is_zero_score: If True, combine B/G/S components with z-score normalization.
        """
        self.is_zero_score = is_zero_score
        self.param = list(param) if param is not None else list(DEFAULT_PARAMS)

    @property
    def param_dict(self):
        return param_list_to_dict(self.param)

    @property
    def weights(self):
        """Return the last 3 parameters as fusion weights (b1, g1, s1)."""
        return self.param[-3:]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_ood_data(self, dataset_type):
        """Load OoD-ViT-NAS benchmark (Tiny/Small/Base)."""
        file_path = f"./data/ood/OoD-ViT-NAS-{dataset_type}.json"
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            print(f"Loaded OoD dataset: {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
            return None

    def load_bench_data(self, file_names=None):
        """Load Bench dataset (.pth files)."""
        if file_names is None:
            file_names = ["gt_autoformer.pth"]
        if isinstance(file_names, str):
            file_names = [file_names]

        all_data = []
        for file_name in file_names:
            try:
                file_path = f"./data/bench/{file_name}"
                import torch
                data = torch.load(file_path, weights_only=False)
                print(f"Loaded Bench dataset: {file_path}")
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except FileNotFoundError:
                print(f"Error: {file_path} not found.")

        return all_data if all_data else None

    def load_our_data(self, dataset_name, scale="small"):
        """Load custom benchmark dataset.

        Args:
            dataset_name: e.g. 'cifar-10_tiny', 'cifar-100_small'.
            scale: 'tiny' or 'small', selects ViT-Info-Bench_{scale}.json.
        """
        file_path = f"./data/our/ViT-Info-Bench_{scale}.json"
        print(f"Loading Our dataset: {file_path}")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            print(f"Loaded Our dataset: {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
            return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_arch(self, arch_config):
        """Score a single architecture config dict.

        Args:
            arch_config: Dict with keys 'layer_num', 'mlp_ratio', 'num_heads', 'embed_dim'.

        Returns:
            (ratio, b_value, g_value, s_value)
        """
        return calculate_information_value_ratio(
            arch_config["layer_num"],
            arch_config["mlp_ratio"],
            arch_config["num_heads"],
            arch_config["embed_dim"],
            self.param_dict,
        )

    # ------------------------------------------------------------------
    # Dataset evaluation (OoD)
    # ------------------------------------------------------------------

    def evaluate_ood(self, data):
        """Evaluate Info-NAS on an OoD dataset.

        Returns:
            (scores, performances, b_values, g_values, s_values, archs)
        """
        archs, info_value_ratios, performances = [], [], []
        b_values, g_values, s_values = [], [], []

        for sample in data.values():
            arch = sample["net_setting"]
            performance = sample["performance"]["Imagenet"]["clean"]
            archs.append(arch)

            ratio, b_value, g_value, s_value = self.score_arch(arch)
            b_values.append(b_value)
            g_values.append(g_value)
            s_values.append(s_value)
            info_value_ratios.append(ratio)
            performances.append(performance)

        if self.is_zero_score:
            scores = zero_score_combine(b_values, g_values, s_values, self.weights)
        else:
            scores = info_value_ratios

        return scores, performances, b_values, g_values, s_values, archs

    # ------------------------------------------------------------------
    # Dataset evaluation (Bench)
    # ------------------------------------------------------------------

    def evaluate_bench(self, data, metric_key="c100_kd_acc"):
        """Evaluate Info-NAS on a Bench dataset (.pth format).

        Returns:
            (scores, performances, b_values, g_values, s_values, archs)
        """
        archs, info_value_ratios, performances = [], [], []
        b_values, g_values, s_values = [], [], []

        for i in range(len(data) - 1):
            sample = data[i]
            arch = sample["arch"]
            layer_num = arch["depth"]
            mlp_ratios = arch["mlp_ratio"]
            num_heads = arch["num_heads"]
            embed_dims = [arch["hidden_dim"]] * layer_num

            if metric_key == "flower_kd_acc" and sample.get("flower_kd_acc", 0) < 50:
                continue

            archs.append(arch)
            performance = sample.get(metric_key, 0)
            performances.append(performance)

            ratio, b_value, g_value, s_value = calculate_information_value_ratio(
                layer_num, mlp_ratios, num_heads, embed_dims, self.param_dict
            )
            b_values.append(b_value)
            g_values.append(g_value)
            s_values.append(s_value)
            info_value_ratios.append(ratio)

        if self.is_zero_score:
            scores = zero_score_combine(b_values, g_values, s_values, self.weights)
        else:
            scores = info_value_ratios

        return scores, performances, b_values, g_values, s_values, archs

    # ------------------------------------------------------------------
    # Dataset evaluation (Our)
    # ------------------------------------------------------------------

    def evaluate_our(self, data, dataset):
        """Evaluate Info-NAS on a custom dataset.

        Returns:
            (scores, performances, b_values, g_values, s_values, archs)
        """
        archs, info_value_ratios, performances = [], [], []
        b_values, g_values, s_values = [], [], []

        t1 = time.time()

        for sample in data.values():
            arch = sample["net_setting"]
            performance = sample["performance"][dataset]
            archs.append(arch)

            ratio, b_value, g_value, s_value = self.score_arch(arch)
            b_values.append(b_value)
            g_values.append(g_value)
            s_values.append(s_value)
            info_value_ratios.append(ratio)
            performances.append(performance)

        if self.is_zero_score:
            scores = zero_score_combine(b_values, g_values, s_values, self.weights)
        else:
            scores = info_value_ratios

        delta_time = time.time() - t1
        avg = delta_time / len(archs)
        print(
            f"Evaluated {len(archs)} architectures in {delta_time:.3f}s "
            f"({avg*1000:.3f}ms/arch)"
        )

        return scores, performances, b_values, g_values, s_values, archs

    # ------------------------------------------------------------------
    # Convenience runners
    # ------------------------------------------------------------------

    def run_ood_test(self, dataset_type, plot=False):
        """Run full evaluation on an OoD dataset."""
        print(f"\n=== OoD-{dataset_type} ===")
        data = self.load_ood_data(dataset_type)
        if data is None:
            return

        scores, performances, *_ = self.evaluate_ood(data)
        print_correlation_only(scores, performances)

        if plot:
            plot_score_vs_performance(scores, performances, title=f"OoD-{dataset_type}")

    def run_bench_test(self, file_names=None, metric_key="c100_kd_acc", plot=False):
        """Run full evaluation on a Bench dataset."""
        if file_names is None:
            file_names = ["gt_autoformer.pth"]
        print(f"\n=== Bench: {file_names} (metric: {metric_key}) ===")
        data = self.load_bench_data(file_names)
        if data is None:
            return
        scores, performance, *_ = self.evaluate_bench(data, metric_key)
        print_correlation_only(scores, performance)

        if plot:
            plot_score_vs_performance(
                scores, performance, title=f"Bench ({metric_key})"
            )

    def run_our_test(
        self, dataset_name, dataset, scale="small", plot=False
    ):
        """Run full evaluation on a custom dataset."""
        print(f"\n=== Our: {dataset_name} ===")
        data = self.load_our_data(dataset_name, scale=scale)
        if data is None:
            return

        scores, performances, b_values, g_values, s_values, archs = self.evaluate_our(
            data, dataset
        )

        print_correlation_only(scores, performances)

        if plot:
            plot_score_vs_performance(scores, performances, title=dataset_name)

    # ------------------------------------------------------------------
    # Proxy extraction
    # ------------------------------------------------------------------

    def _extract_proxy(self, data):
        """Extract all proxy scores from a dataset dict."""
        data_value = {}
        for sample in data:
            for key in data[sample]["proxy"].keys():
                if key not in data_value:
                    data_value[key] = []
                data_value[key].append(data[sample]["proxy"][key])
        return data_value


def plot_score_vs_performance(scores, performances, title=""):
    """Plot Info-NAS score vs. validation accuracy scatter with Kendall's tau."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # ==================== 样式参数（在此微调） ====================
    font_family = "Times New Roman"
    base_font_size = 20  # 全局基础字号
    xlabel_font_size = 0  # x轴标签字号 ("Info-NAS Score")
    ylabel_font_size = 32  # y轴标签字号 ("Validation Accuracy (%)")
    tick_font_size = 30  # 刻度字号
    tau_font_size = 32  # Kendall's tau 标注字号
    tau_font_weight = "bold"  # tau 标注粗细 ("bold" / "normal")
    xlabel_font_weight = "bold"  # x轴标签粗细
    ylabel_font_weight = "bold"  # y轴标签粗细
    scatter_size = 60  # 散点大小
    scatter_cmap = "winter"  # 散点颜色映射
    spine_width = 1.0  # 边框粗细
    spine_color = "black"  # 边框颜色
    grid_linestyle = "--"  # 网格线样式
    grid_alpha = 0.6  # 网格透明度
    figure_size = (8, 6)  # 图像尺寸
    subplots_left = 0.167   # 左边距 (0~1)
    subplots_right = 0.98  # 右边距 (0~1)
    subplots_top = 0.993    # 上边距 (0~1)
    subplots_bottom = 0.07 # 下边距 (0~1)
    # ============================================================

    tau, _ = calculate_correlation(scores, performances)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [font_family] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = base_font_size
    plt.rcParams["axes.labelsize"] = max(xlabel_font_size, ylabel_font_size)
    plt.rcParams["xtick.labelsize"] = tick_font_size
    plt.rcParams["ytick.labelsize"] = tick_font_size

    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    ax.scatter(
        scores,
        performances,
        edgecolors="none",
        linewidth=0.5,
        s=scatter_size,
        c=list(scores),
        cmap=scatter_cmap,
    )

    # ax.set_xlabel("Info-NAS Score", fontsize=xlabel_font_size, fontweight=xlabel_font_weight)
    ax.set_ylabel(
        "Validation Accuracy (%)",
        fontsize=ylabel_font_size,
        fontweight=ylabel_font_weight,
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_color(spine_color)

    ax.text(
        0.03,
        0.96,
        r"Kendall's $\tau$=" + f"{tau:.3f}",
        transform=ax.transAxes,
        fontsize=tau_font_size,
        fontweight=tau_font_weight,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.subplots_adjust(left=subplots_left, right=subplots_right, top=subplots_top, bottom=subplots_bottom)
    plt.show()


def print_correlation_only(ratios, performance, name=""):
    """Print all three correlation metrics."""
    tau, _ = calculate_correlation(ratios, performance)
    print(f"{name} kendall : {tau:.3f}")
    rho, _ = calculate_spearman_correlation(ratios, performance)
    print(f"spearman : {rho:.3f}")
    r, _ = calculate_pearson_correlation(ratios, performance)
    print(f"pearson : {r:.3f}")
