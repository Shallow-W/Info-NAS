"""Architecture search CLI for Info-NAS.

Scores, ranks, and compares ViT architectures using the zero-cost proxy.
Only supports ViT-Info-Bench (our) dataset format.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from info_nas import (
    DEFAULT_PARAMS,
    calculate_information_value_ratio,
)
from info_nas.config import param_list_to_dict
from info_nas.utils import z_score_normalization, zero_score_combine

DISPLAY_PROXIES = ["dss", "grasp", "meco", "SNIP", "AutoProxA", "synflow"]


class ArchitectureSearch:
    """Score, rank, and compare ViT architectures with Info-NAS."""

    def __init__(self, param=None, use_zero_score=True):
        self.use_zero_score = use_zero_score

        if param is None:
            self.param = DEFAULT_PARAMS
        else:
            self.param = list(param)

        self.param_dict = param_list_to_dict(self.param)

    @property
    def weights(self):
        return self.param[-3:]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, file_path_input):
        """Load ViT-Info-Bench JSON file(s), comma-separated paths."""
        file_paths = [p.strip() for p in file_path_input.split(",")]
        all_data = {}

        for file_path in file_paths:
            if not file_path:
                continue

            real_path = file_path
            if not os.path.exists(real_path):
                our_path = os.path.join("./data/our/", file_path)
                if os.path.exists(our_path):
                    print(f"Found in ./data/our/: {file_path}")
                    real_path = our_path
                else:
                    print(f"File not found: {file_path}")
                    continue

            try:
                with open(real_path, "r") as f:
                    current_data = json.load(f)
                print(f"Loaded: {real_path} ({len(current_data)} architectures)")
                all_data.update(current_data)
            except Exception as e:
                print(f"Error loading {real_path}: {e}")

        if not all_data:
            return None
        print(f"Total: {len(all_data)} architectures")
        return all_data

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_single_architecture(self, arch):
        """Score a single architecture. Returns (score, b, g, s, cost_time)."""
        start_time = time.time()

        c = arch["net_setting"]
        score, b_value, g_value, s_value = calculate_information_value_ratio(
            c["layer_num"],
            c["mlp_ratio"],
            c["num_heads"],
            c["embed_dim"],
            self.param_dict,
        )
        cost_time = time.time() - start_time
        return score, b_value, g_value, s_value, cost_time

    def apply_zero_score(self, b_values, g_values, s_values):
        """Z-score normalize and combine B/G/S components."""
        return zero_score_combine(b_values, g_values, s_values, self.weights)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_top_architectures(self, data, top_k=10, verbose=True):
        """Score all architectures and return top-K."""
        results = []
        total_time = 0
        b_values, g_values, s_values = [], [], []

        items = list(data.items())
        print(f"\nScoring {len(items)} architectures...")
        if self.use_zero_score:
            print("Z-Score normalization enabled\n")

        for idx, (key, arch_data) in enumerate(items):
            try:
                score, b, g, s, ct = self.score_single_architecture(arch_data)
                total_time += ct
                results.append(
                    {
                        "key": key,
                        "score": score,
                        "b_value": b,
                        "g_value": g,
                        "s_value": s,
                        "cost_time": ct,
                        "arch_data": arch_data,
                    }
                )
                b_values.append(b)
                g_values.append(g)
                s_values.append(s)
                if verbose and (idx + 1) % 100 == 0:
                    avg = total_time / (idx + 1)
                    print(
                        f"Progress: {idx+1}/{len(items)} | avg: {avg*1000:.3f}ms/arch"
                    )
            except Exception as e:
                print(f"Scoring failed (key={key}): {e}")

        if self.use_zero_score and results:
            normalized = self.apply_zero_score(b_values, g_values, s_values)
            for i, r in enumerate(results):
                r["original_score"] = r["score"]
                r["score"] = normalized[i]

        results.sort(key=lambda x: x["score"], reverse=True)
        avg_time = total_time / len(results) if results else 0

        print(f"\nDone! Total: {total_time:.4f}s, avg: {avg_time*1000:.4f}ms/arch")
        print(f"Valid: {len(results)}/{len(items)}\n")
        return results[:top_k], avg_time

    # ------------------------------------------------------------------
    # Proxy comparison
    # ------------------------------------------------------------------

    def get_proxy_top_architectures(self, data, top_k=1):
        """Get top-K architectures per proxy metric."""
        items = list(data.items())
        archs_with_proxy = []
        available_proxies = set()

        for key, arch_data in items:
            if "proxy" in arch_data and isinstance(arch_data["proxy"], dict):
                available_proxies.update(arch_data["proxy"].keys())
                archs_with_proxy.append(
                    {"key": key, "arch_data": arch_data, "proxy": arch_data["proxy"]}
                )

        if not archs_with_proxy:
            print("No proxy data found")
            return {}

        print(
            f"\nFound {len(available_proxies)} proxy metrics: {', '.join(sorted(available_proxies))}"
        )

        proxy_results = {}
        for pname in sorted(available_proxies):
            valid = [
                a
                for a in archs_with_proxy
                if pname in a["proxy"] and a["proxy"][pname] is not None
            ]
            if valid:
                sorted_archs = sorted(
                    valid, key=lambda x: float(x["proxy"][pname]), reverse=True
                )
                proxy_results[pname] = sorted_archs[:top_k]
        return proxy_results

    def compare_with_our_method(self, data, top_k=1):
        """Compare Info-NAS with all proxy methods."""
        print("\n" + "=" * 100)
        print("Comparison: Info-NAS vs Proxy Methods".center(100))
        print("=" * 100)

        our_results, avg_time = self.search_top_architectures(
            data, top_k=top_k, verbose=False
        )
        proxy_results = self.get_proxy_top_architectures(data, top_k=top_k)

        return {
            "our_method": our_results,
            "other_proxies": proxy_results,
            "avg_time": avg_time,
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_accuracy(arch_data):
        """Extract all dataset accuracies from our-format data."""
        perf = arch_data.get("performance")
        if not isinstance(perf, dict):
            return None
        entries = {
            k: f"{v:.2f}%" for k, v in perf.items() if isinstance(v, (int, float))
        }
        if entries:
            return "  ".join(f"{k}: {v}" for k, v in entries.items())
        return None

    def display_top_architectures(self, top_results, show_details=True):
        """Print top architecture results."""
        print("=" * 80)
        print(f"Top-{len(top_results)} Architectures".center(80))
        print("=" * 80)
        for rank, r in enumerate(top_results, 1):
            print(f"\n[Rank {rank}]")
            print(f"  ID: {r['key']}")
            print(f"  Score: {r['score']:.6f}")
            acc = self._extract_accuracy(r["arch_data"])
            if acc is not None:
                print(f"  Accuracy: {acc}")
            print(f"  B={r['b_value']:.6f}  G={r['g_value']:.6f}  S={r['s_value']:.6f}")
            print(f"  Time: {r['cost_time']*1000:.4f}ms")
            if show_details and "arch_data" in r:
                ns = r["arch_data"]["net_setting"]
                print(
                    f"  Config: depth={ns['layer_num']}, mlp={ns['mlp_ratio']}, heads={ns['num_heads']}, dim={ns['embed_dim']}"
                )
            print("-" * 80)

    def save_top_architectures(self, top_results, output_path):
        """Save top architectures to JSON file."""
        save_data = []
        for r in top_results:
            save_data.append(
                {
                    "key": r["key"],
                    "score": float(r["score"]),
                    "b_value": float(r["b_value"]),
                    "g_value": float(r["g_value"]),
                    "s_value": float(r["s_value"]),
                    "cost_time": float(r["cost_time"]),
                    "arch_data": r["arch_data"],
                }
            )
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Saved to: {output_path}")


def main():
    """CLI entry point."""
    searcher = ArchitectureSearch()

    print("Enter ViT-Info-Bench file path (comma-separated for multiple):")
    print("  Examples:")
    print("    ViT-Info-Bench_tiny.json")
    print("    ViT-Info-Bench_small.json")

    file_path = input("\nPath (Enter for default): ").strip()
    if not file_path:
        file_path = "./data/our/ViT-Info-Bench_tiny.json"
        print(f"Using default: {file_path}")

    data = searcher.load_data(file_path)
    if data is None:
        return

    print("\nSelect mode:")
    print("  1 - Info-NAS top-K only")
    print("  2 - All proxy methods top-K")
    print("  3 - Compare Info-NAS vs all proxies")
    mode = input("\nMode (default 1): ").strip()
    mode = int(mode) if mode.isdigit() else 1

    top_k = input("Top-K (default 1): ").strip()
    top_k = int(top_k) if top_k.isdigit() else 1

    if mode == 1:
        results, _ = searcher.search_top_architectures(data, top_k=top_k)
        searcher.display_top_architectures(results)
    elif mode == 2:
        proxy_results = searcher.get_proxy_top_architectures(data, top_k=top_k)
        filtered = {k: v for k, v in proxy_results.items() if k in DISPLAY_PROXIES}
        if filtered:
            for pname, top_archs in filtered.items():
                print(f"\n{pname.upper()}: top-1 = {top_archs[0]['key']}")
    elif mode == 3:
        comparison = searcher.compare_with_our_method(data, top_k=top_k)
        searcher.display_top_architectures(comparison["our_method"])

        proxy_results = comparison["other_proxies"]
        filtered = {k: v for k, v in proxy_results.items() if k in DISPLAY_PROXIES}
        if filtered:
            print("\n" + "=" * 90)
            print(f"Proxy Methods Top-{top_k} Comparison".center(90))
            print("=" * 90)
            print(
                f"\n{'Proxy':<20} {'Rank':<6} {'ID':<10} {'Score':<15} {'Accuracy':<10}"
            )
            print("-" * 90)
            for pname, top_archs in filtered.items():
                for rank, arch in enumerate(top_archs, 1):
                    score = arch["proxy"][pname]
                    acc = ArchitectureSearch._extract_accuracy(arch["arch_data"])
                    acc_str = acc if acc is not None else "N/A"
                    print(
                        f"{pname:<20} {rank:<6} {arch['key']:<10} {float(score):<15.6f} {acc_str:<10}"
                    )
                print("-" * 90)
    else:
        print("Invalid mode")
        return

    print("\nDone!")


if __name__ == "__main__":
    main()
