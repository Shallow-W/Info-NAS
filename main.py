"""Main entry point for Info-NAS evaluation.

Usage examples:
    # OoD-ViT-NAS benchmark (Tiny / Small / Base)
    python main.py --benchmark ood --scale Small --plot
    python main.py --benchmark ood --scale Tiny

    # AutoFormer / PiT benchmark
    python main.py --benchmark bench --files gt_autoformer.pth --metric c100_kd_acc --plot

    # ViT-Info-Bench (cifar-10, cifar-100, tiny-imagenet, miniimagenet)
    python main.py --benchmark our --dataset cifar-10 --scale tiny --plot
    python main.py --benchmark our --dataset cifar-100 --scale small
"""

import argparse

from info_nas.evaluator import Evaluator

OOD_SCALES = ["Tiny", "Small", "Base"]
OUR_DATASETS = ["cifar-10", "cifar-100", "tiny-imagenet", "miniimagenet"]
OUR_SCALES = ["tiny", "small"]
BENCH_METRICS = ["c100_kd_acc", "c10_kd_acc", "flower_kd_acc", "food_kd_acc"]


def main():
    parser = argparse.ArgumentParser(
        description="Info-NAS: Zero-Cost Proxy for ViT Architecture Search"
    )
    parser.add_argument(
        "-b", "--benchmark",
        required=True,
        choices=["ood", "bench", "our"],
        help="Benchmark type: ood (OoD-ViT-NAS), bench (AutoFormer/PiT), our (ViT-Info-Bench)",
    )
    parser.add_argument(
        "-s", "--scale",
        default="Small",
        choices=["Tiny", "Small", "Base", "tiny", "small"],
        help="Dataset scale (default: Small)",
    )
    parser.add_argument(
        "-d", "--dataset",
        help="Dataset name for 'our' benchmark (cifar-10, cifar-100, tiny-imagenet, miniimagenet)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["gt_autoformer.pth"],
        help="Bench data files (default: gt_autoformer.pth)",
    )
    parser.add_argument(
        "--metric",
        default="c100_kd_acc",
        choices=BENCH_METRICS,
        help="Evaluation metric for bench (default: c100_kd_acc)",
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="Plot score vs accuracy scatter",
    )
    parser.add_argument(
        "--no-zero-score",
        action="store_true",
        help="Disable z-score normalization",
    )

    args = parser.parse_args()

    if args.benchmark == "our" and not args.dataset:
        parser.error("'our' benchmark requires --dataset (cifar-10, cifar-100, tiny-imagenet, miniimagenet)")

    evaluator = Evaluator(is_zero_score=not args.no_zero_score)

    if args.benchmark == "ood":
        scale = args.scale
        print(f"\n{'='*60}")
        print(f" Benchmark: OoD-ViT-NAS | Scale: {scale}")
        print(f"{'='*60}")
        evaluator.run_ood_test(scale, plot=args.plot)

    elif args.benchmark == "bench":
        print(f"\n{'='*60}")
        print(f" Benchmark: Bench | Files: {args.files} | Metric: {args.metric}")
        print(f"{'='*60}")
        evaluator.run_bench_test(file_names=args.files, metric_key=args.metric, plot=args.plot)

    elif args.benchmark == "our":
        scale = args.scale.lower()
        print(f"\n{'='*60}")
        print(f" Benchmark: ViT-Info-Bench | Dataset: {args.dataset} | Scale: {scale}")
        print(f"{'='*60}")
        evaluator.run_our_test(
            f"{args.dataset}_{scale}",
            dataset=args.dataset,
            scale=scale,
            plot=args.plot,
        )


if __name__ == "__main__":
    main()
