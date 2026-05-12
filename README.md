# Info-NAS

Official implementation of **"Info-Driven Zero-Cost Proxy: Rethinking Vision Transformer Architecture Evaluation via Information Quantification"** (IJCAI 2026).

Info-NAS is a zero-cost proxy method that evaluates ViT architectures using only structural parameters — no forward or backward propagation required. It quantifies information transmission through three components: global information capacity, local information gradient, and global consistency.

## Repository Structure

```
info_nas/             Core algorithm package
  calculator.py       Info-NAS scoring (information value ratio)
  config.py           Default hyperparameters
  metrics.py          Correlation metrics (Kendall, Spearman, Pearson)
  utils.py            Z-score normalization and score fusion
  evaluator.py        Dataset loading, scoring, and evaluation pipeline

data/                 Benchmark datasets
  ood/                OoD-ViT-NAS (Tiny / Small / Base)
                         Vision Transformer Neural Architecture Search for Out-of-Distribution Generalization: Benchmark and Insights
  our/                ViT-Info-Bench (Tiny / Small)
  bench/              AutoFormer / PiT architectures
                         Auto-Prox: Training-Free Vision Transformer Architecture Search via Automatic Proxy Discovery

scripts/
  search.py           Interactive architecture search
  train.py            Hyperparameter optimization

main.py               Entry point for evaluation
pyproject.toml        Package configuration
```

## Installation

```bash
uv sync
```

## Usage

### Evaluation

Run full evaluation with Kendall's tau, Spearman's rho, and Pearson's r:

```bash
# OoD-ViT-NAS benchmark
python main.py --benchmark ood --scale Tiny
python main.py --benchmark ood --scale Small

# ViT-Info-Bench
python main.py --benchmark our --dataset cifar-10 --scale tiny
python main.py --benchmark our --dataset cifar-100 --scale small

# AutoFormer / PiT benchmark (requires torch)
python main.py --benchmark bench --files gt_autoformer.pth --metric c100_kd_acc
```

### Architecture Search

```bash
python scripts/search.py
```





## Citation

```bibtex

```
If you find this work helpful, please cite our paper.
