# Mechanistic Interpretability Pilot: Analyzing Optimizer-Driven Circuit Emergence

This repository contains the code and analysis for a pilot study in mechanistic interpretability. The project aims to build a foundational toolkit for dissecting the internal workings of small neural networks, with a specific focus on how the choice of optimizer (AdamW vs. SGD) influences the emergence of learned features, or "circuits."

## Core Features

This project provides a complete, end-to-end pipeline for:

* **Training Simple Models**: A 3-layer MLP is trained on a synthetic, algorithmic task.
* **Deep Activation Logging**: Forward hooks capture and save intermediate layer activations at regular training intervals.
* **Optimizer Comparison**: Easily train models with either `AdamW` or `SGD` to compare their learning dynamics.
* **Representational Analysis**: A Jupyter notebook (`notebooks/week1_probe.ipynb`) provides tools for analyzing how representations evolve and diverge, using cosine similarity as a primary metric.
* **Causal Patching**: A dedicated script (`src/patch.py`) allows for causal interventions by "patching" activations from one model into another to measure their functional impact on performance.

## Getting Started

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Vishal-sys-code/optimizer-phase-transitions.git
cd optimizer-phase-transitions
pip install -r requirements.txt
```

### 2. Running Experiments

The primary script is `src/train_small_model.py`. It will train the model, save checkpoints, and log activations.

**To train a model with AdamW:**
```bash
python src/train_small_model.py --optimizer adamw --lr 1e-3 --seed 42 --num_steps 1001 --save_every 100
```

**To train a model with SGD:**
```bash
python src/train_small_model.py --optimizer sgd --lr 1e-3 --seed 42 --num_steps 1001 --save_every 100
```
The generated artifacts will be saved in the `/checkpoints` and `/analysis` directories.

### 3. Analyzing the Results

*   **Quantitative Analysis**: Open and run the `notebooks/week1_probe.ipynb` notebook to visualize the representational drift and cross-optimizer alignment.
*   **Causal Analysis**: Use the `src/patch.py` script to test the causal impact of a specific layer's representation. For example, to patch the `act1` layer from the AdamW model (at step 100) into the SGD model:

```bash
python src/patch.py --source_model checkpoints/adamw/step_100.pt --target_model checkpoints/sgd/step_100.pt --patch_layer act1
```

## Project Structure

```
.
├── analysis/         # Saved activation tensors (.npz)
│   └── activations/
├── checkpoints/      # Model checkpoints (.pt)
├── configs/          # (Future use) Configuration files
├── data/             # (Future use) Datasets
├── docs/             # Project documentation
│   └── week1_summary.md # Summary of initial findings
├── notebooks/        # Jupyter notebooks for analysis
│   └── week1_probe.ipynb
├── src/              # Source code
│   ├── train_small_model.py # Main training script
│   └── patch.py             # Causal patching script
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## Findings

A summary of the initial findings from this pilot study can be found in [docs/week1_summary.md](./docs/week1_summary.md). The report details the differences in metric trends, representational alignment, and the results of our causal patching experiments.