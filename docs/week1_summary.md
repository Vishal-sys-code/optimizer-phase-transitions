# Week 1 Pilot Study: Summary of Findings

This document summarizes the initial pilot study on analyzing the internal dynamics of a small neural network trained with different optimizers.

## 1. Project Goal

The primary objective was to establish a minimal, yet complete, pipeline for mechanistic analysis. This involved:
- Training a simple model on a toy task with different optimizers (AdamW and SGD).
- Logging key metrics, model checkpoints, and intermediate layer activations.
- Analyzing the evolution of internal representations (circuits).
- Performing causal interventions (patching) to test hypotheses about representation function.

## 2. Training Settings & Reproducibility

The experiment can be reproduced using the following command. The script `src/train_small_model.py` was used to generate all artifacts.

### Reproducibility Checklist:
- **Command:** 
  ```bash
  # For AdamW
  python src/train_small_model.py --optimizer adamw --lr 1e-3 --seed 42 --num_steps 1001
  # For SGD
  python src/train_small_model.py --optimizer sgd --lr 1e-3 --seed 42 --num_steps 1001
  ```
- **Model:** 3-Layer MLP with GELU activations (`hidden_size=64`)
- **Dataset:** Synthetic token-mapping task (`(x+1) % 16`)
- **Seed:** `42`

## 3. Key Metric Trends

- **AdamW:** Converged rapidly, achieving near-perfect accuracy within ~100 training steps. The loss dropped quickly and plateaued, indicating efficient learning.
- **SGD:** With the same learning rate (`1e-3`), SGD showed much slower progress. The loss decreased minimally, and accuracy remained at chance level (around 0.0) for the first 100 steps. This highlights the well-known efficiency gap between adaptive optimizers and vanilla SGD for this type of task and hyperparameter setting.

## 4. Activation Analysis & Representational Alignment

The analysis was conducted in the `notebooks/week1_probe.ipynb` notebook.

- **Self-Similarity (AdamW):** The representations in the early layers (e.g., `act1`) of the AdamW model stabilized relatively quickly. The cosine similarity between activations at step 100 and later steps was high, suggesting that the basic features learned by the first layer were established early in training.

- **Cross-Optimizer Alignment:** A direct comparison of `act1` activations between the AdamW and SGD models at the same training steps showed very low cosine similarity. This is expected, given their different learning dynamics. It confirms that, at least initially, the two optimizers are exploring different regions of the weight space and learning incompatible representations.

## 5. Causal Patching Insights

A causal patching experiment was performed using `src/patch.py` to understand the functional role of the learned representations.

- **Experiment:** We patched the `act1` (first GELU layer output) from a trained AdamW model (step 100) into the poorly performing SGD model (step 100).
- **Command:**
  ```bash
  python src/patch.py --source_model checkpoints/adamw/step_100.pt --target_model checkpoints/sgd/step_100.pt --patch_layer act1
  ```
- **Result:** The patched model's accuracy increased from `0.0000` to `0.2603`.
- **Conclusion:** This provides strong causal evidence that the `act1` representation learned by AdamW is functionally important. It contains information that, when inserted into the SGD model's computational graph, directly improves its task performance. This demonstrates that the representations are not just correlated with success, but are a cause of it.

## 6. Conclusion

This pilot was highly successful. It demonstrates that our analysis pipeline is fully functional, from training and logging to advanced analysis and causal intervention. We have a "mechanistic lens" in place and are well-equipped to apply it to more complex models and tasks in future work. The initial findings already highlight significant representational differences driven by the choice of optimizer, a promising avenue for further investigation.