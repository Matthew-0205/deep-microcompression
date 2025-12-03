# LeNet-5 Experiment Verification with NAS

This notebook verifies the LeNet-5 experiment using Neural Architecture Search (NAS) to find optimal pruning configurations, followed by quantization.

## Overview
1. **Baseline Model:** Train or load the baseline LeNet-5 model.
2. **NAS (Neural Architecture Search):** 
    - Generate random pruning configurations.
    - Train an `Estimator` (MLP) to predict accuracy based on configuration.
    - Use brute-force search with the estimator to find the best configuration under constraints (e.g., >99% relative accuracy).
3. **Pruning & Quantization:** Apply the optimal configuration found by NAS to the model and verify performance.

**Verification:** Prune, Quantize, and Retrain (QAT) the model using the found configuration to verify the results (Table I).