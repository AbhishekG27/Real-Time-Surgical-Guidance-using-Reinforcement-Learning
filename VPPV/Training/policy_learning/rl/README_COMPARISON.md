# Algorithm Comparison: DDPG vs Double DDPG vs Dual DDPG

## Overview

This document describes the implementation and comparison of three Deep Deterministic Policy Gradient (DDPG) variants for surgical robot learning.

## Implemented Algorithms

### 1. DDPG (Baseline)
- **File**: `agents/ddpg.py`
- **Description**: Standard Deep Deterministic Policy Gradient algorithm
- **Key Features**:
  - Single Q-network (critic)
  - Deterministic policy (actor)
  - Target networks with soft updates

### 2. Double DDPG
- **File**: `agents/double_ddpg.py`
- **Description**: DDPG with two Q-networks using minimum for target computation
- **Key Features**:
  - Two independent Q-networks (Q1, Q2)
  - Uses `min(Q1, Q2)` for target computation (reduces overestimation bias)
  - Similar to TD3 but without delayed policy updates
- **Advantages**: Reduces overestimation bias in Q-values

### 3. Dual DDPG
- **File**: `agents/dual_ddpg.py`
- **Description**: DDPG with two Q-networks using averaging
- **Key Features**:
  - Two independent Q-networks (Q1, Q2)
  - Uses `(Q1 + Q2) / 2` for both target and actor updates
  - Ensemble averaging approach
- **Advantages**: More stable Q-value estimates through ensemble averaging

## Key Differences

| Feature | DDPG | Double DDPG | Dual DDPG |
|---------|------|-------------|-----------|
| Number of Critics | 1 | 2 | 2 |
| Target Q-value | Q_target | min(Q1_target, Q2_target) | (Q1_target + Q2_target) / 2 |
| Actor Update | Q | min(Q1, Q2) | (Q1 + Q2) / 2 |
| Overestimation Bias | Higher | Lower | Moderate |

## Usage

### Training Individual Algorithms

```bash
# DDPG (baseline)
python train.py agent=ddpg task=NeedlePick-v0 seed=1

# Double DDPG
python train.py agent=doubleddpg task=NeedlePick-v0 seed=1

# Dual DDPG
python train.py agent=dualddpg task=NeedlePick-v0 seed=1
```

### Running Comparison

```bash
# Run comparison script (runs all algorithms with multiple seeds)
python compare_algorithms.py
```

### Configuration

All algorithms use the same hyperparameters by default (see `configs/agent/ddpg.yaml`, `double_ddpg.yaml`, `dual_ddpg.yaml`):
- Learning rate: 1e-3 (actor and critic)
- Discount factor: 0.99
- Soft target update: 0.005
- Hidden dimension: 256
- Update epochs: 40

## Expected Results

### Double DDPG
- **Expected**: Lower Q-value estimates, potentially more stable learning
- **Trade-off**: May be more conservative in action selection

### Dual DDPG
- **Expected**: Smoother Q-value estimates, balanced between DDPG and Double DDPG
- **Trade-off**: Ensemble averaging provides stability without being too conservative

## Reproducing Results

To reproduce the existing DDPG results and compare with new methods:

1. **Baseline (DDPG)**:
   ```bash
   python train.py agent=ddpg task=NeedlePick-v0 seed=1 n_train_steps=1000000
   ```

2. **Double DDPG**:
   ```bash
   python train.py agent=doubleddpg task=NeedlePick-v0 seed=1 n_train_steps=1000000
   ```

3. **Dual DDPG**:
   ```bash
   python train.py agent=dualddpg task=NeedlePick-v0 seed=1 n_train_steps=1000000
   ```

## Evaluation Metrics

Key metrics to compare:
- **Success Rate**: Percentage of successful episodes
- **Episode Return**: Average cumulative reward
- **Q-value Estimates**: Monitor for overestimation
- **Training Stability**: Variance in performance across seeds

## Notes

- All algorithms use HER (Hindsight Experience Replay) by default
- Results are saved in `./exp_local/{task}/{agent}/d{num_demo}/s{seed}/`
- Use Weights & Biases (`use_wb=True`) for better visualization and tracking

