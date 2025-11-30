# Deep Reinforcement Learning for Surgical Robot Control

## Overview

This repository implements and compares three Deep Reinforcement Learning (RL) algorithms for continuous control tasks in surgical robotics using the SurRoL platform. The algorithms are evaluated on the **NeedleGrasp-Traj-v0** task, which involves learning a policy for a surgical robot to grasp and manipulate a needle.

### Implemented Algorithms

1. **DDPG (Deep Deterministic Policy Gradient)** - Baseline algorithm
2. **Double DDPG** - Extension using double Q-learning with minimum Q-value selection
3. **Dual DDPG** - Extension using ensemble averaging of Q-values

## Key Results

| Algorithm | Validation Success Rate | Training Episodes | Best For |
|-----------|------------------------|-------------------|----------|
| **DDPG** | **100%** | ~300 | Efficiency & Speed |
| **Double DDPG** | **100%** | ~3,100 | Robustness & Bias Reduction |
| **Dual DDPG** | **98%** | ~2,100 | Balanced Approach |

All algorithms successfully learn the task, with DDPG and Double DDPG achieving perfect validation performance.

---

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Features

- ✅ Implementation of DDPG, Double DDPG, and Dual DDPG algorithms
- ✅ Training and evaluation scripts with Hydra configuration
- ✅ Comprehensive comparison of algorithm performance
- ✅ Visualization tools for training curves
- ✅ Checkpoint management for resuming training
- ✅ Support for CPU and GPU training
- ✅ Integration with SurRoL surgical robot simulation platform

---

## System Requirements

- **OS**: Ubuntu 20.04 (recommended) or Linux
- **Python**: 3.7
- **CUDA**: Optional (for GPU acceleration)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: ~5GB for dependencies and training data

### Dependencies

- PyTorch >= 1.13.0
- OpenAI Gym >= 0.15.6
- NumPy >= 1.21.1
- PyBullet (for physics simulation)
- Hydra (for configuration management)
- Matplotlib (for visualization)
- Pandas (for data analysis)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SurRoL-SR-VPPV/VPPV/Training/policy_learning/rl
```

### 2. Create Virtual Environment

```bash
# Create Python 3.7 virtual environment
python3.7 -m venv py37
source py37/bin/activate  # On Linux/Mac
# or
py37\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision
pip install gym==0.15.6
pip install numpy scipy pandas matplotlib
pip install hydra-core
pip install pybullet
```

### 4. Install SurRoL

Follow the SurRoL installation instructions from the main repository to set up the surgical robot simulation environment.

---

## Quick Start

### Training an Algorithm

```bash
# Activate virtual environment
source py37/bin/activate

# Train DDPG
python train.py agent=ddpg task=NeedleGrasp-Traj-v0 seed=1 device=cpu n_train_steps=100000

# Train Double DDPG
python train.py agent=double_ddpg task=NeedleGrasp-Traj-v0 seed=1 device=cpu n_train_steps=100000

# Train Dual DDPG
python train.py agent=dual_ddpg task=NeedleGrasp-Traj-v0 seed=1 device=cpu n_train_steps=100000
```

### Evaluating a Trained Model

```bash
python evaluate_model.py \
    agent=double_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
    ckpt_episode=latest \
    n_eval_episodes=50 \
    use_wb=False
```

### Generating Comparison Plots

```bash
python plot_results.py
```

This generates `training_comparison.png` with training curves for all algorithms.

---

## Usage

### Training Configuration

Training parameters can be configured via command-line arguments or by editing the configuration files in `configs/`:

```bash
python train.py \
    agent=double_ddpg \              # Algorithm: ddpg, double_ddpg, dual_ddpg
    task=NeedleGrasp-Traj-v0 \        # Task name
    seed=1 \                          # Random seed
    device=cpu \                      # Device: cpu or cuda
    n_train_steps=100000 \           # Total training steps
    replay_buffer_capacity=50000 \   # Replay buffer size
    batch_size=64 \                   # Batch size
    use_wb=False                      # Disable Weights & Biases logging
```

### Resuming Training

To resume training from a checkpoint:

```bash
python train.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    resume_training=True \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model \
    ckpt_episode=latest
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate_model.py \
    agent=<algorithm_name> \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=<path_to_checkpoint_directory> \
    ckpt_episode=latest \
    n_eval_episodes=50
```

---

## Algorithm Details

### DDPG (Deep Deterministic Policy Gradient)

**Architecture:**
- Single Q-network (Critic)
- Deterministic Actor network
- Standard DDPG implementation

**Key Features:**
- Fast convergence
- Simple architecture
- Prone to overestimation bias

**Target Q-value:**
```
target_Q = r + γ · Q_target(s', a')
```

### Double DDPG

**Architecture:**
- Two independent Q-networks (Q1 and Q2)
- Uses minimum of Q1 and Q2 for target computation
- Reduces overestimation bias

**Key Features:**
- More robust Q-value estimates
- Reduces overestimation bias
- Similar to TD3 approach (without delayed updates)

**Target Q-value:**
```
target_Q = r + γ · min(Q1_target(s', a'), Q2_target(s', a'))
```

**Actor Update:**
```
actor_loss = -mean(min(Q1(s, π(s)), Q2(s, π(s))))
```

### Dual DDPG

**Architecture:**
- Two independent Q-networks (Q1 and Q2)
- Uses average of Q1 and Q2 for target computation
- Ensemble averaging approach

**Key Features:**
- Balanced Q-value estimates
- Stable learning
- Ensemble benefits

**Target Q-value:**
```
target_Q = r + γ · (Q1_target(s', a') + Q2_target(s', a')) / 2
```

**Actor Update:**
```
actor_loss = -mean((Q1(s, π(s)) + Q2(s, π(s))) / 2)
```

### Network Architecture

Both Actor and Critic networks use 4-layer MLPs:
- **Input Layer**: State + Action (for Critic) or State (for Actor)
- **Hidden Layers**: 3 layers with 256 units each, ReLU activation
- **Output Layer**: Q-value (Critic) or Action (Actor with tanh)

---

## Results

### Validation Performance

All algorithms were evaluated on 50 independent episodes with deterministic policies (no exploration noise):

| Algorithm | Success Rate | Performance Level |
|-----------|--------------|-------------------|
| DDPG | **100%** | Excellent |
| Double DDPG | **100%** | Excellent |
| Dual DDPG | **98%** | Very Good |

### Training Statistics

| Algorithm | Training Episodes | Final Training SR* | Max Training SR | Total Steps |
|-----------|-------------------|-------------------|-----------------|-------------|
| DDPG | ~300 | 90% | 100% | ~30,000 |
| Double DDPG | ~3,100 | 80% | 100% | ~31,000 |
| Dual DDPG | ~2,100 | 70% | 100% | ~21,000 |

*Average success rate of last 10 training episodes

### Key Findings

1. **All algorithms achieve excellent performance**: Both DDPG and Double DDPG achieve perfect 100% validation success rate.

2. **DDPG baseline performs exceptionally**: Despite being the simplest algorithm, DDPG achieves perfect validation performance with the fastest convergence.

3. **Double DDPG shows robustness**: Achieves 100% validation success rate with better theoretical properties (reduces overestimation bias).

4. **Dual DDPG provides balanced performance**: Achieves 98% validation success rate using ensemble averaging.

### Training Curves

See `training_comparison.png` for detailed training curves showing success rate and reward over episodes for all three algorithms.

---

## Project Structure

```
rl/
├── agents/                    # Algorithm implementations
│   ├── ddpg.py               # DDPG baseline
│   ├── double_ddpg.py        # Double DDPG implementation
│   ├── dual_ddpg.py          # Dual DDPG implementation
│   ├── factory.py            # Agent factory
│   └── ...
├── configs/                   # Configuration files
│   ├── agent/
│   │   ├── ddpg.yaml
│   │   ├── double_ddpg.yaml
│   │   └── dual_ddpg.yaml
│   └── train.yaml
├── modules/                   # Neural network modules
│   ├── critics.py            # Critic networks (including DoubleCritic)
│   ├── policies.py           # Actor networks
│   └── subnetworks.py        # MLP implementation
├── trainers/                  # Training logic
│   └── rl_trainer.py         # Main training loop
├── components/                # Utility components
│   ├── checkpointer.py       # Checkpoint management
│   ├── logger.py            # Logging utilities
│   └── normalizer.py         # State normalization
├── train.py                   # Training script
├── evaluate_model.py         # Evaluation script
├── plot_results.py           # Visualization script
├── exp_local/                # Experiment results
│   └── NeedleGrasp-Traj-v0/
│       ├── DDPG/
│       ├── DoubleDDPG/
│       └── DualDDPG/
└── README.md                  # This file
```

---

## Configuration

### Hyperparameters

Default hyperparameters (can be modified in `configs/agent/*.yaml`):

- **Learning Rates**: 
  - Actor: 0.001
  - Critic: 0.001
- **Replay Buffer**: 50,000 transitions
- **Batch Size**: 64
- **Discount Factor (γ)**: 0.98
- **Soft Update Rate (τ)**: 0.005
- **Noise Epsilon (ε)**: 0.1 (for exploration)
- **Hidden Dimension**: 256

### Task Configuration

The `NeedleGrasp-Traj-v0` task is a goal-conditioned RL task where:
- **State**: Robot pose + goal position
- **Action**: Continuous joint velocities
- **Reward**: Sparse binary (0 for success, -1 per step)
- **Episode Length**: 10 steps

---

## Troubleshooting

### Common Issues

1. **Segmentation Fault**: 
   - Ensure PyBullet is properly installed
   - Try running with `device=cpu` if GPU issues occur
   - Check EGL renderer compatibility

2. **Out of Memory**:
   - Reduce `replay_buffer_capacity`
   - Reduce `batch_size`
   - Use CPU instead of GPU

3. **Training Not Resuming**:
   - Check checkpoint directory path
   - Ensure `ckpt_episode` is set correctly (use `latest` for most recent)
   - Verify checkpoint files exist

4. **Import Errors**:
   - Ensure virtual environment is activated
   - Install all dependencies
   - Check Python version (3.7 required)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{surrol_rl_implementation,
  title = {Deep Reinforcement Learning for Surgical Robot Control},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

### Related Papers

- **DDPG**: Lillicrap et al., "Continuous control with deep reinforcement learning", ICLR 2016
- **Double Q-Learning**: Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016
- **SurRoL**: Original SurRoL platform paper (cite if using the simulation environment)

---

## License

This project is based on the SurRoL platform. Please refer to the original SurRoL repository for licensing information.

---

## Acknowledgments

- SurRoL platform developers for the surgical robot simulation environment
- OpenAI for the Gym framework
- PyBullet team for the physics engine

---

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers.

---

## Additional Resources

- **Architecture Diagrams**: See `ARCHITECTURE_DIAGRAMS.md` for detailed algorithm architectures
- **Comparison Report**: See `ALGORITHM_COMPARISON_REPORT.md` for detailed analysis
- **Results Section**: See `REPORT_RESULTS_SECTION.md` for report-ready content
- **Quick Start**: See `QUICK_START.md` for quick reference

---

**Last Updated**: November 2025
