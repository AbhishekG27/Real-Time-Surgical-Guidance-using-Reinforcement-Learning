# Dual DDPG Training Started

## Training Command
```bash
python train.py agent=dual_ddpg task=NeedleGrasp-Traj-v0 seed=1 device=cpu n_train_steps=100000 use_wb=False replay_buffer_capacity=50000 batch_size=64
```

## Training Details
- **Algorithm**: Dual DDPG
- **Task**: NeedleGrasp-Traj-v0
- **Seed**: 1
- **Device**: CPU
- **Training Steps**: 100,000
- **Replay Buffer**: 50,000
- **Batch Size**: 64

## Output Location
Training data will be saved to:
```
exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/
```

## Monitor Training

### Check if training is running:
```bash
ps aux | grep "train.py agent=dual_ddpg" | grep -v grep
```

### View training progress:
```bash
# Watch training CSV file
tail -f exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/train.csv

# Or check periodically
cat exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/train.csv | tail -10
```

### Check training logs:
```bash
tail -f exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/*.log
```

### View checkpoints:
```bash
ls -lh exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model/*.pth | tail -5
```

## After Training Completes

### 1. Evaluate the model:
```bash
python evaluate_model.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model \
    ckpt_episode=latest \
    n_eval_episodes=50 \
    use_wb=False
```

### 2. Update comparison plots:
```bash
python plot_results.py
```

This will now include all three algorithms:
- DDPG (Baseline)
- Double DDPG
- Dual DDPG

## Expected Training Time
Based on previous runs:
- ~30,000 steps ≈ 1,000 episodes
- Full 100,000 steps ≈ 3,000+ episodes
- Estimated time: Several hours (depending on CPU speed)

## Resume Training (if interrupted)
```bash
python train.py agent=dual_ddpg task=NeedleGrasp-Traj-v0 seed=1 device=cpu \
    n_train_steps=100000 use_wb=False replay_buffer_capacity=50000 batch_size=64 \
    resume_training=True ckpt_episode=latest
```

## Key Differences: Dual DDPG vs Double DDPG

| Feature | Double DDPG | Dual DDPG |
|---------|-------------|-----------|
| Target Q | `min(Q1, Q2)` | `(Q1 + Q2) / 2` |
| Actor Update | `min(Q1, Q2)` | `(Q1 + Q2) / 2` |
| Bias Reduction | Conservative (minimum) | Balanced (average) |
| Stability | Very stable | Stable |

Good luck! 🚀

