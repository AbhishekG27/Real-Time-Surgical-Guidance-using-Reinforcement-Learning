# Resume Dual DDPG Training

## Problem
Training stopped at episode 1690, but only checkpoint at episode 10 exists. When resuming, it starts from the beginning.

## Solution

The checkpoint at episode 10 contains:
- Episode: 10
- Global step: 100

However, since training reached episode 1690, we need to resume from the latest checkpoint. The issue is that checkpoints weren't being saved frequently enough.

## Fixed Checkpoint Saving
I've updated the code to:
1. Save checkpoints every 5 episodes (instead of 10)
2. Add a safety save if 20+ episodes pass without saving
3. Improve resume logic to default to 'latest' checkpoint

## Resume Command

To resume Dual DDPG training from the latest checkpoint:

```bash
cd /home/abhishek/SurRoL-SR-VPPV/VPPV/Training/policy_learning/rl
source /home/abhishek/SurRoL-SR-VPPV/py37/bin/activate

python train.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    n_train_steps=100000 \
    use_wb=False \
    replay_buffer_capacity=50000 \
    batch_size=64 \
    resume_training=True \
    ckpt_episode=latest
```

**Note**: Since only episode 10 checkpoint exists, it will resume from episode 10, not 1690. The training will continue from there, and with the improved checkpoint saving (every 5 episodes), you won't lose as much progress if it stops again.

## Alternative: Resume from Specific Episode

If you want to resume from episode 10 specifically:

```bash
python train.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    n_train_steps=100000 \
    use_wb=False \
    replay_buffer_capacity=50000 \
    batch_size=64 \
    resume_training=True \
    ckpt_episode=10
```

## Check Available Checkpoints

To see what checkpoints are available:

```bash
ls -lh exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model/*.pth
```

## What Changed in the Code

1. **More frequent checkpoint saving**: Now saves every 5 episodes instead of 10
2. **Safety save**: If 20+ episodes pass without saving, it forces a save
3. **Better resume logic**: Defaults to 'latest' if ckpt_episode is not specified

## Expected Behavior After Fix

- Checkpoints will be saved every 5 episodes (episodes 5, 10, 15, 20, 25, etc.)
- If training stops, you can resume from the latest checkpoint
- Resume will restore both episode number and global step from the checkpoint

## If Training Stops Again

1. Check the latest checkpoint:
   ```bash
   ls -lt exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model/*.pth | head -1
   ```

2. Note the episode number from the filename (e.g., `weights_ep1690.pth`)

3. Resume with:
   ```bash
   python train.py agent=dual_ddpg task=NeedleGrasp-Traj-v0 seed=1 device=cpu \
       n_train_steps=100000 use_wb=False replay_buffer_capacity=50000 batch_size=64 \
       resume_training=True ckpt_episode=latest
   ```

The training will continue from where it left off, and you won't lose progress beyond the last checkpoint (now every 5 episodes).

