# Video Generation Guide

This guide explains how to generate video demonstrations of your trained Double DDPG and Dual DDPG models, similar to the data generation script.

## Quick Start

### Generate Video for Double DDPG

```bash
cd /home/abhishek/SurRoL-SR-VPPV/VPPV/Training/policy_learning/rl
source /home/abhishek/SurRoL-SR-VPPV/py37/bin/activate

python generate_video.py \
    agent=double_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=3 \
    max_steps=30 \
    use_wb=False
```

### Generate Video for Dual DDPG

```bash
python generate_video.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=3 \
    max_steps=30 \
    use_wb=False
```

### Generate Videos for All Algorithms (Batch Script)

```bash
./generate_videos.sh
```

Or manually:

```bash
bash generate_videos.sh
```

## Parameters

- `agent`: Algorithm name (`ddpg`, `double_ddpg`, or `dual_ddpg`)
- `task`: Task name (`NeedleGrasp-Traj-v0`)
- `seed`: Random seed (usually `1`)
- `device`: Device (`cpu` or `cuda`)
- `ckpt_dir`: Path to checkpoint directory
- `ckpt_episode`: Checkpoint episode (`latest` for most recent, or specific number like `995`)
- `num_episodes`: Number of episodes to record (default: 1)
- `max_steps`: Maximum steps per episode (default: environment default, usually 10)
- `use_wb`: Whether to use Weights & Biases (set to `False`)

## Output

Videos are saved in the `video_demos/` directory (or specified `output_dir`):

```
video_demos/
├── DoubleDDPG_NeedleGrasp-Traj-v0_episode_1_success.mp4
├── DoubleDDPG_NeedleGrasp-Traj-v0_episode_2_success.mp4
├── DoubleDDPG_NeedleGrasp-Traj-v0_episode_3_failed.mp4
└── DoubleDDPG_NeedleGrasp-Traj-v0_combined_3episodes.mp4
```

### Video Naming Convention

- `{Algorithm}_{Task}_episode_{N}_success.mp4` - Successful episode
- `{Algorithm}_{Task}_episode_{N}_failed.mp4` - Failed episode
- `{Algorithm}_{Task}_combined_{N}episodes.mp4` - Combined video of all episodes

## Examples

### Single Episode (30 steps)

```bash
python generate_video.py \
    agent=double_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=1 \
    max_steps=30 \
    use_wb=False
```

### Multiple Episodes (for comparison)

```bash
python generate_video.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=5 \
    max_steps=30 \
    use_wb=False
```

### Using Specific Checkpoint

```bash
python generate_video.py \
    agent=double_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
    ckpt_episode=995 \
    num_episodes=3 \
    max_steps=30 \
    use_wb=False
```

## Comparison with Data Generation

The video generation script works similarly to `data_generation.py`:

| Feature | data_generation.py | generate_video.py |
|---------|-------------------|-------------------|
| Uses oracle actions | ✅ | ❌ |
| Uses trained policy | ❌ | ✅ |
| Records video | ✅ | ✅ |
| Saves frames | ✅ | ✅ |
| Multiple episodes | ✅ | ✅ |

## Troubleshooting

### No Images Captured

If videos are empty or no images are captured:

1. **Check render mode**: Ensure the environment supports rendering
2. **Check image format**: The script handles BGR to RGB conversion automatically
3. **Check episode length**: Ensure `max_steps` is appropriate for the task

### Video Quality Issues

- Videos are saved at 20 FPS by default
- Image resolution depends on environment settings
- For higher quality, adjust environment render settings

### Memory Issues

If you encounter memory issues with multiple episodes:

- Reduce `num_episodes`
- Process episodes one at a time
- Use smaller `max_steps`

## For Presentation

### Recommended Settings

For presentation/demonstration:

```bash
# Generate 3-5 episodes to show variability
num_episodes=5
max_steps=30  # Show full episode
```

### Best Episodes

The script automatically labels episodes as `success` or `failed`. For presentations:
- Use successful episodes to show the algorithm working
- Include a few failed episodes to show robustness testing

### Video Organization

Organize videos by algorithm:

```bash
# Create separate directories
mkdir -p video_demos/DoubleDDPG
mkdir -p video_demos/DualDDPG
mkdir -p video_demos/DDPG

# Generate videos with specific output directories
python generate_video.py ... --output_dir video_demos/DoubleDDPG
```

## Integration with README

Add video links to your README:

```markdown
## Video Demonstrations

- [Double DDPG Success](video_demos/DoubleDDPG/DoubleDDPG_NeedleGrasp-Traj-v0_episode_1_success.mp4)
- [Dual DDPG Success](video_demos/DualDDPG/DualDDPG_NeedleGrasp-Traj-v0_episode_1_success.mp4)
```

---

**Note**: Make sure `imageio` and `imageio-ffmpeg` are installed:

```bash
pip install imageio imageio-ffmpeg
```

