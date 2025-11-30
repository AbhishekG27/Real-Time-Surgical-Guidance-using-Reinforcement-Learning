#!/bin/bash
# Script to generate videos for Double DDPG and Dual DDPG models

# Activate virtual environment
source /home/abhishek/SurRoL-SR-VPPV/py37/bin/activate

# Navigate to RL directory
cd /home/abhishek/SurRoL-SR-VPPV/VPPV/Training/policy_learning/rl

# Create video output directory
mkdir -p video_demos

echo "=========================================="
echo "Generating Videos for Trained Models"
echo "=========================================="

# Generate video for Double DDPG
echo ""
echo "1. Generating video for Double DDPG..."
python generate_video.py \
    agent=double_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=3 \
    max_steps=30 \
    use_wb=False \
    --num_episodes 3 \
    --max_steps 30 \
    --output_dir video_demos/DoubleDDPG

# Generate video for Dual DDPG
echo ""
echo "2. Generating video for Dual DDPG..."
python generate_video.py \
    agent=dual_ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DualDDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=3 \
    max_steps=30 \
    use_wb=False \
    --num_episodes 3 \
    --max_steps 30 \
    --output_dir video_demos/DualDDPG

# Optional: Generate video for DDPG baseline
echo ""
echo "3. Generating video for DDPG (Baseline)..."
python generate_video.py \
    agent=ddpg \
    task=NeedleGrasp-Traj-v0 \
    seed=1 \
    device=cpu \
    ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DDPG/d0/s1/model \
    ckpt_episode=latest \
    num_episodes=3 \
    max_steps=30 \
    use_wb=False \
    --num_episodes 3 \
    --max_steps 30 \
    --output_dir video_demos/DDPG

echo ""
echo "=========================================="
echo "Video Generation Complete!"
echo "Videos saved in: video_demos/"
echo "=========================================="
echo ""
echo "To view videos:"
echo "  ls -lh video_demos/*/*.mp4"
echo ""

