"""
Generate video demonstrations of trained RL models.
Similar to data_generation.py but uses trained policies instead of oracle actions.
"""
import argparse
import os
import sys
import imageio
import numpy as np
import torch
import hydra
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trainers.rl_trainer import RLTrainer
import surrol.gym  # Register SurRoL environments


def generate_video(cfg, output_dir='video_demos', num_episodes=1, max_steps=None):
    """
    Generate video of trained model performing the task.
    
    Args:
        cfg: Hydra configuration object
        output_dir: Directory to save videos
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode (None for default)
    """
    # Set test mode
    cfg.test = True
    cfg.dont_save = True
    cfg.resume_training = True
    
    # Force CPU if CUDA is not available
    if not torch.cuda.is_available():
        cfg.device = 'cpu'
        if hasattr(cfg, 'agent'):
            cfg.agent.device = 'cpu'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    exp = RLTrainer(cfg)
    
    # Override max steps if specified
    if max_steps is not None:
        exp.train_sampler._max_episode_len = max_steps
        exp.eval_sampler._max_episode_len = max_steps
    
    print(f"\n{'='*60}")
    print(f"Generating video for {cfg.agent.name} on {cfg.task}")
    print(f"Checkpoint: {cfg.ckpt_dir}/{cfg.ckpt_episode}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    all_images = []
    success_count = 0
    
    # Generate videos for each episode
    for episode_idx in range(num_episodes):
        print(f"Recording episode {episode_idx + 1}/{num_episodes}...")
        
        # Sample episode with rendering
        episode, rollouts, env_steps = exp.train_sampler.sample_episode(
            is_train=False, 
            render=True
        )
        
        # Extract images from episode
        episode_images = []
        episode_success = False
        
        for step_data in episode:
            if 'image' in step_data:
                # Convert image to uint8 if needed
                img = step_data['image']
                
                # Handle different image formats
                if isinstance(img, np.ndarray):
                    # If BGR (OpenCV format), convert to RGB
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # Check if it's BGR (common in OpenCV)
                        img = img[:, :, ::-1]  # BGR to RGB
                    
                    # Normalize to uint8
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    # Ensure 3 channels (RGB)
                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)
                    elif len(img.shape) == 3 and img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=2)
                    
                    episode_images.append(img)
            
            # Check for success
            if 'success' in step_data and step_data['success'] > 0:
                episode_success = True
        
        if episode_success:
            success_count += 1
            print(f"  ✓ Episode {episode_idx + 1}: SUCCESS ({env_steps} steps)")
        else:
            print(f"  ✗ Episode {episode_idx + 1}: FAILED ({env_steps} steps)")
        
        # Save individual episode video
        if len(episode_images) > 0:
            video_name = f"{cfg.agent.name}_{cfg.task}_episode_{episode_idx + 1}"
            if episode_success:
                video_name += "_success"
            else:
                video_name += "_failed"
            video_name += ".mp4"
            
            video_path = os.path.join(output_dir, video_name)
            writer = imageio.get_writer(video_path, fps=20)
            for img in episode_images:
                writer.append_data(img)
            writer.close()
            print(f"  Saved: {video_path}")
        
        # Collect images for combined video
        all_images.extend(episode_images)
    
    # Save combined video if multiple episodes
    if num_episodes > 1 and len(all_images) > 0:
        combined_video_name = f"{cfg.agent.name}_{cfg.task}_combined_{num_episodes}episodes.mp4"
        combined_video_path = os.path.join(output_dir, combined_video_name)
        writer = imageio.get_writer(combined_video_path, fps=20)
        for img in all_images:
            writer.append_data(img)
        writer.close()
        print(f"\nCombined video saved: {combined_video_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Video Generation Complete!")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Videos saved in: {output_dir}")
    print(f"{'='*60}\n")
    
    return success_count / num_episodes if num_episodes > 0 else 0.0


@hydra.main(version_base=None, config_path="./configs", config_name="eval")
def main(cfg):
    """
    Generate video demonstrations of trained models.
    
    Usage:
        python generate_video.py agent=double_ddpg task=NeedleGrasp-Traj-v0 seed=1 \
            ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
            ckpt_episode=latest num_episodes=3 max_steps=30
    """
    # Get parameters from config or use defaults
    num_episodes = getattr(cfg, 'num_episodes', 1)
    max_steps = getattr(cfg, 'max_steps', None)
    output_dir = getattr(cfg, 'output_dir', 'video_demos')
    
    # Allow override via environment or direct assignment
    import os
    if 'NUM_EPISODES' in os.environ:
        num_episodes = int(os.environ['NUM_EPISODES'])
    if 'MAX_STEPS' in os.environ:
        max_steps = int(os.environ['MAX_STEPS'])
    if 'OUTPUT_DIR' in os.environ:
        output_dir = os.environ['OUTPUT_DIR']
    
    # Generate video
    success_rate = generate_video(
        cfg, 
        output_dir=output_dir,
        num_episodes=num_episodes,
        max_steps=max_steps
    )
    
    return success_rate


if __name__ == "__main__":
    main()

