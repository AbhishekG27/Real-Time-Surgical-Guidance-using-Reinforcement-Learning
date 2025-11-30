"""
Evaluation script for trained RL models.
This script loads a checkpoint and evaluates the model's performance.
"""
import hydra
from trainers.rl_trainer import RLTrainer
import os


@hydra.main(version_base=None, config_path="./configs", config_name="eval")
def main(cfg):
    """
    Evaluate a trained model.
    
    Usage:
        python evaluate_model.py agent=double_ddpg task=NeedleGrasp-Traj-v0 seed=1 \
            ckpt_dir=exp_local/NeedleGrasp-Traj-v0/DoubleDDPG/d0/s1/model \
            ckpt_episode=latest n_eval_episodes=50
    """
    # Set test mode to only evaluate, not train
    cfg.test = True
    cfg.dont_save = True  # Don't save during evaluation
    cfg.resume_training = True  # Load checkpoint
    
    # Force CPU if CUDA is not available (for systems without GPU)
    import torch
    if not torch.cuda.is_available():
        cfg.device = 'cpu'
        if hasattr(cfg, 'agent'):
            cfg.agent.device = 'cpu'
    
    exp = RLTrainer(cfg)
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating {cfg.agent.name} on {cfg.task}")
    print(f"Checkpoint: {cfg.ckpt_dir}/{cfg.ckpt_episode}")
    print(f"Number of evaluation episodes: {cfg.n_eval_episodes}")
    print(f"{'='*60}\n")
    
    score = exp.eval()
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Success Rate: {score:.4f}")
    print(f"{'='*60}\n")
    
    return score


if __name__ == "__main__":
    main()

