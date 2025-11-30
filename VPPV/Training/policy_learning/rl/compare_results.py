"""
Compare evaluation results between DDPG, Double DDPG, and Dual DDPG.
This script evaluates all trained models and generates a comparison report.
"""
import os
import subprocess
import sys
import json
from pathlib import Path
import pandas as pd


def evaluate_model(agent_name, task, seed, ckpt_dir, ckpt_episode='latest', n_episodes=50):
    """Evaluate a single model and return results"""
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} (seed={seed})")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "evaluate_model.py",
        f"agent={agent_name.lower()}",
        f"task={task}",
        f"seed={seed}",
        f"ckpt_dir={ckpt_dir}",
        f"ckpt_episode={ckpt_episode}",
        f"n_eval_episodes={n_episodes}",
        "use_wb=False"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        # Parse output to extract success rate
        output = result.stdout
        # Look for success rate in output
        for line in output.split('\n'):
            if 'Success Rate' in line or 'episode_sr' in line:
                print(line)
        return True, output
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {agent_name}: {e.stderr}")
        return False, None


def find_latest_checkpoint(exp_dir):
    """Find the latest checkpoint in the experiment directory"""
    model_dir = Path(exp_dir) / "model"
    if not model_dir.exists():
        return None
    
    checkpoints = list(model_dir.glob("weights_ep*.pth"))
    if not checkpoints:
        # Try best.pth
        best_ckpt = model_dir / "best.pth"
        if best_ckpt.exists():
            return "best"
        return None
    
    # Extract episode numbers and find latest
    episodes = []
    for ckpt in checkpoints:
        try:
            ep_num = int(ckpt.stem.split('_ep')[-1])
            episodes.append((ep_num, ckpt.name))
        except:
            continue
    
    if episodes:
        latest_ep, latest_name = max(episodes, key=lambda x: x[0])
        return latest_name.replace('.pth', '')
    
    return "latest"


def main():
    """Main comparison function"""
    task = 'NeedleGrasp-Traj-v0'
    seed = 1
    n_episodes = 50  # Number of evaluation episodes
    
    # Define experiment directories
    base_dir = Path(__file__).parent / "exp_local" / task
    
    agents_config = [
        {
            'name': 'ddpg',
            'exp_dir': base_dir / 'DDPG' / 'd0' / f's{seed}',
            'display_name': 'DDPG (Baseline)'
        },
        {
            'name': 'double_ddpg',
            'exp_dir': base_dir / 'DoubleDDPG' / 'd0' / f's{seed}',
            'display_name': 'Double DDPG'
        },
        {
            'name': 'dual_ddpg',
            'exp_dir': base_dir / 'DualDDPG' / 'd0' / f's{seed}',
            'display_name': 'Dual DDPG'
        }
    ]
    
    results = []
    
    print("="*60)
    print("Algorithm Comparison: DDPG vs Double DDPG vs Dual DDPG")
    print("="*60)
    print(f"Task: {task}")
    print(f"Seed: {seed}")
    print(f"Evaluation episodes per algorithm: {n_episodes}")
    print("="*60)
    
    # Change to the RL directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Evaluate each algorithm
    for agent_config in agents_config:
        agent_name = agent_config['name']
        exp_dir = agent_config['exp_dir']
        display_name = agent_config['display_name']
        
        if not exp_dir.exists():
            print(f"\n⚠️  Warning: Experiment directory not found: {exp_dir}")
            print(f"   Skipping {display_name}\n")
            continue
        
        # Find checkpoint
        ckpt_episode = find_latest_checkpoint(exp_dir)
        if ckpt_episode is None:
            print(f"\n⚠️  Warning: No checkpoint found in {exp_dir}")
            print(f"   Skipping {display_name}\n")
            continue
        
        ckpt_dir = str(exp_dir / "model")
        
        print(f"\nFound checkpoint: {ckpt_episode} in {ckpt_dir}")
        
        # Evaluate
        success, output = evaluate_model(
            agent_name, task, seed, ckpt_dir, ckpt_episode, n_episodes
        )
        
        if success:
            results.append({
                'Algorithm': display_name,
                'Checkpoint': ckpt_episode,
                'Status': 'Evaluated'
            })
        else:
            results.append({
                'Algorithm': display_name,
                'Checkpoint': ckpt_episode,
                'Status': 'Failed'
            })
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results)
    print("\n", df.to_string(index=False))
    
    # Save results
    results_file = script_dir / "comparison_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Algorithm Comparison Results\n")
        f.write("="*60 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("Detailed evaluation logs are in the respective experiment directories.\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*60)
    print("Next Steps for Presentation:")
    print("="*60)
    print("1. Check the CSV files in each experiment directory for training curves")
    print("2. Compare success rates from the evaluation runs")
    print("3. Plot training curves using the CSV data")
    print("4. Generate visualizations comparing the algorithms")
    print("="*60)


if __name__ == "__main__":
    main()

