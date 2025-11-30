"""
Comparison script for DDPG, Double DDPG, and Dual DDPG algorithms.
This script runs all three algorithms and compares their performance.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

def run_training(agent_name, task, seed, n_steps=100000, use_wb=False):
    """Run training for a specific agent"""
    print(f"\n{'='*60}")
    print(f"Training {agent_name} on {task} (seed={seed})")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "train.py",
        f"agent={agent_name.lower()}",
        f"task={task}",
        f"seed={seed}",
        f"n_train_steps={n_steps}",
        f"use_wb={use_wb}",
        "n_eval=100",
        "n_save=1000",
        "n_log=1000"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"\n{agent_name} completed in {elapsed:.2f} seconds")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{agent_name} failed after {elapsed:.2f} seconds")
        print(f"Error: {e.stderr}")
        return False, elapsed

def main():
    """Main comparison function"""
    # Configuration
    agents = ['ddpg', 'doubleddpg', 'dualddpg']
    task = 'NeedlePick-v0'  # Change this to your desired task
    seeds = [1, 2, 3]  # Run with multiple seeds for statistical significance
    n_steps = 100000  # Number of training steps (adjust as needed)
    use_wb = False  # Set to True if you want to use Weights & Biases
    
    print("="*60)
    print("Algorithm Comparison: DDPG vs Double DDPG vs Dual DDPG")
    print("="*60)
    print(f"Task: {task}")
    print(f"Seeds: {seeds}")
    print(f"Training steps per run: {n_steps}")
    print("="*60)
    
    results = {}
    
    # Change to the RL directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run each algorithm
    for agent in agents:
        agent_results = []
        for seed in seeds:
            success, elapsed = run_training(agent, task, seed, n_steps, use_wb)
            agent_results.append({
                'seed': seed,
                'success': success,
                'time': elapsed
            })
        results[agent] = agent_results
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for agent, agent_results in results.items():
        successful = sum(1 for r in agent_results if r['success'])
        avg_time = sum(r['time'] for r in agent_results) / len(agent_results)
        print(f"\n{agent.upper()}:")
        print(f"  Successful runs: {successful}/{len(agent_results)}")
        print(f"  Average time: {avg_time:.2f} seconds")
    
    print("\n" + "="*60)
    print("Results saved in respective experiment directories:")
    print("  ./exp_local/{task}/{agent}/d{num_demo}/s{seed}/")
    print("="*60)

if __name__ == "__main__":
    main()

