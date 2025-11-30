"""
Plot training curves and comparison results for presentation.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def load_training_data(exp_dir):
    """Load training CSV data"""
    csv_files = glob.glob(str(Path(exp_dir) / "*.csv"))
    train_csv = [f for f in csv_files if 'train' in f.lower() and 'eval' not in f.lower()]
    if train_csv:
        return pd.read_csv(train_csv[0])
    return None

def plot_training_curve(exp_dir, label, color, ax):
    """Plot training curve for a single algorithm"""
    df = load_training_data(exp_dir)
    if df is None or 'episode_sr' not in df.columns:
        print(f"Warning: Could not load training data from {exp_dir}")
        return
    
    # Plot success rate over episodes
    ax.plot(df['episode'], df['episode_sr'], label=label, color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Training Progress: Success Rate over Episodes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])

def plot_reward_curve(exp_dir, label, color, ax):
    """Plot reward curve for a single algorithm"""
    df = load_training_data(exp_dir)
    if df is None or 'episode_reward' not in df.columns:
        return
    
    ax.plot(df['episode'], df['episode_reward'], label=label, color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Training Progress: Reward over Episodes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

def main():
    """Main plotting function"""
    base_dir = Path(__file__).parent / "exp_local" / "NeedleGrasp-Traj-v0"
    seed = 1
    
    # Define algorithms to plot
    algorithms = [
        {
            'name': 'DDPG',
            'exp_dir': base_dir / 'DDPG' / 'd0' / f's{seed}',
            'color': 'blue',
            'label': 'DDPG (Baseline)'
        },
        {
            'name': 'DoubleDDPG',
            'exp_dir': base_dir / 'DoubleDDPG' / 'd0' / f's{seed}',
            'color': 'green',
            'label': 'Double DDPG'
        },
        {
            'name': 'DualDDPG',
            'exp_dir': base_dir / 'DualDDPG' / 'd0' / f's{seed}',
            'color': 'red',
            'label': 'Dual DDPG'
        }
    ]
    
    # Filter to only existing experiments
    existing_algs = [alg for alg in algorithms if alg['exp_dir'].exists()]
    
    if not existing_algs:
        print("No training data found. Please train at least one algorithm first.")
        return
    
    print(f"Found {len(existing_algs)} algorithm(s) to plot:")
    for alg in existing_algs:
        print(f"  - {alg['label']}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot success rate
    for alg in existing_algs:
        plot_training_curve(alg['exp_dir'], alg['label'], alg['color'], ax1)
    
    # Plot reward
    for alg in existing_algs:
        plot_reward_curve(alg['exp_dir'], alg['label'], alg['color'], ax2)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(__file__).parent / "training_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    summary_data = []
    for alg in existing_algs:
        df = load_training_data(alg['exp_dir'])
        if df is not None and 'episode_sr' in df.columns:
            final_sr = df['episode_sr'].iloc[-10:].mean()  # Average of last 10 episodes
            max_sr = df['episode_sr'].max()
            summary_data.append({
                'Algorithm': alg['label'],
                'Final Success Rate': f"{final_sr:.3f}",
                'Max Success Rate': f"{max_sr:.3f}",
                'Total Episodes': len(df)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n", summary_df.to_string(index=False))
        
        # Save summary
        summary_file = Path(__file__).parent / "summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Training Summary Statistics\n")
            f.write("="*60 + "\n\n")
            f.write(summary_df.to_string(index=False))
        print(f"\n✅ Summary saved to: {summary_file}")
    
    plt.show()

if __name__ == "__main__":
    main()

