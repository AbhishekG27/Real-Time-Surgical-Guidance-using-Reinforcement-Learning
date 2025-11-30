#!/usr/bin/env python3
"""
Generate visual architecture diagrams for Double DDPG and Dual DDPG algorithms.
Requires: matplotlib, graphviz (optional)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

def create_double_ddpg_diagram():
    """Create a visual diagram for Double DDPG architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'Double DDPG Architecture', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((1, 11), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 11.4, 'State\n[s, a]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Double Critic Networks
    q1_box = FancyBboxPatch((0.5, 8.5), 1.5, 2, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(q1_box)
    ax.text(1.25, 9.5, 'Q1 Network\n(MLP)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    q2_box = FancyBboxPatch((2.5, 8.5), 1.5, 2, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(q2_box)
    ax.text(3.25, 9.5, 'Q2 Network\n(MLP)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Q-values
    q1_val = FancyBboxPatch((0.5, 7), 1.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(q1_val)
    ax.text(1.25, 7.3, 'Q1(s,a)', ha='center', va='center', fontsize=9)
    
    q2_val = FancyBboxPatch((2.5, 7), 1.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(q2_val)
    ax.text(3.25, 7.3, 'Q2(s,a)', ha='center', va='center', fontsize=9)
    
    # Min operation
    min_box = FancyBboxPatch((1.5, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(min_box)
    ax.text(2.25, 5.9, 'min(Q1, Q2)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((1.5, 4), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(2.25, 4.4, 'Q_min(s,a)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    # Input to Q networks
    arrow1 = FancyArrowPatch((1.75, 11), (1.25, 10.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((1.75, 11), (3.25, 10.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Q networks to Q values
    arrow3 = FancyArrowPatch((1.25, 8.5), (1.25, 7.6), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='green')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((3.25, 8.5), (3.25, 7.6), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='green')
    ax.add_patch(arrow4)
    
    # Q values to min
    arrow5 = FancyArrowPatch((1.25, 7), (2.0, 6.3), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((3.25, 7), (2.5, 6.3), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax.add_patch(arrow6)
    
    # Min to output
    arrow7 = FancyArrowPatch((2.25, 5.5), (2.25, 4.8), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow7)
    
    # Right side: Training flow
    # Actor
    actor_box = FancyBboxPatch((6, 10), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='lightcyan', edgecolor='black', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(7, 10.75, 'Actor Network\nπ(s)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Target computation
    target_box = FancyBboxPatch((6, 7.5), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor='wheat', edgecolor='black', linewidth=2)
    ax.add_patch(target_box)
    ax.text(7, 8.25, 'Target Computation\nr + γ·min(Q1, Q2)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Loss
    loss_box = FancyBboxPatch((6, 5), 2, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='mistyrose', edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, 5.75, 'Loss\nMSE(Q1, target) +\nMSE(Q2, target)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Update
    update_box = FancyBboxPatch((6, 2.5), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor='lavender', edgecolor='black', linewidth=2)
    ax.add_patch(update_box)
    ax.text(7, 3.25, 'Update Networks\nθ ← θ - α∇L', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrows for training flow
    arrow8 = FancyArrowPatch((7, 10), (7, 9), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow8)
    arrow9 = FancyArrowPatch((7, 7.5), (7, 6.5), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow9)
    arrow10 = FancyArrowPatch((7, 5), (7, 4), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow10)
    
    # Key feature box
    key_box = FancyBboxPatch((0.5, 0.5), 4.5, 2.5, boxstyle="round,pad=0.2", 
                             facecolor='lightyellow', edgecolor='red', linewidth=2)
    ax.add_patch(key_box)
    ax.text(2.75, 2.2, 'KEY FEATURE: Uses min(Q1, Q2) to reduce overestimation bias', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    ax.text(2.75, 1.5, '• Target: r + γ·min(Q1_target, Q2_target)\n• Actor: max min(Q1, Q2)', 
            ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('double_ddpg_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: double_ddpg_architecture.png")
    plt.close()

def create_dual_ddpg_diagram():
    """Create a visual diagram for Dual DDPG architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'Dual DDPG Architecture', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((1, 11), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 11.4, 'State\n[s, a]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Double Critic Networks
    q1_box = FancyBboxPatch((0.5, 8.5), 1.5, 2, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(q1_box)
    ax.text(1.25, 9.5, 'Q1 Network\n(MLP)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    q2_box = FancyBboxPatch((2.5, 8.5), 1.5, 2, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(q2_box)
    ax.text(3.25, 9.5, 'Q2 Network\n(MLP)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Q-values
    q1_val = FancyBboxPatch((0.5, 7), 1.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(q1_val)
    ax.text(1.25, 7.3, 'Q1(s,a)', ha='center', va='center', fontsize=9)
    
    q2_val = FancyBboxPatch((2.5, 7), 1.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(q2_val)
    ax.text(3.25, 7.3, 'Q2(s,a)', ha='center', va='center', fontsize=9)
    
    # Average operation
    avg_box = FancyBboxPatch((1.5, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(avg_box)
    ax.text(2.25, 5.9, '(Q1 + Q2) / 2', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((1.5, 4), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(2.25, 4.4, 'Q_avg(s,a)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    # Input to Q networks
    arrow1 = FancyArrowPatch((1.75, 11), (1.25, 10.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((1.75, 11), (3.25, 10.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Q networks to Q values
    arrow3 = FancyArrowPatch((1.25, 8.5), (1.25, 7.6), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='green')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((3.25, 8.5), (3.25, 7.6), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='green')
    ax.add_patch(arrow4)
    
    # Q values to average
    arrow5 = FancyArrowPatch((1.25, 7), (2.0, 6.3), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((3.25, 7), (2.5, 6.3), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax.add_patch(arrow6)
    
    # Average to output
    arrow7 = FancyArrowPatch((2.25, 5.5), (2.25, 4.8), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow7)
    
    # Right side: Training flow
    # Actor
    actor_box = FancyBboxPatch((6, 10), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='lightcyan', edgecolor='black', linewidth=2)
    ax.add_patch(actor_box)
    ax.text(7, 10.75, 'Actor Network\nπ(s)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Target computation
    target_box = FancyBboxPatch((6, 7.5), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor='wheat', edgecolor='black', linewidth=2)
    ax.add_patch(target_box)
    ax.text(7, 8.25, 'Target Computation\nr + γ·(Q1 + Q2)/2', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Loss
    loss_box = FancyBboxPatch((6, 5), 2, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='mistyrose', edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, 5.75, 'Loss\nMSE(Q1, target) +\nMSE(Q2, target)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Update
    update_box = FancyBboxPatch((6, 2.5), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor='lavender', edgecolor='black', linewidth=2)
    ax.add_patch(update_box)
    ax.text(7, 3.25, 'Update Networks\nθ ← θ - α∇L', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrows for training flow
    arrow8 = FancyArrowPatch((7, 10), (7, 9), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow8)
    arrow9 = FancyArrowPatch((7, 7.5), (7, 6.5), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow9)
    arrow10 = FancyArrowPatch((7, 5), (7, 4), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow10)
    
    # Key feature box
    key_box = FancyBboxPatch((0.5, 0.5), 4.5, 2.5, boxstyle="round,pad=0.2", 
                             facecolor='lightyellow', edgecolor='blue', linewidth=2)
    ax.add_patch(key_box)
    ax.text(2.75, 2.2, 'KEY FEATURE: Uses average(Q1, Q2) for ensemble estimation', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='blue')
    ax.text(2.75, 1.5, '• Target: r + γ·(Q1_target + Q2_target)/2\n• Actor: max (Q1 + Q2)/2', 
            ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('dual_ddpg_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: dual_ddpg_architecture.png")
    plt.close()

def create_comparison_diagram():
    """Create a comparison diagram showing differences"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis('off')
    
    # DDPG
    ax1.text(3, 7.5, 'DDPG (Baseline)', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    
    q_box = FancyBboxPatch((2, 5), 2, 1.5, boxstyle="round,pad=0.1", 
                            facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax1.add_patch(q_box)
    ax1.text(3, 5.75, 'Single Q Network', ha='center', va='center', fontsize=10)
    
    output_box = FancyBboxPatch((2, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                                 facecolor='yellow', edgecolor='black', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(3, 3, 'Q(s,a)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    arrow1 = FancyArrowPatch((3, 5), (3, 3.5), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow1)
    
    # Double DDPG
    ax2.text(3, 7.5, 'Double DDPG', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='red')
    
    q1_box = FancyBboxPatch((1, 5), 1.5, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax2.add_patch(q1_box)
    ax2.text(1.75, 5.75, 'Q1', ha='center', va='center', fontsize=9)
    
    q2_box = FancyBboxPatch((3.5, 5), 1.5, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax2.add_patch(q2_box)
    ax2.text(4.25, 5.75, 'Q2', ha='center', va='center', fontsize=9)
    
    min_box = FancyBboxPatch((2, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='orange', edgecolor='black', linewidth=2)
    ax2.add_patch(min_box)
    ax2.text(3, 3.4, 'min(Q1, Q2)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    output_box2 = FancyBboxPatch((2, 1.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor='yellow', edgecolor='black', linewidth=2)
    ax2.add_patch(output_box2)
    ax2.text(3, 1.9, 'Q_min', ha='center', va='center', fontsize=10, fontweight='bold')
    
    arrow2a = FancyArrowPatch((1.75, 5), (2.5, 3.8), 
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax2.add_patch(arrow2a)
    arrow2b = FancyArrowPatch((4.25, 5), (3.5, 3.8), 
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax2.add_patch(arrow2b)
    arrow2c = FancyArrowPatch((3, 3), (3, 2.3), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax2.add_patch(arrow2c)
    
    # Dual DDPG
    ax3.text(3, 7.5, 'Dual DDPG', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='blue')
    
    q1_box3 = FancyBboxPatch((1, 5), 1.5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax3.add_patch(q1_box3)
    ax3.text(1.75, 5.75, 'Q1', ha='center', va='center', fontsize=9)
    
    q2_box3 = FancyBboxPatch((3.5, 5), 1.5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax3.add_patch(q2_box3)
    ax3.text(4.25, 5.75, 'Q2', ha='center', va='center', fontsize=9)
    
    avg_box = FancyBboxPatch((2, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='orange', edgecolor='black', linewidth=2)
    ax3.add_patch(avg_box)
    ax3.text(3, 3.4, '(Q1 + Q2) / 2', ha='center', va='center', fontsize=10, fontweight='bold')
    
    output_box3 = FancyBboxPatch((2, 1.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor='yellow', edgecolor='black', linewidth=2)
    ax3.add_patch(output_box3)
    ax3.text(3, 1.9, 'Q_avg', ha='center', va='center', fontsize=10, fontweight='bold')
    
    arrow3a = FancyArrowPatch((1.75, 5), (2.5, 3.8), 
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax3.add_patch(arrow3a)
    arrow3b = FancyArrowPatch((4.25, 5), (3.5, 3.8), 
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='blue')
    ax3.add_patch(arrow3b)
    arrow3c = FancyArrowPatch((3, 3), (3, 2.3), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax3.add_patch(arrow3c)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: algorithm_comparison.png")
    plt.close()

if __name__ == '__main__':
    print("Generating architecture diagrams...")
    try:
        create_double_ddpg_diagram()
        create_dual_ddpg_diagram()
        create_comparison_diagram()
        print("\nAll diagrams generated successfully!")
        print("Files created:")
        print("  - double_ddpg_architecture.png")
        print("  - dual_ddpg_architecture.png")
        print("  - algorithm_comparison.png")
    except Exception as e:
        print(f"Error generating diagrams: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")



