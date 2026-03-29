"""
Plot average reward from numpy arrays with exponential smoothing.
Loads rewards_average_exp1.npy and rewards_average_exp2.npy and plots them together.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No GUI — use before importing pyplot (works on VMs/SSH)
import numpy as np
import matplotlib.pyplot as plt


def exponential_moving_average(x: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    """Apply exponential moving average smoothing.
    Args:
        x: Input array
        alpha: Smoothing factor (0.9 means 90% weight on previous value, 10% on new)
    Returns:
        Smoothed array
    """
    smoothed = np.zeros_like(x)
    smoothed[0] = x[0]
    for i in range(1, len(x)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * x[i]
    return smoothed


def main():
    # Paths to the numpy files
    results_dir = Path(__file__).resolve().parent / "results"
    exp1_path = results_dir / "rewards_average_exp1.npy"
    exp2_path = results_dir / "rewards_average_exp2.npy"

    # Load the data
    if not exp1_path.exists():
        print(f"File not found: {exp1_path}")
        sys.exit(1)
    if not exp2_path.exists():
        print(f"File not found: {exp2_path}")
        sys.exit(1)

    rewards_exp1 = np.load(exp1_path)
    rewards_exp2 = np.load(exp2_path)

    # Apply exponential smoothing with alpha=0.99 (TensorBoard-like smoothing)
    smoothed_exp1 = exponential_moving_average(rewards_exp1, alpha=0.99)
    smoothed_exp2 = exponential_moving_average(rewards_exp2, alpha=0.99)

    # Create timesteps (assuming each entry is one episode/timestep)
    timesteps_exp1 = np.arange(len(rewards_exp1))
    timesteps_exp2 = np.arange(len(rewards_exp2))

    # TensorBoard-like styling
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    
    # Colors similar to TensorBoard (vibrant blues and oranges)
    color1 = '#1f77b4'  # TensorBoard blue
    color2 = '#ff7f0e'  # TensorBoard orange
    
    # Plot raw data (very transparent, like TensorBoard)
    ax.plot(timesteps_exp1, rewards_exp1, color=color1, alpha=0.15, linewidth=0.5, zorder=1)
    ax.plot(timesteps_exp2, rewards_exp2, color=color2, alpha=0.15, linewidth=0.5, zorder=1)
    
    # Plot smoothed curves (main lines, TensorBoard style)
    ax.plot(timesteps_exp1, smoothed_exp1, color=color1, linewidth=2.5, label="APF-SAC-DR", zorder=3)
    ax.plot(timesteps_exp2, smoothed_exp2, color=color2, linewidth=2.5, label="SAC-DR", zorder=3)
    
    # Styling
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average reward", fontsize=12)
    ax.set_title("Training progress", fontsize=14, fontweight='bold')
    # Limit x-axis to first 2000 episodes
    ax.set_xlim(0, 2000)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    
    out_path = results_dir / "reward_plot_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
