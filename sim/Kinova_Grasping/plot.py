"""
Plot average reward (rolling window) from Stable-Baselines3 monitor CSV.
Usage: python plot.py [path_to_monitor.csv]
       If no path given, uses results/exp11/monitor.monitor.csv
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No GUI — use before importing pyplot (works on VMs/SSH)
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(csv_path: str) -> np.ndarray:
    """Load reward column from monitor CSV (skips JSON header line)."""
    rewards = []
    with open(csv_path, "r") as f:
        next(f)  # skip #{"t_start": ...}
        reader = csv.DictReader(f)  # uses "r,l,t" as fieldnames
        for row in reader:
            rewards.append(float(row["r"]))
    return np.array(rewards)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; first (window-1) values use partial window."""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start : i + 1])
    return out


def main():
    window = 100
    default_csv = Path(__file__).resolve().parent / "results" / "exp11" / "monitor.monitor.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = str(default_csv)

    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    rewards = load_rewards(csv_path)
    episodes = np.arange(1, len(rewards) + 1, dtype=float)
    avg_reward = rolling_mean(rewards, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, avg_reward, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_title(f"Training progress ")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(csv_path).parent / "reward_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
