"""
Plot actor_loss and critic_loss from training log (e.g. output11.log).
Usage: python plot_loss.py [path_to_log]
       If no path given, uses output11.log in current dir.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    """Extract total_timesteps, actor_loss, critic_loss from SB3-style log."""
    timesteps = []
    actor_losses = []
    critic_losses = []
    # Pattern: "|    total_timesteps | 12000    |" etc.
    pat_ts = re.compile(r"\|\s+total_timesteps\s+\|\s+([\d\s]+)\|")
    pat_actor = re.compile(r"\|\s+actor_loss\s+\|\s+([-\d.eE+]+)\s*\|")
    pat_critic = re.compile(r"\|\s+critic_loss\s+\|\s+([-\d.eE+]+)\s*\|")

    pending_ts = None
    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip()
            m = pat_ts.search(line)
            if m:
                pending_ts = int(m.group(1).replace(" ", "").strip())
                continue
            if pending_ts is None:
                continue
            m = pat_actor.search(line)
            if m:
                actor_losses.append(float(m.group(1)))
                continue
            m = pat_critic.search(line)
            if m:
                critic_losses.append(float(m.group(1)))
                timesteps.append(pending_ts)
                pending_ts = None

    return np.array(timesteps), np.array(actor_losses), np.array(critic_losses)


def main():
    base = Path(__file__).resolve().parent
    default_log = base / "output11.log"

    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = default_log

    if not log_path.exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    timesteps, actor_loss, critic_loss = parse_log(str(log_path))
    if len(timesteps) == 0:
        print("No train/ loss blocks found in log.")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(timesteps, actor_loss, color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Actor loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timesteps, critic_loss, color="coral", linewidth=0.8)
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Critic loss")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Training loss ")
    plt.tight_layout()
    out_path = log_path.parent / "loss_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
