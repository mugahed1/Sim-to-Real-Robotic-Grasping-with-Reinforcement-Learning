import numpy as np
import time
from stable_baselines3 import SAC
from sb3_env_wrapper import SB3MujocoEnv

import argparse
from pathlib import Path

def log_obs_and_action(obs_normalized: np.ndarray, action: np.ndarray, env: SB3MujocoEnv, step: int) -> None:
    """
    Append to a default file the observation (raw and normalized) with labels, and the action.
    
    Args:
        obs_normalized: Observation returned by env (already normalized).
        action: Action applied.
        env: The SB3 wrapper environment (to access normalization stats).
    """
    # Labels as constructed in MujocoKinovaGraspEnv.get_observation (23 dims)
    labels = [
        "ee_x", "ee_y", "ee_z",
        "goal_x", "goal_y", "goal_z",
        "rel_dx", "rel_dy", "rel_dz",
        "ee_vx", "ee_vy", "ee_vz",
        "ee_qw", "ee_qx", "ee_qy", "ee_qz",
        "J0", "J1", "J2", "J3", "J4", "J5",
        "gripper",
    ]
    
    # Access underlying env and its normalization stats
    base_env = env._env
    rms = base_env.obs_rms
    mean = np.asarray(rms.mean, dtype=np.float64)
    var = np.asarray(rms.var, dtype=np.float64)
    eps = float(rms.epsilon)
    
    obs_normalized = np.asarray(obs_normalized, dtype=np.float64).reshape(-1)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    
    if obs_normalized.shape[0] != len(labels):
        # Fallback: avoid crash if shapes drift
        raise ValueError(f"Expected normalized observation of size {len(labels)}, got {obs_normalized.shape[0]}")
    
    # Denormalize using the same epsilon used during normalization
    # normalized = (obs - mean) / (sqrt(var) + eps)
    # => obs = normalized * (sqrt(var) + eps) + mean
    scale = np.sqrt(var) + eps
    obs_raw = (obs_normalized * scale) + mean
    
    # Default log path
    path = Path("logs/eval_obs.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("a", encoding="utf-8") as f:
        f.write(f"=== Observation and Action step {step} ===\n")
        for i, label in enumerate(labels):
            f.write(f"{label}: {obs_raw[i]: .6f} (norm: {obs_normalized[i]: .6f})\n")
        f.write("action: [" + ", ".join(f"{a: .6f}" for a in action) + "]\n")
        f.write("\n")

def main():
   

    model_path = "results/exp4/sac_model"
    visualize = True

    # Create environment
    env = SB3MujocoEnv(visualize=visualize, mode="test", exp="exp4")
    
    # Load the trained SAC model
    print(f"Loading model from {model_path}")
    model = SAC.load(model_path, env=env)
    print("[OK] Model loaded successfully")
    
    # Get timing info from wrapped env
    dt = env._env.model.opt.timestep          # 0.002
    substeps = 20                             # because you do mj_step 20 times per env.step
    control_dt = dt * substeps                # simulated seconds per env.step
    
    n_eval_episodes = 20
    collision_cnt = 0
    success_cnt = 0

    
    
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        steps = 1

      
        for t in range(env.max_step_count):
            # Get action from SB3 model (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)

            # log_obs_and_action(obs, action, env, steps)
            
            # Clip action to valid range (SB3 should handle this, but just in case)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            
            
            
            ep_return += reward
            steps += 1
            
            if done:
                print(f"Episode ended at step {steps}")
                break
            time.sleep(control_dt)

       
           
        
        if env.collision:
            collision_cnt += 1
        if env.success_cnt > success_cnt:
            success_cnt = env.success_cnt
        
        print(f"[EVAL] ep={ep:02d} return={ep_return:.2f} steps={steps} "
              f"distance={env.distance_to_goal():.3f} success={env.success_cnt} collision={env.collision}")
    
    print(f"\nEvaluation complete: {n_eval_episodes} episodes")
    print(f"Total successes: {success_cnt}, Total collisions: {collision_cnt}")
    
    env.close()


if __name__ == "__main__":
    main()
