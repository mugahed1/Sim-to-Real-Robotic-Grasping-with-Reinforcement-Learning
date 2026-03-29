import numpy as np
import tensorflow as tf
import time
from env_dynamic_goal import MujocoKinovaGraspEnv
from networks import ActorNetwork
from pathlib import Path
ACTOR_WEIGHTS = "tmp/exp7/actor_sac.h5"

step =1

def log_obs_and_action(obs_normalized: np.ndarray, action: np.ndarray, env: MujocoKinovaGraspEnv, step: int) -> None:
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
        
        "J0", "J1", "J2", "J3", "J4", "J5",
        
    ]
    
    # Access underlying env and its normalization stats
    rms =env.obs_rms
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
    # 1) Create env without viewer first (avoid segfault if something crashes)
    env_tmp = MujocoKinovaGraspEnv(visualize=False, mode="test")
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]   # should be 6

    # 2) Rebuild actor EXACTLY like training
    actor = ActorNetwork(fc1_dims=128, fc2_dims=128, n_actions=act_dim)

    # 3) Build variables (very important before load_weights)
    _ = actor(tf.zeros((1, obs_dim), dtype=tf.float32))

    # 4) Load weights
    actor.load_weights(ACTOR_WEIGHTS)
    print("[OK] Loaded actor weights:", ACTOR_WEIGHTS)

    # 5) Now launch viewer env
    env = MujocoKinovaGraspEnv(visualize=True, mode="test")

    dt = env.model.opt.timestep          # 0.002
    substeps = 20                        # because you do mj_step 20 times per env.step
    control_dt = dt * substeps           # simulated seconds per env.step

    n_eval_episodes = 10
    
    # Define 10 different fixed goal positions for evaluation
    # Positions extracted from actual evaluation run (rounded to 2 decimal places)
    fixed_goal_positions = [
        np.array([0.39, 0.34, 0.03], dtype=np.float64),  # Episode 0
        np.array([0.31, 0.38, 0.03], dtype=np.float64),  # Episode 1
        np.array([0.38, 0.39, 0.03], dtype=np.float64),  # Episode 2
        np.array([0.35, 0.34, 0.03], dtype=np.float64),  # Episode 3
        np.array([0.36, 0.34, 0.03], dtype=np.float64),  # Episode 4
        np.array([0.38, 0.35, 0.03], dtype=np.float64),  # Episode 5
        np.array([0.35, 0.38, 0.03], dtype=np.float64),  # Episode 6
        np.array([0.39, 0.38, 0.03], dtype=np.float64),  # Episode 7
        np.array([0.38, 0.40, 0.03], dtype=np.float64),  # Episode 8
        np.array([0.31, 0.36, 0.03], dtype=np.float64),  # Episode 9
    ]
    
    for ep in range(n_eval_episodes):
        # Set the fixed goal position for this episode
        env.set_fixed_goal(fixed_goal_positions[ep])
        print(f"[EVAL] Episode {ep:02d} - Fixed goal position: {fixed_goal_positions[ep]}")
        step = 1
        obs = env.reset()  # Will now use the fixed goal for this episode
        ep_return = 0.0
        print("object position: ", env.data.xpos[env.goal_body])
        for t in range(env.max_step_count):
            obs_t = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)

            a = actor.deterministic_action(obs_t).numpy()[0]
            log_obs_and_action(obs_t, a, env, step)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            
            obs, r, done, info,_ = env.step(a)
            time.sleep(2.0 * control_dt)   
            ep_return += r
            if done:
                print("steps : ", t)
                break
            step += 1

        print(f"[EVAL] ep={ep:02d} return={ep_return:.2f}")
        
    env.closeSim()

if __name__ == "__main__":
    main()