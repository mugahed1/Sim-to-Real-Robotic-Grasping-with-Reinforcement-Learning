import numpy as np
import tensorflow as tf
import time
from env_dynamic_goal import MujocoKinovaGraspEnv
from networks import ActorNetwork

ACTOR_WEIGHTS = "tmp/exp1/actor_sac.h5"



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
        
        obs = env.reset()  # Will now use the fixed goal for this episode
        ep_return = 0.0
        print("object position: ", env.data.xpos[env.goal_body])
        for t in range(env.max_step_count):
            obs_t = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)

            a = actor.deterministic_action(obs_t).numpy()[0]

            a = np.clip(a, env.action_space.low, env.action_space.high)

            obs, r, done, info,_ = env.step(a)
            time.sleep(2.0 * control_dt)   
            ep_return += r
            if done:
                print("steps : ", t)
                break

        print(f"[EVAL] ep={ep:02d} return={ep_return:.2f}")

    env.closeSim()

if __name__ == "__main__":
    main()