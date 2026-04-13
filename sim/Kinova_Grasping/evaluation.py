import numpy as np
import tensorflow as tf
import time
from env_dynamic_goal import MujocoKinovaGraspEnv
from networks import ActorNetwork

ACTOR_WEIGHTS = "tmp/exp1/actor_sac.h5"




def main():
    
    env_tmp = MujocoKinovaGraspEnv(visualize=False, mode="test")
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]   

   
    actor = ActorNetwork(fc1_dims=128, fc2_dims=128, n_actions=act_dim)

   
    _ = actor(tf.zeros((1, obs_dim), dtype=tf.float32))

   
    actor.load_weights(ACTOR_WEIGHTS)
    print("[OK] Loaded actor weights:", ACTOR_WEIGHTS)

    
    env = MujocoKinovaGraspEnv(visualize=True, mode="test",exp="exp1")

    dt = env.model.opt.timestep          
    substeps = 40                        
    control_dt = dt * substeps           

    n_eval_episodes = 10
    episode_returns = []
    episode_successes = []

    
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
        
        obs = env.reset()  
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
                print("gripper state: ", env.gripper_state())
                break

        episode_returns.append(ep_return)
        episode_successes.append(1 if env.object_grasped else 0)
        print(f"[EVAL] ep={ep:02d} return={ep_return:.2f}")

    mean_return = float(np.mean(episode_returns))
    success_rate = float(np.mean(episode_successes))
    print(
        f"\n[EVAL] Summary over {n_eval_episodes} episodes: "
        f"average reward = {mean_return:.2f}, success rate = {success_rate:.1%} "
        f"({int(sum(episode_successes))}/{n_eval_episodes})"
    )

    env.closeSim()

if __name__ == "__main__":
    main()