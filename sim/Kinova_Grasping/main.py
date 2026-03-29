import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sb3_env_wrapper import SB3MujocoEnv
from setup_flags import set_up
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import sys
import os
import subprocess
import math
import json
from tqdm import tqdm
import logging
from pathlib import Path
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)


FLAGS = set_up()
expname = FLAGS.exp

def shutdown_vm():
    print("[INFO] Training finished. Shutting down VM...")
    subprocess.run(["sudo", "shutdown", "-h", "now"])

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


class EpisodeTrackingCallback(BaseCallback):
    """Callback to track episode metrics during SB3 training"""
    def __init__(self, verbose=0):
        super(EpisodeTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track rewards and lengths
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0].get('episode')
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True



if __name__ == '__main__':
    # Create wrapped environment for SB3
    env = SB3MujocoEnv(visualize=True, mode="train")
    
    # Setup output directory
    output_dir = Path("results") / expname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SAC configuration
    sac_cfg = {
        "learning_rate": 5e-5,
        "buffer_size": 1000000,
        "learning_starts": 10000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
    }
    
    train_cfg = {
        "seed": 42,
    }
    
    device = "auto"  # or "cuda" or "cpu"
    
    load_checkpoint = FLAGS.evaluate
    model_path = output_dir / "sac_model"
    
    # Wrap env with Monitor for training (for episode tracking)
    if not load_checkpoint:
        env = Monitor(env, str(output_dir / "monitor"))
    
    try:
        # Helper function to unwrap environment (handles Monitor wrapper)
        def unwrap_env(env):
            """Unwrap environment to get the underlying SB3MujocoEnv"""
            if hasattr(env, 'env'):  # Monitor wrapper
                return env.env
            return env
        
        if load_checkpoint:
            # Load model for evaluation
            print(f"Loading model from {model_path}")
            model = SAC.load(str(model_path), env=env, device=device)
            evaluate = True
        else:
            # Create new SAC model
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=sac_cfg["learning_rate"],
                buffer_size=sac_cfg["buffer_size"],
                learning_starts=sac_cfg["learning_starts"],
                batch_size=sac_cfg["batch_size"],
                tau=sac_cfg["tau"],
                gamma=sac_cfg["gamma"],
                train_freq=sac_cfg["train_freq"],
                gradient_steps=sac_cfg["gradient_steps"],
                verbose=1,
                seed=train_cfg["seed"],
                device=device,
                tensorboard_log=str(output_dir / "tensorboard"),
            )
            evaluate = False
        
        if load_checkpoint:
            # Evaluation mode: run episodes manually
            n_games = 3
            collision_cnt = 0
            score_history = []
            
            for i in range(n_games):
                observation, _ = env.reset()
                done = False
                score = 0
                unwrapped = unwrap_env(env)
                step_bar = tqdm(total=unwrapped.max_step_count-1, desc=f"Ep {i}", leave=False, ncols=80)
                
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    score += reward
                    step_bar.update(1)
                
                step_bar.close()
                unwrapped = unwrap_env(env)
                if unwrapped.collision:
                    collision_cnt += 1
                
                score_history.append(score)
                avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else 0
                
                print(expname, 'episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                     'final distance', unwrapped.distance_to_goal(), "success_cnt", unwrapped.success_cnt, "collision_cnt", collision_cnt)
        else:
            # Training mode: use SB3's learn() method
            # Calculate total timesteps (n_games * average_episode_length)
            n_games = 4000
            # Get episode length from unwrapped env
            unwrapped_env = unwrap_env(env)
            avg_episode_length = unwrapped_env.max_step_count
            total_timesteps = n_games * avg_episode_length
            
            # Custom callback to track custom metrics and save model periodically
            class CustomCallback(BaseCallback):
                def __init__(self, model, model_path, save_freq=1000, verbose=0):
                    super(CustomCallback, self).__init__(verbose)
                    self.model = model
                    self.model_path = model_path
                    self.save_freq = save_freq
                    self.episode_count = 0
                    self.score_history = []
                    
                def _on_step(self) -> bool:
                    # Check if episode ended
                    if self.locals.get('dones', [False])[0]:
                        self.episode_count += 1
                        # Get episode info from monitor
                        if hasattr(self.training_env, 'get_episode_rewards'):
                            if len(self.training_env.get_episode_rewards()) > 0:
                                ep_reward = self.training_env.get_episode_rewards()[-1]
                                self.score_history.append(ep_reward)
                                if len(self.score_history) % 10 == 0:
                                    avg_score = np.mean(self.score_history[-100:]) if len(self.score_history) > 0 else 0
                                    print(f"Episode {self.episode_count}, Avg Score: {avg_score:.1f}")
                        
                        # Save model every save_freq episodes
                        if self.episode_count % self.save_freq == 0:
                            checkpoint_path = str(self.model_path) + f"_ep{self.episode_count}"
                            self.model.save(checkpoint_path)
                            print(f"Model saved to {checkpoint_path} (Episode {self.episode_count})")
                            # Keep obs normalization in sync with the saved checkpoint.
                            # This overwrites tmp/<expname>/mean.npy,var.npy,count.npy each time.
                            try:
                                unwrapped_env = self.training_env
                                if hasattr(unwrapped_env, 'env'):  # Monitor wrapper
                                    unwrapped_env = unwrapped_env.env
                                if hasattr(unwrapped_env, '_env'):  # SB3MujocoEnv wrapper
                                    unwrapped_env._env.obs_rms.save(exp=expname)
                                    print("Normalization stats saved (obs_rms)")
                            except Exception as e:
                                print(f"[WARN] Failed to save normalization stats at episode {self.episode_count}: {e}")
                    return True
            
            # Create callback after model is created
            callback = CustomCallback(model, model_path, save_freq=500)
            
            print(f"Starting training for {total_timesteps} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=10,
                progress_bar=True
            )
            
            # After training, run a few evaluation episodes to get final metrics
            print("\nRunning evaluation episodes...")
            collision_cnt = 0
            score_history = []
            n_eval_episodes = 10
            
            for i in range(n_eval_episodes):
                observation, _ = env.reset()
                done = False
                score = 0
                
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    score += reward
                
                unwrapped = unwrap_env(env)
                if unwrapped.collision:
                    collision_cnt += 1
                
                score_history.append(score)
                avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else 0
                
                print(expname, 'eval episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                     'final distance', unwrapped.distance_to_goal(), "success_cnt", unwrapped.success_cnt, "collision_cnt", collision_cnt)
            
            # Save score history
            avg_score_history = []
            for i in range(len(score_history)):
                avg_score_history.append(np.mean(score_history[max(0, i-100):(i+1)]))
            
            os.makedirs("results", exist_ok=True)
            np.save(f"results/rewards_average_{expname}.npy", avg_score_history)
        
        # Finish training
        if not load_checkpoint:
            model.save(str(model_path))
            print(f"Model saved to {model_path}")
            
            # Save normalization stats
            # Handle both wrapped (Monitor) and unwrapped environments
            unwrapped_env = env
            if hasattr(env, 'env'):  # Monitor wrapper
                unwrapped_env = env.env
            if hasattr(unwrapped_env, '_env'):  # SB3MujocoEnv wrapper
                unwrapped_env._env.obs_rms.save(exp=expname)
                print(f"Normalization stats saved")
            
            # Plot learning curve if we have score history
            if 'score_history' in locals() and len(score_history) > 0:
                x = [i+1 for i in range(len(score_history))]
                plot_learning_curve(x, score_history, figure_file=f'apf_ddpg_{expname}.png')
        
        env.close()
        print("Finish training!")
        shutdown_vm()
        
    except (Exception, KeyboardInterrupt) as error:
        print('\nTraining exited early.')
        print(error)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        breakpoint()
        env.close()




