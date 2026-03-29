import gym
import numpy as np
from agent import Agent, PotentialAgent
import matplotlib.pyplot as plt
from env_dynamic_goal import MujocoKinovaGraspEnv
from setup_flags import set_up
import sys
import os
import subprocess
import math
import tensorflow as tf
import json
from tqdm import tqdm
import logging
import cv2
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
    print("Success in setting memory growth")
except:
    print("Failed to set memory growth, invalid device or cannot modify virtual devices once initialized.")


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



if __name__ == '__main__':
    env = MujocoKinovaGraspEnv(visualize=False)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    potentialagent = PotentialAgent(gamma=agent.gamma)
    
    n_games = 3000
    collision_cnt = 0
    score_history, avg_score_history = [], []
    load_checkpoint = FLAGS.evaluate

    try:
        if load_checkpoint:
            n_steps = 0
            n_games = 3
            while n_steps <= agent.batch_size:
                observation = env.reset()
                action = env.action_space.sample()
                observation_, reward, done, info, terminated = env.step(action)
                agent.remember(observation, action, reward, observation_, done)
                n_steps += 1
            agent.learn()
            agent.load_models(FLAGS.exp)
            evaluate = True
        else:
            evaluate = False

        for i in range(n_games):
            observation = env.reset()
            done = False
            terminated = False
            score = 0
            step_per_episode = 0
            reward_in_episode = []
            step_bar = tqdm(total=env.max_step_count-1, desc=f"Ep {i}", leave=False, ncols=80)

            obs_floor = [math.floor(obs*100)/100.0 for obs in observation[:6]]
            trajectory_epi = [obs_floor]

            episode_timed_out = False
            while not done:
                action = agent.choose_action(observation, evaluate)
                
                observation_, reward, done, info,terminated = env.step(action)

                score += reward
                step_per_episode += 1

                obs_floor_ = [math.floor(obs*100)/100.0 for obs in observation_[:6]]
                trajectory_epi.append(obs_floor_)

                shaping_reward = potentialagent.reward_shaping(obs_floor, obs_floor_, evaluate)
                logging.debug(f"epi {i} step {step_per_episode}, reward: {reward} + {shaping_reward}")
                logging.debug(f"old obs: {observation}, floor {obs_floor}")
                logging.debug(f"new obs: {observation_}, floor {obs_floor_}")
                if evaluate:
                    reward_in_episode.append(f"{reward}+{shaping_reward}")
                # reward += shaping_reward

                agent.remember(observation, action, reward, observation_, terminated)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_
                obs_floor = obs_floor_.copy()

                step_bar.update(1)

            
            step_bar.close()
            if env.collision:
                collision_cnt+=1

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)

            # After one episode ##
            if trajectory_epi:
                potentialagent.add_trajectory(trajectory_epi, score, done)
                
            potentialagent.learn_pf()

            if potentialagent.potential_learn_cnt == 10:
                os.makedirs("tmp", exist_ok=True)
                with open(f'tmp/meta.json', 'a') as f:
                    f.write(json.dumps({**{'expname': expname, \
                                            'potential_learn_cnt': potentialagent.potential_learn_cnt, \
                                            'at episode': i}}))

            print(expname, 'episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, \
                 'final distance', env.distance_to_goal(), "succes_cnt", env.success_cnt, "collision_cnt", collision_cnt)
            if evaluate:
                print(reward_in_episode)    
            
           

        # finish training:
        if not load_checkpoint:
            os.makedirs("results", exist_ok=True)
            np.save(f"results/rewards_average_{expname}.npy", avg_score_history)
            agent.save_models(exp=expname)
            potentialagent.save_models(exp=expname)
            env.obs_rms.save(exp=expname)
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, score_history, figure_file=f'apf_ddpg_{expname}.png')

        potentialagent.memory_traj.get_nlargest_trajectories(N_good=1)

        env.closeSim()
        print("Finish training!")
        shutdown_vm()

    except (Exception, KeyboardInterrupt) as error:
        print('\nTraining exited early.')
        print(error)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        breakpoint()
        env.closeSim()




