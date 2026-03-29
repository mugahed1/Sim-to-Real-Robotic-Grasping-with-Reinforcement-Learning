"""
Gym wrapper for MujocoKinovaGraspEnv to make it compatible with Stable-Baselines3
"""
import gymnasium as gym
from gymnasium import spaces
from env_dynamic_goal import MujocoKinovaGraspEnv

class SB3MujocoEnv(gym.Env):
    """
    Wrapper that converts MujocoKinovaGraspEnv to a standard gym.Env
    for use with Stable-Baselines3
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, visualize=True, mode="train",exp="exp1"):
        super().__init__()
        self._env = MujocoKinovaGraspEnv(visualize=visualize, mode=mode,exp=exp)
        self.observation_space = self._env.observation_space  # Box(19,)
        self.action_space = self._env.action_space            # Box(7,)
        self.max_step_count = self._env.max_step_count

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation and info.
        
        Gymnasium API: reset(seed=None, options=None) -> (observation, info)
        """
        # Set seed if provided (for reproducibility)
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
        
        # Call underlying reset (which doesn't accept seed/options)
        obs = self._env.reset()
        
        # Return (observation, info) as per gymnasium API
        # info can be empty dict if no additional info is available
        info = {}
        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.
        Gymnasium API: step(action) -> (observation, reward, terminated, truncated, info)
        
        Semantic distinction:
        - terminated=True: Episode ended due to task completion (success or collision)
        - truncated=True: Episode ended due to timeout
        """
        obs, reward, done, info, terminated = self._env.step(action)
        # Gymnasium API uses separate terminated and truncated flags
        # terminated: episode ended due to task completion (success/collision)
        # truncated: episode ended due to timeout
        truncated = bool(done and not terminated)  # Timeout case
        # Store in info for backward compatibility if needed
        info['terminated'] = terminated
        info['truncated'] = truncated
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Rendering is handled internally by the MuJoCo viewer"""
        pass

    def close(self):
        """Close the environment"""
        self._env.closeSim()

    @property
    def collision(self):
        """Expose collision attribute from wrapped env"""
        return self._env.collision

    @property
    def success_cnt(self):
        """Expose success_cnt attribute from wrapped env"""
        return self._env.success_cnt

    def distance_to_goal(self):
        """Expose distance_to_goal method from wrapped env"""
        return self._env.distance_to_goal()
