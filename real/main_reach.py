import os, sys
import argparse
import time
import termios
import tty
import numpy as np

import utilities
from kinova_gen3 import KinovaGen3
from normalization import RunningMeanStd
from reach_actor_numpy import ReachActorNP

OUTPUT_LOG_PATH = "results.txt"


def deg2rad(deg):
    return np.asarray(deg, dtype=np.float32) * (np.pi / 180.0)


def convert_action(action):
    action = np.asarray(action, dtype=np.float32)

    joint_delta = action * (180.0 / 64.0)          
    

    return joint_delta.astype(np.float32)


def wait_for_space():
    """
    Block until user presses SPACE to take one control step,
    or 'q'/'Q' to quit the loop.
    """
    print("Press SPACE to take one step, or 'q' to quit... ", end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                print()
                return True
            if ch in ("q", "Q"):
                print()
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)



RMS_PATH = "model/reach_sac-dr"  

# Load normalization + model
rms = RunningMeanStd(shape=(15,))
rms.load(RMS_PATH)

# Pure NumPy actor (weights must be exported separately from TF env)
ACTOR_WEIGHTS_NPZ = "model/reach_sac-dr/reach_actor_weights.npz"
actor = ReachActorNP(ACTOR_WEIGHTS_NPZ)
print("[OK] Loaded NumPy actor weights:", ACTOR_WEIGHTS_NPZ)

# Parse connection arguments
parser = argparse.ArgumentParser()
args = utilities.parseConnectionArguments(parser)

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

    
with utilities.DeviceConnection.createTcpConnection(args) as router:
    kinova = KinovaGen3(router)
    
    for goal_pos in fixed_goal_positions:
        kinova.SetJointPositions([0, 0, 0, 0, 0, 0])
        print(f"Moving to goal position: {goal_pos}")
        reward=0.0
        max_step = 100
        for step in range(max_step):

            # Wait for user input before each control step
            # if not wait_for_space():
            #     break

            ee_pos = np.array(kinova.GetEndEffectorPosition(), dtype=np.float32)  # (3,)
                
            if np.linalg.norm(ee_pos - goal_pos) < 0.10:
                print("-------------------Goal reached-------------------")
                break

            joint_angles = deg2rad(np.array(kinova.GetJointAngles(), dtype=np.float32))          
        

        

            relative_vector = (goal_pos - ee_pos).astype(np.float32)
            obs = np.concatenate([
                ee_pos,            # 3
                goal_pos,          # 3
                relative_vector,   # 3
                joint_angles,      # 6
            ]).astype(np.float32)  # total 15

            obs_norm = rms.normalize(obs).astype(np.float32)  # shape (15,)

            # Get action from NumPy actor (same as deterministic_action)
            action = actor.forward(obs_norm)
            action = np.clip(action, -1.0, 1.0)

            action_cmd = convert_action(action)

            kinova.SetJointAnglesIncremental(action_cmd.tolist())   # delta degrees

            reward -= np.linalg.norm(ee_pos - goal_pos)

        print(f"Reward epsiode {goal_pos}: {reward} total steps: {step+1}")
        with open(OUTPUT_LOG_PATH, "a") as f:
            f.write(f"goal={goal_pos.tolist()} reward={reward:.4f} steps={step+1}\n")
