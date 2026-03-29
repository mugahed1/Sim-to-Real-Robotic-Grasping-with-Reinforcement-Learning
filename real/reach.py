import os, sys
import argparse
from pathlib import Path
import time
import termios
import tty
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC

import utilities
from kinova_gen3 import KinovaGen3
from normalization import RunningMeanStd




def deg2rad(deg):
    return np.asarray(deg, dtype=np.float32) * (np.pi / 180.0)


def convert_euler_to_quaternion(theta_x, theta_y, theta_z):
    
    
    # Convert raw euler to quaternion
    rot = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)
    
    # Apply -90° z correction as quaternion multiplication
    rot_corrected = rot * R.from_euler('z', -90, degrees=True)
    
    q = rot_corrected.as_quat()  # (x,y,z,w)
    w, x, y, z = q[3], q[0], q[1], q[2]
    
    # Apply axis swap to match simulation frame
    return np.array([w, x, y, z], dtype=np.float32)

def convert_action(action):
    action = np.asarray(action, dtype=np.float32)

    joint_delta = action * (180.0 / 64.0)       

    return joint_delta.astype(np.float32)


def format_obs_labeled(obs: np.ndarray) -> str:
    """
    Format 23-dim observation with labels:
    ee_pos, goal_pos_f32, relative_vector, ee_linear_vel, ee_quat, joint_angles, gripper_state
    """
    return (
        f"  ee_pos={np.array2string(obs[0:3], precision=3, separator=', ')}\n"
        f"  goal_pos_f32={np.array2string(obs[3:6], precision=3, separator=', ')}\n"
        f"  relative_vector={np.array2string(obs[6:9], precision=3, separator=', ')}\n"
        f"  ee_linear_vel={np.array2string(obs[9:12], precision=3, separator=', ')}\n"
        f"  ee_quat={np.array2string(obs[12:16], precision=3, separator=', ')}\n"
        f"  joint_angles={np.array2string(obs[16:22], precision=3, separator=', ')}\n"
        f"  gripper_state={np.array2string(obs[22:23], precision=3, separator=', ')}"
    )






def log_obs_and_action(obs: np.ndarray, obs_normalized: np.ndarray, action: np.ndarray, step: int) -> None:
    """
    Append to a default file the observation (raw and normalized) with labels, and the action.
    
    Args:
        obs_normalized: Observation returned by env (already normalized).
        action: Action applied.
    """
    # Labels as constructed in MujocoKinovaGraspEnv.get_observation (23 dims)
    labels = [
        "ee_x", "ee_y", "ee_z",
        "goal_x", "goal_y", "goal_z",
        "rel_dx", "rel_dy", "rel_dz",
        "ee_vx", "ee_vy", "ee_vz",
        "ee_qw", "ee_qx", "ee_qy", "ee_qz",
        "J0", "J1", "J2", "J3", "J4", "J5",
       
    ]
    
    
    # Default log path
    path = Path("logs/eval_obs2.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("a", encoding="utf-8") as f:
        f.write(f"=== Observation and Action step {step} ===\n")
        for i, label in enumerate(labels):
            f.write(f"{label}: {obs[i]: .6f} (norm: {obs_normalized[i]: .6f})\n")
        f.write("action: [" + ", ".join(f"{a: .6f}" for a in action) + "]\n")
        f.write("\n")
    step+=1

def main():
    MODEL_PATH = "model/reach_test/sac_model"
    RMS_PATH = "model/reach_test"  

    # Load normalization + model
    rms = RunningMeanStd(shape=(23,))
    rms.load(RMS_PATH)

    model = SAC.load(MODEL_PATH, env=None)

    # Parse connection arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Define a fixed goal (example) — must match training meaning/units
    goal_pos = np.array([0.337, 0.395, 0.03], dtype=np.float32)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        kinova = KinovaGen3(router)
        kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

        max_step = 150
        for step in range(max_step):

            ee_pos = np.array(kinova.GetFingerCenterPosition(), dtype=np.float32)  # (3,)
                
        
            theta_x, theta_y, theta_z = kinova.GetEndEffectorOrientation()

            ee_quat = convert_euler_to_quaternion(theta_x, theta_y, theta_z)                                   # (4,)
            joint_angles = deg2rad(np.array(kinova.GetJointAngles(), dtype=np.float32))          
        

            

            ee_linear_vel = kinova.GetEndEffectorLinearVelocity()       
            # print(f"ee_linear_vel: {ee_linear_vel}")
            relative_vector = (goal_pos - ee_pos).astype(np.float32)  
            print(f"relative_vector: {np.linalg.norm(relative_vector)}")
            if np.linalg.norm(relative_vector) < 0.1:
                print("-------------------Goal reached-------------------")
                break                 # (3,)
            # print(f"gripper_state: {gripper_state}")
            obs = np.concatenate([
                ee_pos,            # 3
                goal_pos,          # 3
                relative_vector,   # 3
                ee_linear_vel,     # 3
                ee_quat,           # 4
                joint_angles,      # 6
            ]).astype(np.float32)  # total 22
            # Print to console and append to output log file
            
            
            obs_norm = rms.normalize(obs).astype(np.float32)  # shape (23,)

            action, _ = model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            
            action_cmd = convert_action(action)

            log_obs_and_action(obs, obs_norm, action,step)

            kinova.SetJointAnglesIncremental(action_cmd.astype(np.float32).tolist())   # delta degrees
           

            
if __name__ == "__main__":
    main()