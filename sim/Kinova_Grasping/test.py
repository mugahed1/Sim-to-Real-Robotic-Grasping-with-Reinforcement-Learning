# import mujoco
# import numpy as np

# model = mujoco.MjModel.from_xml_path("scene.xml")
# data = mujoco.MjData(model)

# ee_body = model.body("END_EFFECTOR").id
# joint_names = ["J0", "J1", "J2", "J3", "J4", "J5"]
# ref_joint_qpos_adr = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]

# TEST_CONFIGS = [
#     ("home",          [0,    0,   0,   0,   0,   0]),
#     ("J0=30",         [30,   0,   0,   0,   0,   0]),
#     ("J1=45",         [0,   45,   0,   0,   0,   0]),
#     ("J0=30,J1=45",   [30,  45,   0,   0,   0,   0]),
#     ("all=20",        [20, 25, 23, 45, 30, 12]),
# ]

# print("=== SIMULATION xquat[END_EFFECTOR] ===\n")
# for label, joints_deg in TEST_CONFIGS:
#     mujoco.mj_resetData(model, data)
#     for i, adr in enumerate(ref_joint_qpos_adr):
#         data.qpos[adr] = np.deg2rad(joints_deg[i])
#     mujoco.mj_forward(model, data)
    
#     quat = data.xquat[ee_body].copy()
#     print(f"{label}:")
#     print(f"  joints (deg): {joints_deg}")
#     print(f"  quat (w,x,y,z): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")




# import mujoco
# import numpy as np
# import math

# model = mujoco.MjModel.from_xml_path("scene.xml")
# data = mujoco.MjData(model)

# ee_body = model.body("END_EFFECTOR").id
# joint_names = ["J0", "J1", "J2", "J3", "J4", "J5"]
# ref_joint_qpos_adr = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]

# # Reset to home
# mujoco.mj_resetData(model, data)
# for adr in ref_joint_qpos_adr:
#     data.qpos[adr] = 0.0
# data.qvel[:] = 0.0
# for i in range(6):
#     data.ctrl[i] = 0.0
# mujoco.mj_forward(model, data)

# # Warmup
# for _ in range(10):
#     mujoco.mj_step(model, data)

# TEST_ACTIONS = [
#     ("all joints +1",  [1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
#     ("J0 only +1",     [1.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
#     ("J1 only +1",     [0.0,  1.0,  0.0,  0.0,  0.0,  0.0]),
#     ("all joints -1",  [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
# ]

# print("=== SIMULATION EE VELOCITY ===\n")
# for label, action in TEST_ACTIONS:
#     # Reset to home
#     mujoco.mj_resetData(model, data)
#     for adr in ref_joint_qpos_adr:
#         data.qpos[adr] = 0.0
#     data.qvel[:] = 0.0
#     for i in range(6):
#         data.ctrl[i] = 0.0
#     mujoco.mj_forward(model, data)
#     for _ in range(10):
#         mujoco.mj_step(model, data)

#     # Apply action — same as apply_controls
#     delta = np.array(action) * (math.pi / 64.0)
#     current = data.ctrl[:6].copy()
#     for i in range(6):
#         data.ctrl[i] = np.clip(current[i] + delta[i],
#                                model.actuator(i).ctrlrange[0],
#                                model.actuator(i).ctrlrange[1])

#     # Run 5 substeps
#     for _ in range(5):
#         mujoco.mj_step(model, data)

#     # Read velocity
#     cvel = data.cvel[ee_body].copy()
#     ee_vel = cvel[3:]

#     print(f"{label}:")
#     print(f"  delta_rad: {delta.round(4).tolist()}")
#     print(f"  ee_vel (vx,vy,vz): [{ee_vel[0]:.6f}, {ee_vel[1]:.6f}, {ee_vel[2]:.6f}]")
#     print(f"  magnitude: {np.linalg.norm(ee_vel):.6f}\n")



import mujoco
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

ee_body = model.body("END_EFFECTOR").id
joint_names = ["J0", "J1", "J2", "J3", "J4", "J5"]
ref_joint_qpos_adr = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]

ACTION = [0.5, -0.3, 0.8, -0.6, 0.4, -0.2]
GOAL_POS = np.array([0.373, 0.359, 0.030])

def get_obs(data):
    # EE position — finger center exactly as in your env
    RIGHT = data.body("RIGHT_FINGER_PROX").xpos.copy()
    LEFT  = data.body("LEFT_FINGER_PROX").xpos.copy()
    ee_pos = ((RIGHT + LEFT) / 2).astype(np.float32)

    # Relative vector
    relative_vector = (GOAL_POS - ee_pos).astype(np.float32)

    # EE velocity
    cvel = data.cvel[ee_body].copy()
    ee_linear_vel = cvel[3:].astype(np.float32)

    # EE quaternion
    ee_quat = data.xquat[ee_body].copy().astype(np.float32)

    # Joint angles in radians
    joint_angles = np.array([data.qpos[adr] for adr in ref_joint_qpos_adr], dtype=np.float32)

    return ee_pos, relative_vector, ee_linear_vel, ee_quat, joint_angles

# Reset and warmup
mujoco.mj_resetData(model, data)
for i in range(6):
    data.ctrl[i] = 0.0
mujoco.mj_forward(model, data)
for _ in range(10):
    mujoco.mj_step(model, data)

# Apply action
delta = np.array(ACTION) * (math.pi / 5.0)
for i in range(6):
    data.ctrl[i] = np.clip(
        data.ctrl[i] + delta[i],
        model.actuator(i).ctrlrange[0],
        model.actuator(i).ctrlrange[1]
    )
for _ in range(100):
    mujoco.mj_step(model, data)

ee_pos, rel, vel, quat, joints = get_obs(data)

print("=== SIMULATION ===\n")
print(f"action:           {ACTION}")
print(f"delta_rad:        {delta.round(5).tolist()}")
print(f"ee_pos:           [{ee_pos[0]:.6f}, {ee_pos[1]:.6f}, {ee_pos[2]:.6f}]")
print(f"relative_vector:  [{rel[0]:.6f}, {rel[1]:.6f}, {rel[2]:.6f}]")
print(f"ee_linear_vel:    [{vel[0]:.6f}, {vel[1]:.6f}, {vel[2]:.6f}]")
print(f"ee_quat(w,x,y,z): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
print(f"joint_angles(rad):[{', '.join(f'{j:.6f}' for j in joints)}]")