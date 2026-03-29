# import argparse
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import utilities
# from kinova_gen3 import KinovaGen3

# TEST_CONFIGS = [
#     ("home",          [0,    0,   0,   0,   0,   0]),
#     ("J0=30",         [30,   0,   0,   0,   0,   0]),
#     ("J1=45",         [0,   45,   0,   0,   0,   0]),
#     ("J0=30,J1=45",   [30,  45,   0,   0,   0,   0]),
#     ("all=20",        [20,  25,  23,  45,  30,  12]),
# ]

# def convert_euler_to_quaternion(theta_x, theta_y, theta_z):
    
    
#     # Convert raw euler to quaternion
#     rot = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)
    
#     # Apply -90° z correction as quaternion multiplication
#     rot_corrected = rot * R.from_euler('z', -90, degrees=True)
    
#     q = rot_corrected.as_quat()  # (x,y,z,w)
#     w, x, y, z = q[3], q[0], q[1], q[2]
    
#     # Apply axis swap to match simulation frame
#     return np.array([w, x, y, z], dtype=np.float32)

# def main():
#     parser = argparse.ArgumentParser()
#     args = utilities.parseConnectionArguments(parser)

#     with utilities.DeviceConnection.createTcpConnection(args) as router:
#         kinova = KinovaGen3(router)

#         print("=== REAL ROBOT quat (euler z-90) ===\n")
#         for label, joints_deg in TEST_CONFIGS:
#             kinova.SetJointPositions(joints_deg)
#             import time; time.sleep(0.3)
#             theta_x, theta_y, theta_z = kinova.GetEndEffectorOrientation()
#             quat = convert_euler_to_quaternion(theta_x, theta_y, theta_z)
#             print(f"{label}:")
#             print(f"  joints (deg): {joints_deg}")
#             print(f"  quat (w,x,y,z): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")

#         kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

# if __name__ == "__main__":
#     main()



# import argparse
# import numpy as np
# import math
# import utilities
# from kinova_gen3 import KinovaGen3

# TEST_ACTIONS = [
#     ("all joints +1",  [1.0,  1.0,  1.0,  1.0,  1.0,  1.0]),
#     ("J0 only +1",     [1.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
#     ("J1 only +1",     [0.0,  1.0,  0.0,  0.0,  0.0,  0.0]),
#     ("all joints -1",  [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
# ]

# def main():
#     parser = argparse.ArgumentParser()
#     args = utilities.parseConnectionArguments(parser)

#     with utilities.DeviceConnection.createTcpConnection(args) as router:
#         kinova = KinovaGen3(router)

#         print("=== REAL ROBOT EE VELOCITY ===\n")
#         for label, action in TEST_ACTIONS:
#             # Return to home before each test
#             kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

#             # Apply same delta as simulation
#             delta_deg = np.array(action) * (180.0 / 64.0)
#             kinova.SetJointAnglesIncremental(delta_deg.tolist())

#             # Read velocity
#             ee_vel = kinova.GetEndEffectorLinearVelocity()

#             print(f"{label}:")
#             print(f"  delta_deg: {delta_deg.round(4).tolist()}")
#             print(f"  ee_vel (vx,vy,vz): [{ee_vel[0]:.6f}, {ee_vel[1]:.6f}, {ee_vel[2]:.6f}]")
#             print(f"  magnitude: {np.linalg.norm(ee_vel):.6f}\n")

#         kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

# if __name__ == "__main__":
#     main()



import argparse
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import utilities
from kinova_gen3 import KinovaGen3

ACTION = [0.5, -0.3, 0.8, -0.6, 0.4, -0.2]
GOAL_POS = np.array([0.373, 0.359, 0.030])

def convert_euler_to_quaternion(theta_x, theta_y, theta_z):
    rot = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)
    rot_corrected = rot * R.from_euler('z', -90, degrees=True)
    q = rot_corrected.as_quat()  # (x,y,z,w)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

def deg2rad(deg):
    return np.asarray(deg, dtype=np.float32) * (math.pi / 180.0)

def get_obs(kinova):
    ee_pos = np.array(kinova.GetFingerCenterPosition(), dtype=np.float32)
    relative_vector = (GOAL_POS - ee_pos).astype(np.float32)
    ee_linear_vel = kinova.GetEndEffectorLinearVelocity()
    theta_x, theta_y, theta_z = kinova.GetEndEffectorOrientation()
    ee_quat = convert_euler_to_quaternion(theta_x, theta_y, theta_z)
    joint_angles = deg2rad(np.array(kinova.GetJointAngles(), dtype=np.float32))
    return ee_pos, relative_vector, ee_linear_vel, ee_quat, joint_angles

def main():
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        kinova = KinovaGen3(router)

        kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

        delta_deg = np.array(ACTION) * (180.0 / 5.0)
        kinova.SetJointAnglesIncremental(delta_deg.tolist())

        ee_pos, rel, vel, quat, joints = get_obs(kinova)

        print("=== REAL ROBOT ===\n")
        print(f"action:           {ACTION}")
        print(f"delta_deg:        {delta_deg.round(5).tolist()}")
        print(f"ee_pos:           [{ee_pos[0]:.6f}, {ee_pos[1]:.6f}, {ee_pos[2]:.6f}]")
        print(f"relative_vector:  [{rel[0]:.6f}, {rel[1]:.6f}, {rel[2]:.6f}]")
        print(f"ee_linear_vel:    [{vel[0]:.6f}, {vel[1]:.6f}, {vel[2]:.6f}]")
        print(f"ee_quat(w,x,y,z): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        print(f"joint_angles(rad):[{', '.join(f'{j:.6f}' for j in joints)}]")

        kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    main()