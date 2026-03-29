import argparse
import numpy as np
import math
import time
import utilities
from kinova_gen3 import KinovaGen3

ACTION    = [0.5, -0.3, 0.8, -0.6, 0.4, -0.2]



def deg2rad(deg):
    return float(deg) * (math.pi / 180.0)

def main():
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        kinova = KinovaGen3(router)

        
        import time; time.sleep(0.5)

        delta_deg = np.array(ACTION) * (180.0 / 64.0)

        print("=== REAL ROBOT — 10 steps same action ===\n")
        print(f"{'Step':>4} | {'J0(rad)':>10} | {'disp_x':>10} | {'disp_y':>10} | {'disp_z':>10} | {'vel_x':>10} | {'vel_y':>10} | {'vel_z':>10}")
        print("-" * 90)

        for step in range(10):
            prev_ee_pos = np.array(kinova.GetFingerCenterPosition(), dtype=np.float64)

            kinova.SetJointAnglesIncremental(delta_deg.tolist())

            new_ee_pos  = np.array(kinova.GetFingerCenterPosition(), dtype=np.float64)
            displacement = new_ee_pos - prev_ee_pos
            vel = kinova.GetEndEffectorLinearVelocity()

            J0 = deg2rad(kinova.GetJointAngles()[0])

            print(f"{step+1:>4} | {J0:>10.6f} | {displacement[0]:>10.6f} | {displacement[1]:>10.6f} | {displacement[2]:>10.6f} | {vel[0]:>10.4f} | {vel[1]:>10.4f} | {vel[2]:>10.4f}")

        kinova.SetJointPositions([0, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    main()