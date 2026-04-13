import sys
import os
import time
import collections
import threading
import numpy as np
# Compatibility shim
if not hasattr(collections, "MutableMapping"):
    from collections.abc import MutableMapping, MutableSequence
    collections.MutableMapping = MutableMapping
    collections.MutableSequence = MutableSequence


from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2


class KinovaLowLevelExample:

    def __init__(self, router, router_real_time, proportional_gain=2.0):

        self.proportional_gain = proportional_gain

        self.router = router
        self.router_real_time = router_real_time

        # TCP client
        self.base = BaseClient(self.router)

        # UDP cyclic client
        self.base_cyclic = BaseCyclicClient(self.router_real_time)

        # Create command object
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0

        # Add motor command
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()

        # Get first feedback
        base_feedback = self.base_cyclic.RefreshFeedback()

        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[0].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

        # Initialize actuators
        for actuator in base_feedback.actuators:

            actuator_cmd = self.base_command.actuators.add()
            actuator_cmd.position = actuator.position
            actuator_cmd.velocity = 0.0
            actuator_cmd.torque_joint = 0.0
            actuator_cmd.command_id = 0
            actuator_cmd.flags = 1

            print("Position =", actuator.position)

        # Save current servoing mode
        self.previous_servoing_mode = self.base.GetServoingMode()

        # Set LOW LEVEL SERVOING
        servoing_mode_info = Base_pb2.ServoingModeInformation()
        servoing_mode_info.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode_info)

        self._lock = threading.Lock()
        self._ee_linear_vel = None

        self.joint_limits = np.array([
            [-154.1,  154.1],
            [-150.1,  150.1],
            [-150.1,  150.1],
            [-148.98, 148.98],
            [-144.97, 145.0],
            [-148.98, 148.98],
        ], dtype=np.float64)

    def Cleanup(self):

        self.base.SetServoingMode(self.previous_servoing_mode)

    
   

    def GotoJoints(self, joint_deltas,gripper_delta, velocity=100.0, threshold=0.5, timeout=10.0):
        
        target_gripper_position = self.motorcmd.position + gripper_delta*100.0
        if target_gripper_position > 100.0:
            target_gripper_position = 100.0
        if target_gripper_position < 0.0:
            target_gripper_position = 0.0

        base_feedback = self.base_cyclic.RefreshFeedback()

        current_position = base_feedback.interconnect.gripper_feedback.motor[0].position
        position_error = target_gripper_position - current_position

        proportional_gain = 35
        target = [(self.base_command.actuators[i].position + joint_deltas[i]) % 360.0
                for i in range(len(joint_deltas))]

        target = np.clip(target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        errors = []
        for i in range(len(target)):
            error = target[i] - self.base_command.actuators[i].position
            error = ((error + 180) % 360) - 180
            errors.append(error)

        

        # Track command positions
        cmd_positions = [self.base_command.actuators[i].position 
                        for i in range(len(joint_deltas))]

        velocity_samples = []
        t_start = time.time()

        while (time.time() - t_start) < 0.1:
            with self._lock:
                base_feedback = self.base_cyclic.Refresh(self.base_command)

            velocity_samples.append([
                base_feedback.base.tool_twist_linear_x,
                base_feedback.base.tool_twist_linear_y,
                base_feedback.base.tool_twist_linear_z,
            ])

            for i in range(len(joint_deltas)):
                # Fixed-step control — direction from pre-computed error
                step = 0.001 * min(proportional_gain * abs(errors[i]), velocity) * (1 if errors[i] > 0 else -1)
                cmd_positions[i] = (cmd_positions[i] + step) % 360.0
                self.base_command.actuators[i].position = cmd_positions[i]


                if abs(position_error) < 1.5:

                    self.motorcmd.position = 0
                else:

                    velocity_gripper = self.proportional_gain * abs(position_error)
                    velocity_gripper = min(velocity_gripper, 100)

                    self.motorcmd.velocity = velocity_gripper
                    self.motorcmd.position = target_gripper_position

                    self.base_cyclic.Refresh(self.base_command)

            time.sleep(0.01)

        self.ee_vel = velocity_samples[len(velocity_samples) // 2] if velocity_samples else [0, 0, 0]
        print("mid: ", self.ee_vel)
        return True
        
    def GetGripperState(self):
        base_feedback = self.base_cyclic.RefreshFeedback()

        current_position = base_feedback.interconnect.gripper_feedback.motor[0].position
        return np.clip(current_position/100.0, 0.0, 1.0)

    def GetJointAngles(self):
        angles_deg = []
        for i in range(6):
            a = self.base_command.actuators[i].position
            a_wrapped = ((a + 180.0) % 360.0) - 180.0
            angles_deg.append(a_wrapped)
        return angles_deg



    def GetEndEffectorPosition(self):
        feedback = self.base_cyclic.RefreshFeedback()
        return (
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z
        )

    def GetEndEffectorOrientation(self):
        feedback = self.base_cyclic.RefreshFeedback()
        return (
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z
        )

    def GetEndEffectorLinearVelocity(self):
        if self._ee_linear_vel is None:
            return np.zeros(3, dtype=np.float32)
        return self._ee_linear_vel


    def GetFingerCenterPosition(self):
        """
        Returns EE position offset 19cm along the EE z-axis,
        matching simulation's (RIGHT_FINGER_PROX + LEFT_FINGER_PROX) / 2
        """
        feedback = self.base_cyclic.RefreshFeedback()

        tool_pos = np.array([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z
        ], dtype=np.float64)

        # Get EE orientation
        from scipy.spatial.transform import Rotation as R
        theta_x = feedback.base.tool_pose_theta_x
        theta_y = feedback.base.tool_pose_theta_y
        theta_z = feedback.base.tool_pose_theta_z
        rot = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)

        # Offset along local +z of tool frame (meters); tune to match sim gripper midpoint
        offset_local = np.array([0.0, 0.0, -0.057])
        offset_world = rot.apply(offset_local)

        return (tool_pos + offset_world).astype(np.float32)



                

        

    