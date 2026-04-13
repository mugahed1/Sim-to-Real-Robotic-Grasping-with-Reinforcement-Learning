import sys
import os
import time
import threading

import argparse
import utilities

import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

TIMEOUT_DURATION = 0.5


class KinovaGen3:
    def __init__(self, router):

        self.router = router

        # Create base client using TCP router
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)


        # gripper : 
        self.gripper_command = Base_pb2.GripperCommand()
        self.finger = self.gripper_command.gripper.finger.add()
        self.gripper_command.mode = Base_pb2.GRIPPER_POSITION
        self.finger.finger_identifier = 1
        self.gripper_request = Base_pb2.GripperRequest()
        self.gripper_request.mode = Base_pb2.GRIPPER_POSITION

        #joints : 
        self.actuator_count = self.base.GetActuatorCount().count

        self.joint_limits = np.array([
            [-154.1,  154.1],  # J0
            [-150.1,  150.1],  # J1
            [-150.1,  150.1],  # J2
            [-148.98,  148.98],  # J3
            [-144.97,  145.0],  # J4
            [-148.98,  148.98],  # J5
        ], dtype=np.float64)

        # EE velocity/motion timing state
        self._prev_ee_pos = None
        self._new_ee_pos = None
        self._ee_time = None
        self.prev_joint_angles=np.zeros(6, dtype=np.float32)
        self.prev_ee_pos=np.zeros(3, dtype=np.float32)
        

    
    def check_for_end_or_abort(self,e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            # print("EVENT : " + \
            #     Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def SetGripperCommands(self, position: float = 0.0):
        # Clamp [0, 1] 0 is open, 1 is closed
        gripper_state = self.GetGripperState() + position
        gripper_state = max(0.0, min(1.0, float(gripper_state)))

        self.finger.value = gripper_state
        # print("Going to position {:0.2f}...".format(self.finger.value))
        self.base.SendGripperCommand(self.gripper_command)
        time.sleep(TIMEOUT_DURATION)


    def GetGripperState(self):
        meas = self.base.GetMeasuredGripperMovement(self.gripper_request)
        return meas.finger[0].value if len(meas.finger) else None

    def GetJointAngles(self):
        """
        Return joint angles in degrees, wrapped to [-180, 180).
        """
        joint_angles = self.base.GetMeasuredJointAngles().joint_angles

        angles_deg = []
        for angle in joint_angles:
            a = angle.value
            a_wrapped = ((a + 180.0) % 360.0) - 180.0
            angles_deg.append(a_wrapped)

        return angles_deg

    def SetJointAnglesIncremental(self, delta_angles: list[float]):
        """
        Move arm by joint angle increments (degrees).
        """

        if self.actuator_count != len(delta_angles):
            raise ValueError(
                f"Joint count mismatch: robot has {self.actuator_count} actuators, "
                f"but {len(delta_angles)} increments were given."
            )

        # Read current joint angles (degrees)
        current_angles = self.GetJointAngles()

        # Compute absolute target
        target_angles = (
            np.array(current_angles, dtype=np.float64)
            + np.array(delta_angles, dtype=np.float64)
        )

        return self.SetJointPositions(target_angles)

    
    

    def SetJointPositions(self, target_angles: list[float]):
        """
        
        """
        self._prev_ee_pos = np.array(self.GetFingerCenterPosition(), dtype=np.float64)

        if self.actuator_count != len(target_angles):
            raise ValueError(
                f"Joint count mismatch: robot has {self.actuator_count} actuators, "
                f"but {len(target_angles)} positions were given."
            )

        # Read current joint angles (degrees)
        current_angles = self.GetJointAngles()
        if len(current_angles) < self.actuator_count:
            raise RuntimeError(
                f"Expected {self.actuator_count} joint angles from robot, "
                f"got {len(current_angles)}."
            )

        
        action = Base_pb2.Action()
        action.name = "Set joint positions"
        action.application_data = ""

        target_angles = np.clip(
            target_angles,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        ).tolist()

        for joint_id in range(self.actuator_count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = target_angles[joint_id]  # degrees

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        # print("Executing action")
        self.base.ExecuteAction(action)

        # print("Waiting for movement to finish ...")
        
        finished = e.wait(TIMEOUT_DURATION)
        
        self.base.Unsubscribe(notification_handle)
        
        
        
        self._new_ee_pos = np.array(self.GetFingerCenterPosition(), dtype=np.float64)

        
        return finished


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
        """
        Get EE linear velocity directly from BaseCyclic feedback.
        Measured Cartesian linear velocity of the gripper in m/s.
        """
        feedback = self.base_cyclic.RefreshFeedback()
        return np.array([
            feedback.base.tool_twist_linear_x,
            feedback.base.tool_twist_linear_y,
            feedback.base.tool_twist_linear_z
        ], dtype=np.float32)

    def move_to_home_position(self):
    
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        # Move arm to ready position
        # print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle
                break

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)

        # Leave time to action to complete
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if not finished:
            print("Timeout on action notification wait")
        return finished

    def ReachCartesianPosition(self, x: float, y: float, z: float, theta_x: float = 0.0, theta_y: float = 0.0, theta_z: float = 0.0):
        """
        Move the robot TCP to an absolute Cartesian position (x,y,z) in meters,
        keeping the current orientation (theta_x/y/z) from feedback.
        """

        print("Starting Cartesian action movement ...")
        action = Base_pb2.Action()
        action.name = "Reach Cartesian position"
        action.application_data = ""

        # Keep current orientation from feedback
        feedback = self.base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = float(x)  # meters
        cartesian_pose.y = float(y)  # meters
        cartesian_pose.z = float(z)  # meters

        # Keep current orientation (degrees)
        cartesian_pose.theta_x = theta_x
        cartesian_pose.theta_y = theta_y
        cartesian_pose.theta_z = theta_z

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")

        return finished

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



        

    