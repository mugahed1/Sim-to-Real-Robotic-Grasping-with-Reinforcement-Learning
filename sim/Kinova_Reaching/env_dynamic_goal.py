import mujoco
import mujoco.viewer
import numpy as np
import math
from gymnasium.spaces import Box
from catalyst_rl.rl.core import EnvironmentSpec
from catalyst_rl.rl.utils import extend_space
import logging
import os 
from normalization import RunningMeanStd

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)



class MujocoKinovaGraspEnv(EnvironmentSpec):

    def __init__(self, visualize=True, mode="train", **params):
        super().__init__(visualize=visualize, mode=mode)

        self.mode = mode

        self.obs_rms = RunningMeanStd(
            shape=(15,),
            epsilon=1e-8,
            batch_size=200 
        )

        if self.mode =="test":
            self.obs_rms.load(exp='exp7')
            

        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)

        self.visualize = visualize
        self.viewer = None
        if visualize:
            print("Launching MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        
        self.joint_names = [
            "J0","J1","J2","J3","J4","J5"
        ]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]

        self.gripper_joint_names = [
            "RIGHT_BOTTOM", "RIGHT_TIP", 
            "LEFT_BOTTOM", "LEFT_TIP"
        ]
        self.gripper_joint_ids = [self.model.joint(name).id for name in self.gripper_joint_names]

        self.joint_limits = []
        for joint_name in self.joint_names:
            joint_obj = self.model.joint(joint_name)
            self.joint_limits.append([joint_obj.range[0], joint_obj.range[1]])
        
        self.joint_limits = np.array(self.joint_limits)

        # --- BODIES: EE tip + goal ---
        self.goal_body = self.model.body("target").id
        self.ee_body = self.model.body("END_EFFECTOR").id  
        
        
        target_body = self.model.body("target")
        self.target_joint_id = target_body.jntadr[0]

        # Geom IDs for contact detection
        self.right_prox = self.model.geom("RIGHT_FINGER_PROX_GEOM").id
        self.left_prox  = self.model.geom("LEFT_FINGER_PROX_GEOM").id
        self.right_tip  = self.model.geom("RIGHT_FINGER_DIST_GEOM").id
        self.left_tip   = self.model.geom("LEFT_FINGER_DIST_GEOM").id
        self.obj_geom   = self.model.geom("target").id

        # --- ENV SPACES ---
        self.max_step_count = 100
        self.step_counter = 0
        self.target_pose = None
        self.initial_distance = None
        self._history_len = 1
        self.fixed_goal = None  # For evaluation: fixed goal position

        
        self._observation_space = Box(-np.inf, np.inf, (15,))
        # Action: 6 joints only (gripper kept always open)
        self._action_space = Box(-1, 1, (6,))
        try:
            self._state_space = extend_space(self._observation_space, self._history_len)
        except Exception as e:
            
            if self._history_len == 1:
                self._state_space = self._observation_space
            else:
                obs_shape = self._observation_space.shape
                new_shape = (obs_shape[0] * self._history_len,)
                self._state_space = Box(-np.inf, np.inf, new_shape, dtype=self._observation_space.dtype)
        
        
        self.x_range = [0.3, 0.4]    
        self.y_range = [0.3, 0.4]   
        self.z_range = 0.03    

        self.gripper_open = True  # Start with open gripper
        self.object_grasped = False
        self.grasp_threshold = 0.10  # Distance for successful grasp
        self.phase=0 
        self.success_cnt = 0
        self.collision = False

        # Gripper actuator limits 
        self.gripper_open_targets = np.array([0.96, 0.21, -0.96, -0.21])
        self.gripper_closed_targets = np.array([0.45, -1.03, -0.45, 1.03])

        self.init_q_arm = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.init_gripper = -1.0  
        self._ref_joint_qpos_adr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names]

        self.required_stable_steps=5
        self.grasp_stable_steps = 0

        self._total_steps = 0
        self._norm_freeze_after_steps = 100000

        

    
    @property
    def action_space(self): 
        return self._action_space

    @property
    def observation_space(self): 
        return self._observation_space

    @property
    def state_space(self): 
        return self._state_space

    @property
    def history_len(self):
        return self._history_len

    
    def reset(self):
        logging.info("Episode reset...")

        self.step_counter = 0
        self.gripper_open = True
        self.object_grasped = False
        self.phase=0 
        self.collision = False
        self.grasp_stable_steps = 0

        # Full MuJoCo reset 
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Initialize ctrl to current qpos so motors don’t “snap”

        for i, qpos_adr in enumerate(self._ref_joint_qpos_adr):
            self.data.qpos[qpos_adr] = self.init_q_arm[i]
        self.data.qvel[:] = 0.0
        for i in range(6):
            self.data.ctrl[i] = self.init_q_arm[i]
        self._set_gripper(gripper_action=self.init_gripper)
        mujoco.mj_forward(self.model, self.data)

        # Warm up simulation a bit 
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # If visualization is enabled
        if self.visualize and self.viewer:
            self.viewer.sync()

        # Sample new goal 
        self.setup_goal()

        


        return self.get_observation()


    
    def step(self, action):
        done = False
        terminated=False
        info = {}

        self._total_steps += 1
        self.apply_controls(action)
        self.step_counter += 1

        

        
        # logger.info(f'****** distance: {dist} *****')
        
        
        reward = self.compute_reward(action) 
        
        if self.success_check():
            done = True
            reward += 10  
            self.success_cnt +=1
            terminated=True
            logger.info('--------Reset: Success--------')
            return self.get_observation(), reward, done, info,terminated
            
        if self.collision_check():
            done = True
            self.collision = True
            reward -= 100
            terminated=True
            logger.info('--------Reset: Object Pushed-------')
            return self.get_observation(), reward, done, info,terminated

        if self.step_counter >= self.max_step_count:
            done = True
            logger.info('--------Reset: Timeout--------')
            return self.get_observation(), reward, done, info,terminated


        if self.visualize and self.viewer:
            self.viewer.sync()

        return self.get_observation(), reward, done, info, terminated
    
    def compute_reward(self,action):
        """
        Reward based on distance between end-effector and target.
        Action controls only the 6 arm joints; gripper is kept open.
        """
        action_magnitude = np.linalg.norm(action[:6])

        dist = self.distance_to_goal()
        # Get end-effector velocity
        cvel = self.data.cvel[self.ee_body].copy()
        ee_linear_vel = cvel[3:]
        vel_magnitude = np.linalg.norm(ee_linear_vel)
        
        
        reward = -dist
       
            
       
        
        
        return reward
    
    def end_effector_pos(self):
        return self.data.site("END_EFFECTOR_SITE").xpos.copy()

    def distance_to_goal(self):
        """Distance between end effector and goal"""
        ee_pos = self.end_effector_pos()
        goal_pos = self.data.xpos[self.goal_body]
        return np.linalg.norm(ee_pos - goal_pos)


    def set_fixed_goal(self, goal_pos):
        """Set a fixed goal position for evaluation (will be reused in reset)."""
        self.fixed_goal = np.array(goal_pos, dtype=np.float64)
    
    def setup_goal(self):
        """Setup goal position by teleporting the free-joint cylinder."""
        if self.fixed_goal is not None:
            # Use fixed goal for evaluation
            self.target_pose = self.fixed_goal.copy()
        else:
            # Random goal for training
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
            z = self.z_range
            self.target_pose = np.array([x, y, z], dtype=np.float64)

        
        j = self.target_joint_id
        if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Cylinder joint is not a free joint")

        qpos_adr = self.model.jnt_qposadr[j]
        qvel_adr = self.model.jnt_dofadr[j]

        # qpos for free joint: [x y z qw qx qy qz]
        self.data.qpos[qpos_adr:qpos_adr+3] = self.target_pose
        self.data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # qvel for free joint (6 dof): [vx vy vz wx wy wz]
        self.data.qvel[qvel_adr:qvel_adr+6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.initial_distance = self.distance_to_goal()
        # logger.info(f"Setting goal at {self.target_pose.tolist()}, distance: {self.initial_distance}")
        
        
        

    def get_observation(self):
        # EE position and goal position
        tip_pos = self.end_effector_pos()
        goal_pos = self.data.xpos[self.goal_body].copy()
        ee_pos = tip_pos.astype(np.float32)       # 3: ee x, y, z
        goal_pos_f32 = goal_pos.astype(np.float32)  # 3: goal x, y, z

        # 3 for dx dy dz: end-effector distance to the goal (goal - ee)
        relative_vector = (goal_pos - tip_pos).astype(np.float32)  # dx, dy, dz

        # 6 for joint angles
        joint_angles = np.array([
            self.data.qpos[self.joint_ids[0]],
            self.data.qpos[self.joint_ids[1]],
            self.data.qpos[self.joint_ids[2]],
            self.data.qpos[self.joint_ids[3]],
            self.data.qpos[self.joint_ids[4]],
            self.data.qpos[self.joint_ids[5]],
        ], dtype=np.float32)

        # Observation: 15 dims (ee_pos + goal_pos + relative + joints)
        obs_raw = np.concatenate([
            ee_pos,            # 3
            goal_pos_f32,      # 3
            relative_vector,   # 3
            joint_angles,      # 6
        ]).astype(np.float32)

        
       
        if self.mode == "train" and self._total_steps < self._norm_freeze_after_steps:
            self.obs_rms.update(obs_raw)

        return self.obs_rms.normalize(obs_raw).astype(np.float32)

        
      

    def collision_check(self):
        
        target = self.data.xpos[self.goal_body]

     
        if target[0] > 0.7 or target[1]>0.7 : 
            print("collision detected at step", self.step_counter)
            return True
        
        return False

    def success_check(self):
        dist = self.distance_to_goal()
        if dist < self.grasp_threshold :
            print("success detected at step", self.step_counter)
            return True
        else:
            return False
    
    def apply_controls(self, action):
        
       
        action = action[:6] * (math.pi / 64.0)
        
        
        current_positions = self.data.ctrl.copy()
        
        
        joints_to_control = [0, 1, 2,3,4,5]  

        for i, joint_idx in enumerate(joints_to_control):
            
            current_target = current_positions[joint_idx]
            
            
            new_target = current_target + action[i]
            
            # Clip to actuator control range
            actuator = self.model.actuator(joint_idx)
            new_target = np.clip(
                new_target,
                actuator.ctrlrange[0],
                actuator.ctrlrange[1]
            )
            
            # Set new target position
            self.data.ctrl[joint_idx] = new_target
        
        # Keep gripper always open
        self._set_gripper(self.init_gripper)
        
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

    def gripper_state(self):
        """
        Gripper state in [0, 1]
        0 = fully open
        1 = fully closed
        Uses ACTUAL joint positions (qpos)
        """

        normalized = []

        for i, joint_name in enumerate(self.gripper_joint_names):
            joint_id = self.model.joint(joint_name).id
            current = self.data.qpos[joint_id]

            open_i   = self.gripper_open_targets[i]
            closed_i = self.gripper_closed_targets[i]

            if np.isclose(open_i, closed_i):
                continue

            # Normalize to [0, 1]
            alpha = (current - open_i) / (closed_i - open_i)
            alpha = np.clip(alpha, 0.0, 1.0)

           
            normalized.append(alpha)

        if not normalized:
            return 0.0

        # Conservative grasp definition
        gripper_state = np.min(normalized)

        return float(np.clip(gripper_state, 0.0, 1.0))



    def _set_gripper(self, gripper_action):
        """
        gripper_action ∈ [-1, 1]
        -1 = fully open
        0 = half closed
        +1 = fully closed
        
        All 4 actuators [RIGHT_BOTTOM, RIGHT_TIP, LEFT_BOTTOM, LEFT_TIP] change.
        """

        # Normalize to [0, 1]
        alpha = (gripper_action + 1.0) / 2.0

        # Initialize targets with open values
        targets = self.gripper_open_targets.copy()
        
        # Interpolate all 4 actuators between open and closed
        for i in range(4):
            targets[i] = (1 - alpha) * self.gripper_open_targets[i] + alpha * self.gripper_closed_targets[i]

        gripper_actuators = ["RIGHT_BOTTOM", "RIGHT_TIP",
                            "LEFT_BOTTOM", "LEFT_TIP"]

        for name, target in zip(gripper_actuators, targets):
            actuator_id = self.model.actuator(name).id
            self.data.ctrl[actuator_id] = target



    def closeSim(self):
        if self.viewer:
            self.viewer.close()
        print("MuJoCo simulation closed.")