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
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)



class MujocoKinovaGraspEnv(EnvironmentSpec):

    def __init__(self, visualize=True, mode="train",exp="exp1", **params):
        super().__init__(visualize=visualize, mode=mode)

        self.mode = mode

        self.obs_rms = RunningMeanStd(
            shape=(29,),
            epsilon=1e-8,
            batch_size=200 
        )

        if self.mode =="test":
            self.obs_rms.load(exp=exp)
            

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
        self.base_body = self.model.body("base_link").id
        
        
        target_body = self.model.body("target")
        self.target_joint_id = target_body.jntadr[0]

        # Geom IDs for contact detection
        self.right_prox = self.model.geom("RIGHT_FINGER_PROX_GEOM").id
        self.left_prox  = self.model.geom("LEFT_FINGER_PROX_GEOM").id
        self.right_tip  = self.model.geom("RIGHT_FINGER_DIST_GEOM").id
        self.left_tip   = self.model.geom("LEFT_FINGER_DIST_GEOM").id
        self.obj_geom   = self.model.geom("target").id
        self.ee_site = self.model.site("END_EFFECTOR_SITE").id
        # Floor geom for contact penalty
        self.floor_geom = self.model.geom("floor").id

        # --- ENV SPACES ---
        self.max_step_count = 150
        self.step_counter = 0
        self.target_pose = None
        self.initial_distance = None
        self._history_len = 1

        
        self._observation_space = Box(-np.inf, np.inf, (29,))
        self._action_space = Box(-1, 1, (7,)) # 6 joints + gripper state action    
        try:
            self._state_space = extend_space(self._observation_space, self._history_len)
        except Exception as e:
            # extend_space from catalyst_rl may not support gymnasium spaces
            # Since _state_space is only used internally and not by SB3, we can set it to observation_space
            # or create a simple compatible version
            if self._history_len == 1:
                self._state_space = self._observation_space
            else:
                # For history_len > 1, create a Box with expanded shape
                obs_shape = self._observation_space.shape
                new_shape = (obs_shape[0] * self._history_len,)
                self._state_space = Box(-np.inf, np.inf, new_shape, dtype=self._observation_space.dtype)
        
        
        self.x_range = [0.3, 0.4]    
        self.y_range = [0.3, 0.4]   
        self.z_range = 0.03    

        
        self.object_grasped = False
        self.grasp_threshold = 0.1  # Distance for successful grasp
        self.phase=0 
        self.success_cnt = 0
        self.collision = False

        # Gripper actuator limits
        # Only RIGHT_BOTTOM and LEFT_BOTTOM are used.
        # RIGHT_TIP and LEFT_TIP are always held at 0.
        # open  = (0.96, 0, -0.96, 0)
        # close = (0,    0,  0,    0)
        self.gripper_open_targets = np.array([0.96, 0.0, -0.96, 0.0], dtype=np.float64)
        self.gripper_closed_targets = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initial pose: J0..J5 (rad), gripper state (-1 = open)
        self.init_q_arm = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.init_gripper = -1.0  # -1 = fully open
        self._ref_joint_qpos_adr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names]

        self.required_stable_steps=1
        self.grasp_stable_steps = 0

        # Total env steps (across episodes) for freezing obs normalization
        self._total_steps = 0
        self._norm_freeze_after_steps = 100000

        # Previous-step trackers for observations
        self.prev_joint_angles = np.zeros(6, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)

        
        
    
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

        # Initialize gripper actuators to OPEN:
        # RIGHT_BOTTOM in [0, 0.96] and LEFT_BOTTOM in [-0.96, 0]
        self.data.ctrl[self.model.actuator("RIGHT_BOTTOM").id] = 0.96
        self.data.ctrl[self.model.actuator("LEFT_BOTTOM").id] = -0.96
        # Tips always held at 0
        self.data.ctrl[self.model.actuator("RIGHT_TIP").id] = 0.0
        self.data.ctrl[self.model.actuator("LEFT_TIP").id] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Warm up simulation a bit 
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)

        
        # If visualization is enabled
        if self.visualize and self.viewer:
            self.viewer.sync()

        # Sample new goal 
        self.setup_goal()

        # Initialize previous trackers based on current state
        self.prev_joint_angles = np.array([
            self.data.qpos[self.joint_ids[0]],
            self.data.qpos[self.joint_ids[1]],
            self.data.qpos[self.joint_ids[2]],
            self.data.qpos[self.joint_ids[3]],
            self.data.qpos[self.joint_ids[4]],
            self.data.qpos[self.joint_ids[5]],
        ], dtype=np.float32)
        self.prev_ee_pos = self.end_effector_pos().astype(np.float32)

        return self.get_observation()


    
    def step(self, action):
        done = False
        terminated=False
        info = {}

        self._total_steps += 1
        self.apply_controls(action)
        self.step_counter += 1

        
        reward = self.compute_reward(action) 
        
        if self.success_check():
            done = True
            reward += 50  
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
        Two-phase reward:
        Phase 0: Reach the object with open gripper
        Phase 1: Grasp 
        """
        gripper_action = action[6]

        dist = self.distance_to_goal()

        
        
        # Get end-effector velocity
        cvel = self.data.cvel[self.ee_body].copy()
        ee_linear_vel = cvel[3:]
        vel_magnitude = np.linalg.norm(ee_linear_vel)
        
        
        vel_reward = vel_magnitude**2
        
        reward = -dist - 0.1*vel_reward
       
        # ============================================================
        # PHASE 0: REACHING
        # ============================================================
        if self.phase == 0:
            # Base reward from reaching
            # reward -= 0.2*self.gripper_state()
            
            
            
            # Gripper opening reward (keep gripper open while approaching)
            if gripper_action<0:
                reward += 0.1*abs(gripper_action)
            
            # Transition to grasping phase when close enough
            if dist < self.grasp_threshold:  
                reward += 1.0  
                self.phase = 1
                print(f"!!! PHASE 1 at step {self.step_counter}, dist={dist:.3f}")
        
        # ============================================================
        # PHASE 1: GRASPING 
        # ============================================================
        elif self.phase == 1:
            # reward+=self.gripper_state()
            # if gripper_action>0:
            #     reward += 0.5*gripper_action
            

            touch_right_finger = False
            touch_left_finger = False

            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # Right finger contact
                if (contact.geom1 == self.right_prox and contact.geom2 == self.obj_geom) or \
                (contact.geom1 == self.obj_geom and contact.geom2 == self.right_prox) or (contact.geom1 == self.right_tip and contact.geom2 == self.obj_geom) or \
                (contact.geom1 == self.obj_geom and contact.geom2 == self.right_tip):
                    touch_right_finger = True
                
                # Left finger contact  
                if (contact.geom1 == self.left_prox and contact.geom2 == self.obj_geom) or \
                (contact.geom1 == self.obj_geom and contact.geom2 == self.left_prox) or (contact.geom1 == self.left_tip and contact.geom2 == self.obj_geom) or \
                (contact.geom1 == self.obj_geom and contact.geom2 == self.left_tip):
                    touch_left_finger = True
        
                
            
            if touch_left_finger and touch_right_finger:
                # Both fingers touching
                self.grasp_stable_steps += 1
                
                print(f"Contact: {self.grasp_stable_steps}/{self.required_stable_steps}")
                
                # Progressive reward for maintaining contact
                contact_reward = 5.0 + (self.grasp_stable_steps * 2.0)
                reward += contact_reward
                
                # Check if achieved stable grasp (5 consecutive steps)
                if self.grasp_stable_steps >= self.required_stable_steps:
                    
                    self.object_grasped = True
                    print(f">>> STABLE GRASP achieved at step {self.step_counter}! <<<")
                   
    
            else:
                # Contact broken - reset counter
                if self.grasp_stable_steps > 0:
                    print(f"Contact lost at {self.grasp_stable_steps} steps")
                self.grasp_stable_steps = 0
                
                
        # Penalty if any robot geom touches the floor
        # if self._touching_floor():
        #     reward -= 10.0

        return reward
    
    def end_effector_pos(self):
        right_pos = self.data.body("RIGHT_FINGER_PROX").xpos.copy()
        left_pos = self.data.body("LEFT_FINGER_PROX").xpos.copy()
        
        # Calculate center point between fingers in bottom 
        center_pos = (right_pos + left_pos) / 2

        return center_pos

    def distance_to_goal(self):
        """Distance between end effector and goal"""
        ee_pos = self.end_effector_pos()
        goal_pos = self.data.xpos[self.goal_body]
        return np.linalg.norm(ee_pos - goal_pos)


    def setup_goal(self):
        """Setup goal position by teleporting the free-joint cylinder."""
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])
        z = self.z_range

        self.target_pose = np.array([x, y, z], dtype=np.float64)

        # Use the cylinder joint ID stored during initialization
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

        # Replace velocity with previous joint angles (6) and previous EE position (3)
        prev_joint_angles = self.prev_joint_angles.astype(np.float32)
        prev_ee_pos = self.prev_ee_pos.astype(np.float32)
        
        # 4 for gripper (end-effector) orientation as quaternion (w, x, y, z)
        ee_quat = self.data.xquat[self.ee_body].astype(np.float32)
        
        # 6 for joint angles
        joint_angles = np.array([
            self.data.qpos[self.joint_ids[0]],
            self.data.qpos[self.joint_ids[1]],
            self.data.qpos[self.joint_ids[2]],
            self.data.qpos[self.joint_ids[3]],
            self.data.qpos[self.joint_ids[4]],
            self.data.qpos[self.joint_ids[5]],
        ], dtype=np.float32)

        # 1 for gripper state: open or closed
        gripper_state = np.array([self.gripper_state()], dtype=np.float32)

        # Observation: 29 dims (ee_pos + goal_pos + relative + prev_joints + prev_ee_pos + orientation + joints + gripper)
        obs_raw = np.concatenate([
            ee_pos,            # 3
            goal_pos_f32,      # 3
            relative_vector,   # 3
            prev_joint_angles, # 6
            prev_ee_pos,       # 3
            ee_quat,           # 4
            joint_angles,      # 6
            gripper_state,     # 1
        ]).astype(np.float32)

       #  Online normalization (freeze after 100k steps to avoid distribution drift)
        if self.mode == "train" and self._total_steps < self._norm_freeze_after_steps:
            self.obs_rms.update(obs_raw)
        
        # Update previous trackers for next observation
        self.prev_joint_angles = joint_angles.copy()
        self.prev_ee_pos = ee_pos.copy()
        
        return self.obs_rms.normalize(obs_raw).astype(np.float32)

        
      

    def collision_check(self):
        
        target = self.data.xpos[self.goal_body]

     #  if object is pushed out the reaching zone, we stop the episode 
        if target[0] > 0.7 or target[1]>0.7 : 
            print("collision detected at step", self.step_counter)
            return True
        
        return False

    def _touching_floor(self):
        """
        Returns True if any contact involves the floor geom.
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.floor_geom or c.geom2 == self.floor_geom:
                # Identify the "other" geom (not the floor) and its body
                other_geom = c.geom2 if c.geom1 == self.floor_geom else c.geom1
                other_body = self.model.geom_bodyid[other_geom]
                # Ignore floor contact with object or robot base
                if other_body in (self.goal_body, self.base_body):
                    continue
                print(f"Touching floor at step {self.step_counter}")
                return True
                
        return False

    def success_check(self):
        
        if self.object_grasped :
            return True
        
        return False   
    
    def apply_controls(self, action):
        
        gripper_action = action[6]*0.05
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
        
        left_bottom = current_positions[self.model.actuator("LEFT_BOTTOM").id]
        right_bottom = current_positions[self.model.actuator("RIGHT_BOTTOM").id]

        left_bottom_new = np.clip(left_bottom + gripper_action, -0.96, 0.0)
        right_bottom_new = np.clip(right_bottom - gripper_action, 0.0, 0.96)

        self.data.ctrl[self.model.actuator("LEFT_BOTTOM").id] = left_bottom_new
        self.data.ctrl[self.model.actuator("RIGHT_BOTTOM").id] = right_bottom_new
        
        
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

    def gripper_state(self):
        current_positions = self.data.ctrl.copy()   
        right_bottom = current_positions[self.model.actuator("RIGHT_BOTTOM").id]
        
        # RIGHT_BOTTOM: 0.96=open, 0.0=closed
        # Normalize to [0=open, 1=closed]
        state = 1.0 - (right_bottom / 0.96)
        
        return float(np.clip(state, 0.0, 1.0))

    def closeSim(self):
        if self.viewer:
            self.viewer.close()
        print("MuJoCo simulation closed.")