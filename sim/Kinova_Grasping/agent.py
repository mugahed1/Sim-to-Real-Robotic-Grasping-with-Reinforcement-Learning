import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, PotentialNetwork
from core import PriorityMemory

import os
import numpy as np


class Agent:
 
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=256, fc2=256, batch_size=256, alpha_sac=0.2):  
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        
        self._target_entropy = -float(n_actions)  

        initial_log_alpha = np.log(alpha_sac) if alpha_sac > 0 else 0.0
        self.log_alpha = tf.Variable(initial_log_alpha, dtype=tf.float32, trainable=True)
       
        self.alpha_optimizer = Adam(learning_rate=alpha)

     
        self.actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, 
                                  n_actions=n_actions, max_action=self.max_action, name='actor')
        self.critic_1 = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='critic_1')
        self.critic_2 = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='critic_2')
        self.target_critic_1 = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='target_critic_2')

        # Compile networks
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        
        self._build_networks(input_dims=input_dims, n_actions=n_actions)
        self.update_network_parameters(tau=1)

    def _build_networks(self, input_dims, n_actions):
        """Force-create variables for all networks by running a single forward pass."""
        obs_dim = int(input_dims[0]) if isinstance(input_dims, (tuple, list)) else int(input_dims)
        dummy_s = tf.zeros((1, obs_dim), dtype=tf.float32)
        dummy_a = tf.zeros((1, n_actions), dtype=tf.float32)

        
        _ = self.actor(dummy_s)
        _ = self.critic_1(dummy_s, dummy_a)
        _ = self.critic_2(dummy_s, dummy_a)
        _ = self.target_critic_1(dummy_s, dummy_a)
        _ = self.target_critic_2(dummy_s, dummy_a)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Use get_weights/set_weights (lists of numpy arrays) for safe Polyak averaging.
        def _polyak_update(source_model, target_model, tau_):
            src = source_model.get_weights()
            tgt = target_model.get_weights()
            if len(src) != len(tgt):
                raise RuntimeError(
                    f"Target/source weight length mismatch: source={len(src)} target={len(tgt)}. "
                    f"Did you forget to build the models?"
                )
            new = [tau_ * s + (1.0 - tau_) * t for s, t in zip(src, tgt)]
            target_model.set_weights(new)

        # Only update target critics (standard SAC has no target actor)
        _polyak_update(self.critic_1, self.target_critic_1, tau)
        _polyak_update(self.critic_2, self.target_critic_2, tau)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, exp):
        print('... saving models ...')
        save_path = f"tmp/{exp}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.actor.save_weights(save_path + self.actor.checkpoint_file)
        self.critic_1.save_weights(save_path + self.critic_1.checkpoint_file)
        self.critic_2.save_weights(save_path + self.critic_2.checkpoint_file)
        self.target_critic_1.save_weights(save_path + self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(save_path + self.target_critic_2.checkpoint_file)
        # Save log_alpha (for automatic alpha tuning)
        np.save(save_path + 'log_alpha.npy', self.log_alpha.numpy())

    def load_models(self, exp):
        print('... loading models ...')
        save_path = f"tmp/{exp}/"
        if not os.path.exists(save_path):
            raise("Path not exists!")
        self.actor.load_weights(save_path + self.actor.checkpoint_file)
        self.critic_1.load_weights(save_path + self.critic_1.checkpoint_file)
        self.critic_2.load_weights(save_path + self.critic_2.checkpoint_file)
        self.target_critic_1.load_weights(save_path + self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(save_path + self.target_critic_2.checkpoint_file)
        # Load log_alpha if it exists (for automatic alpha tuning)
        log_alpha_path = save_path + 'log_alpha.npy'
        if os.path.exists(log_alpha_path):
            self.log_alpha.assign(np.load(log_alpha_path))

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        if evaluate:
            # Use deterministic action for evaluation
            actions = self.actor.deterministic_action(state)
        else:
            # Sample stochastic action for training
            actions, _ = self.actor.sample(state, reparameterize=True)
        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        rewards = tf.reshape(rewards, (-1, 1))  
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Convert done from boolean to float32 tensor for arithmetic (1 - done)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        done = tf.reshape(done, (-1, 1))  
       
        with tf.GradientTape() as alpha_tape:
            current_actions, current_log_probs = self.actor.sample(states, reparameterize=True)
            current_log_probs = tf.squeeze(current_log_probs, 1)
            alpha_loss = -tf.reduce_mean(self.log_alpha * (tf.stop_gradient(current_log_probs) + self._target_entropy))
        
        # Update alpha
        alpha_gradient = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradient, [self.log_alpha]))
        
        # Get current alpha value (exp(log_alpha) ensures alpha > 0)
        alpha = tf.exp(self.log_alpha)
        
        next_actions, next_log_probs = self.actor.sample(states_, reparameterize=True)
        next_log_probs = tf.squeeze(next_log_probs, 1)
        
        
        target_q1 = self.target_critic_1(states_, next_actions) 
        target_q2 = self.target_critic_2(states_, next_actions)  
        target_q = tf.minimum(target_q1, target_q2)  
        
        
        next_log_probs = tf.reshape(next_log_probs, (-1, 1))  
        
        # SAC: Add entropy bonus (alpha * log_prob)
        target = rewards + self.gamma * (target_q - alpha * next_log_probs) * (1 - done)
        target = tf.stop_gradient(target)
        
        # Update both critics (inside GradientTape)
        with tf.GradientTape(persistent=True) as critic_tape:
            current_q1 = self.critic_1(states, actions)  
            current_q2 = self.critic_2(states, actions)  
            critic_loss_1 = keras.losses.MSE(target, current_q1)
            critic_loss_2 = keras.losses.MSE(target, current_q2)

        # Update critics
        critic_1_gradient = critic_tape.gradient(critic_loss_1,
                                                  self.critic_1.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_gradient, self.critic_1.trainable_variables))

        critic_2_gradient = critic_tape.gradient(critic_loss_2,
                                                  self.critic_2.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_gradient, self.critic_2.trainable_variables))
        
        # Explicitly delete persistent tape
        del critic_tape

        
        with tf.GradientTape() as tape:
            new_actions, logp = self.actor.sample(states, reparameterize=True)
            logp = tf.reshape(logp, (-1, 1))  
            
            q1_new = self.critic_1(states, new_actions)  
            q2_new = self.critic_2(states, new_actions)  
            q_new = tf.minimum(q1_new, q2_new)  

            actor_loss = tf.reduce_mean(alpha * logp - q_new)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks
        self.update_network_parameters()

class PotentialAgent:
    def __init__(self, alpha=0.002, env=None,
                 gamma=0.99, max_size=2000,
                 fc1=400, fc2=300, batch_size=64):
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_traj = PriorityMemory(max_size, batch_size, good_ratio=0.5)

        self.potential = PotentialNetwork(name='potential')
        self.potential.compile(optimizer=Adam(learning_rate=alpha))

        self.potential_learn_cnt = 0


    def add_trajectory(self, trajectory_epi, score, done):
        self.memory_traj.add(trajectory_epi, score, done)

    def save_models(self, exp):
        print('... saving potential models ...')
        save_path = f"tmp/{exp}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.potential.save_weights(save_path + self.potential.checkpoint_file)
        self.memory_traj.save(save_path)

    def load_models(self, exp):
        print('... loading potential models ...')
        save_path = f"tmp/{exp}/"
        self.potential.load_weights(save_path + self.potential.checkpoint_file)
        self.memory_traj.load(save_path)

    def reward_shaping(self, observation, observation_, evaluate):

        shaping_reward = 0

        if self.potential_learn_cnt > 100 or evaluate:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            state_ = tf.convert_to_tensor([observation_], dtype=tf.float32)

            p_new = self.potential(state_).numpy().item()
            p_old = self.potential(state).numpy().item()
            #breakpoint()
            shaping_reward = self.gamma * p_new - p_old

        return shaping_reward


    def learn_pf(self):

        good_states_trajs, bad_states_trajs = self.memory_traj.sample()
        if good_states_trajs or bad_states_trajs:

            x_list = np.unique(good_states_trajs + bad_states_trajs, axis=0).tolist()
            states = tf.convert_to_tensor(x_list, dtype=tf.float32)

            #import pdb; pdb.set_trace()
            good_states, good_cnt = np.unique(x_list + good_states_trajs, axis=0, return_counts=True)
            bad_states, bad_cnt = np.unique(x_list + bad_states_trajs, axis=0, return_counts=True)

            with tf.GradientTape() as tape:
                potential_value = tf.squeeze(self.potential(states), 1)
                target = (good_cnt - bad_cnt)/(good_cnt + bad_cnt -2)
                potential_loss = keras.losses.MSE(target, potential_value)

            potential_network_gradient = tape.gradient(potential_loss,
                                                    self.potential.trainable_variables)
            self.potential.optimizer.apply_gradients(zip(
                potential_network_gradient, self.potential.trainable_variables))

            self.potential_learn_cnt += 1


