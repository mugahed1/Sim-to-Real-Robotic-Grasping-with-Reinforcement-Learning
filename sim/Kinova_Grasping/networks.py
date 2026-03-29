import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256,  
                 name='critic', chkpt_dir=''):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_sac.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu', 
                         kernel_initializer='he_uniform')
        self.fc2 = Dense(self.fc2_dims, activation='relu',
                         kernel_initializer='he_uniform')
        self.q = Dense(1, activation=None,
                      kernel_initializer='glorot_uniform')

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.q(x)
        return q

class ActorNetwork(keras.Model):
    """
    SAC Actor Network: Outputs mean and log_std for Gaussian policy
    Supports reparameterization trick for sampling actions
    """
    def __init__(self, fc1_dims=256, fc2_dims=256,  
                 n_actions=2, max_action=1.0, name='actor', chkpt_dir=''):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_sac.h5')

        # Simplified architecture: 2 layers (256-256)
        self.fc1 = Dense(self.fc1_dims, activation='relu',
                         kernel_initializer='he_uniform')
        self.fc2 = Dense(self.fc2_dims, activation='relu',
                         kernel_initializer='he_uniform')
        
        # Output mean and log_std for Gaussian policy
        self.mu = Dense(self.n_actions, activation=None,
                       kernel_initializer='glorot_uniform')
        self.log_std = Dense(self.n_actions, activation=None,
                            kernel_initializer='glorot_uniform')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        # Clamp log_std to reasonable range for numerical stability
        log_std = tf.clip_by_value(log_std, -20, 2)
        return mu, log_std
    
    def sample(self, state, reparameterize=True):
        
        mu, log_std = self.call(state)
        std = tf.exp(log_std)
        
    
        epsilon = tf.random.normal(shape=tf.shape(mu))
        
        if not reparameterize:
            epsilon = tf.stop_gradient(epsilon)
        
        action = mu + std * epsilon
        
        log_prob = self._gaussian_log_prob(action, mu, std)
        
        action_tanh = tf.tanh(action)
        log_prob -= tf.reduce_sum(tf.math.log(1 - action_tanh**2 + 1e-6), axis=1, keepdims=True)
        
        action_scaled = action_tanh * self.max_action
        
        return action_scaled, log_prob
    
    def _gaussian_log_prob(self, x, mu, std):
        return -0.5 * tf.reduce_sum(
            tf.math.log(2.0 * tf.constant(np.pi, dtype=tf.float32)) + 
            2 * tf.math.log(std + 1e-6) + 
            tf.square((x - mu) / (std + 1e-6)), 
            axis=1, keepdims=True
        )
    
    def deterministic_action(self, state):
        mu, _ = self.call(state)
        return tf.tanh(mu) * self.max_action


class PotentialNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=256,
            name='potential', chkpt_dir=''):
        super(PotentialNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_sac.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.potential = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        potential = self.potential(state_value)

        return potential
    
