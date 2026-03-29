import numpy as np
import os

class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-8, batch_size=200):
        """
        Running mean and std with batch updates
        
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  
        self.epsilon = epsilon
        
        # Batch collection
        self.batch_size = batch_size
        self.counter = 0
        self.batches = []

    def update(self, arr: np.ndarray) -> None:
        """
        Collect observation and update stats when batch is full
        """
        self.counter += 1  
        self.batches.append(arr)
        
        
        if self.counter >= self.batch_size:
        
            batches_array = np.array(self.batches)
            
            batch_mean = np.mean(batches_array, axis=0)
            batch_var = np.var(batches_array, axis=0)
            batch_count = batches_array.shape[0]
            
            # Update running statistics
            self.update_from_moments(batch_mean, batch_var, batch_count)
            
            # Reset for next batch
            self.counter = 0
            self.batches = []

    def update_from_moments(
        self, 
        batch_mean: np.ndarray, 
        batch_var: np.ndarray, 
        batch_count: float
    ) -> None:
        """
        Update running statistics from batch statistics
        Uses Welford's online algorithm for numerical stability
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count
        
        # Update variance using parallel variance formula
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        
        # Update count
        new_count = tot_count

        # Store new values
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, obs):
        """
        Normalize observation using running statistics
        
        Args:
            obs: Observation to normalize
        """
        obs = np.asarray(obs, dtype=np.float64)
        normalized = (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)
        
        return normalized.astype(np.float32)

    def save(self, exp):
        """Save normalization statistics as .npy"""
        save_dir = f"tmp/{exp}"
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "mean.npy"), self.mean)
        np.save(os.path.join(save_dir, "var.npy"), self.var)
        np.save(os.path.join(save_dir, "count.npy"), self.count)

        print(f"Saved normalization stats to {save_dir}")
    
    def load(self, exp):
        """Load normalization statistics from .npy"""
        load_dir = f"tmp/{exp}"

        mean_path = os.path.join(load_dir, "mean.npy")
        var_path = os.path.join(load_dir, "var.npy")
        count_path = os.path.join(load_dir, "count.npy")

        if not (os.path.exists(mean_path) and
                os.path.exists(var_path) and
                os.path.exists(count_path)):
            print(f"Warning: normalization files not found in {load_dir}, using defaults")
            return

        self.mean = np.load(mean_path)
        self.var = np.load(var_path)
        self.count = np.load(count_path)

        print(f"Loaded normalization stats from {load_dir}")