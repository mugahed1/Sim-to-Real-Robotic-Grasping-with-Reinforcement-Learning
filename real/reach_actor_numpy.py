import numpy as np


class ReachActorNP:
    """
    Pure NumPy equivalent of the reach ActorNetwork:
    15-dim input -> 6-dim tanh output in [-1, 1].

    Expected weights file: .npz with keys:
      w1, b1, w2, b2, w_mu, b_mu
    where:
      w1: (15, 128), b1: (128,)
      w2: (128, 128), b2: (128,)
      w_mu: (128, 6), b_mu: (6,)
    """

    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w_mu = data["w_mu"]
        self.b_mu = data["b_mu"]

    def forward(self, obs_norm: np.ndarray) -> np.ndarray:
        """
        obs_norm: shape (15,) or (1, 15)
        returns: shape (6,) in [-1, 1]
        """
        x = np.asarray(obs_norm, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]  # (1, 15)

        # Layer 1: ReLU
        x = x @ self.w1 + self.b1
        x = np.maximum(x, 0.0)

        # Layer 2: ReLU
        x = x @ self.w2 + self.b2
        x = np.maximum(x, 0.0)

        # Output mu and tanh (same as deterministic_action)
        mu = x @ self.w_mu + self.b_mu
        return np.tanh(mu)[0]

