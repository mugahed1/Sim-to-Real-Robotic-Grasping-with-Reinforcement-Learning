import numpy as np
import tensorflow as tf

from networks import ActorNetwork


MODEL_PATH = "model/exp1/actor_sac.h5"


OUT_PATH = "model/exp1/grasping_actor_weights.npz"


def main():
    
    actor = ActorNetwork(fc1_dims=128, fc2_dims=128, n_actions=7)

    
    _ = actor(tf.zeros((1, 29), dtype=tf.float32))

  
    actor.load_weights(MODEL_PATH)
    print(f"[OK] Loaded actor weights from {MODEL_PATH}")

    
    w1, b1 = actor.fc1.get_weights()
    w2, b2 = actor.fc2.get_weights()
    w_mu, b_mu = actor.mu.get_weights()

    
    np.savez(
        OUT_PATH,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w_mu=w_mu,
        b_mu=b_mu,
    )
    print(f"[OK] Saved NumPy weights to {OUT_PATH}")


if __name__ == "__main__":
    main()

