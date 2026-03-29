import numpy as np
import tensorflow as tf

from networks import ActorNetwork


# Path to your trained reach actor (TensorFlow/Keras weights)
MODEL_PATH = "model/reach_sac-dr/actor_sac.h5"

# Where to save the NumPy weights used by ReachActorNP
OUT_PATH = "model/reach_sac-dr/reach_actor_weights.npz"


def main():
    # 15-dim obs: [ee_pos(3), goal(3), rel(3), joint_angles(6)]
    actor = ActorNetwork(fc1_dims=128, fc2_dims=128, n_actions=6)

    # Build the network by calling it once with dummy input
    _ = actor(tf.zeros((1, 15), dtype=tf.float32))

    # Load trained weights
    actor.load_weights(MODEL_PATH)
    print(f"[OK] Loaded actor weights from {MODEL_PATH}")

    # Extract layer weights
    w1, b1 = actor.fc1.get_weights()
    w2, b2 = actor.fc2.get_weights()
    w_mu, b_mu = actor.mu.get_weights()

    # Save to .npz for ReachActorNP
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

