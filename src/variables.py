import torch

variables = {
    # Inferences
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Assumptions
    "max_episodes": int(5e6),
    # Values taken from https://arxiv.org/pdf/1509.02971.pdf, section 7
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "l2_weight_decay": 1e-2,
    "gamma": 0.99,
    "tau": 0.001,
    "minibatch_size": 64,
    "replay_buffer_size": 1e6,
    "final_layer_w_b": 3e-3,
}
