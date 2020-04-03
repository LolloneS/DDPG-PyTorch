import torch

variables = {
    # Inferences
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Assumptions
    "max_episodes": int(5e6),
    # Values taken from https://arxiv.org/pdf/1509.02971.pdf, section 7
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "weight_decay": 1e-2,
    "gamma": 0.99,
    "tau": 0.001,
    "minibatch_size": 192,
    "replay_buffer_size": int(1e6),
}
