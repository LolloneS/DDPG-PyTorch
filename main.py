# per i plot provare seaborn

from copy import deepcopy
from pathlib import Path

import gym
import pickle
import torch

from src.ddpg import DDPG
from src.networks import Actor, Critic
from src.variables import variables

env = gym.make("LunarLanderContinuous-v2")
# env = gym.wrappers.Monitor(env, "videos/", force=True)
models_path = "models/"


have_models = Path(models_path).exists() and list(Path(models_path).iterdir()) != []
if have_models:
    print("Loading pre-existent models.")

actor = torch.load(f"{models_path}actor") if have_models else Actor()
critic = torch.load(f"{models_path}critic") if have_models else Critic()
actor_target = torch.load(f"{models_path}actor_target") if have_models else deepcopy(actor)
critic_target = torch.load(f"{models_path}critic_target") if have_models else deepcopy(critic)
with open(models_path + "replay_buffer", "rb") as rb:
    replay_buffer = pickle.load(rb)

ddpg = DDPG(env, actor, critic, actor_target, critic_target, variables, models_path, replay_buffer)

ddpg.train()
