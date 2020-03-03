from random import random
from typing import Any, Dict

import numpy as np
import pickle
import torch
from torch import nn
from torch.optim import Adam
from typing import Optional

from src.networks import Actor, Critic
from src.noise import noise
from src.replay_buffer import ReplayBuffer
from src.transition import Transition
from src.utils import to_tensor_variable


class DDPG:
    def __init__(
        self,
        env,
        actor: Actor,
        critic: Critic,
        actor_target: Actor,
        critic_target: Critic,
        variables: Dict[str, Any],
        models_path: str,
        replay_buffer: Optional[ReplayBuffer] = None
    ):
        self.env = env
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = variables["gamma"]
        self.minibatch_size = variables["minibatch_size"]
        self.device = variables["device"]
        self.max_episodes = variables["max_episodes"]
        self.tau = variables["tau"]
        self.actor_optimizer = Adam(self.actor.parameters(), lr=variables["actor_lr"])
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=variables["critic_lr"], weight_decay=variables["weight_decay"]
        )
        self.critic_loss_fn = nn.MSELoss()
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(variables["replay_buffer_size"])
        self.models_path = models_path
        self.min_act_value = env.action_space.low[0]
        self.max_act_value = env.action_space.high[0]
        self.exploitation_episodes = 0
        self.exploration_episodes = 0
        self.policy_loss = 0
        self.critic_loss = 0

    def compute_expected_return_target(self, rewards, next_states, dones):
        """
        Compute the expected return obtained by evaluating the critic_target
        on the actor_target's policy.
        """
        with torch.no_grad():
            target_expectation = self.critic_target(next_states, self.actor_target(next_states))
            expected_return_target = rewards + (1 - dones) * self.gamma * target_expectation
            return expected_return_target

    def update_critic(self, states, actions, next_states, rewards, dones):
        """Update the critic network by minimizing the difference with the target critic"""
        self.critic_optimizer.zero_grad()
        expected_return_target = self.compute_expected_return_target(rewards, next_states, dones)
        # we take the positive because we want to minimize the loss
        expected_return_pred = self.critic(states, actions)
        self.critic_loss = self.critic_loss_fn(expected_return_pred, expected_return_target)
        self.critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        """Update the actor by maximizing the expected return."""
        self.actor_optimizer.zero_grad()
        # we take the negative because we want to maximize J (expected return)
        self.policy_loss = (-self.critic(states, self.actor(states))).mean()
        self.policy_loss.backward()
        self.actor_optimizer.step()

    def update_target_networks(self):
        """Soft update the target networks"""
        for param_a, param_a_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            param_a_target.data.copy_((1 - self.tau) * param_a_target.data + self.tau * param_a.data)

        for param_c, param_c_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_c_target.data.copy_((1 - self.tau) * param_c_target.data + self.tau * param_c.data)

    def get_minibatch(self):
        """Return `minibatch_size` (state, action, next_state, reward, done) tuples as Tensors."""
        minibatch = self.replay_buffer.get(self.minibatch_size)
        states = torch.stack([mb.state for mb in minibatch])[:, -1, :]
        actions = torch.tensor([mb.action for mb in minibatch], device=self.device)[:, -1, :]
        next_states = torch.stack([mb.next_state for mb in minibatch])[:, -1, :]
        rewards = torch.tensor([[mb.reward] for mb in minibatch], device=self.device)
        dones = torch.tensor([[int(mb.done)] for mb in minibatch], device=self.device)
        return states, actions, next_states, rewards, dones

    def log(self, rand, steps, sum_of_rewards):
        """Print information about the latest episode."""
        s = ""
        if rand:
            s += f"Exploration episode {self.exploration_episodes} "
        else:
            s += f"Exploitation episode {self.exploitation_episodes} "
        s += f"of {self.exploration_episodes + self.exploitation_episodes} total "
        s += f"finished after {steps} steps with sum of rewards {sum_of_rewards}"
        print(s)
        print(f"policy_loss = {self.policy_loss}")
        print(f"critic_loss = {self.critic_loss}\n")

    def save_models(self):
        """Save the current models to disk."""
        torch.save(self.critic, self.models_path + "critic")
        torch.save(self.actor, self.models_path + "actor")
        torch.save(self.critic_target, self.models_path + "critic_target")
        torch.save(self.actor_target, self.models_path + "actor_target")
        with open(self.models_path + "replay_buffer", "wb") as rb:
            pickle.dump(self.replay_buffer, rb, pickle.HIGHEST_PROTOCOL)

    def run_episode(self, rand=False):
        """Run a single episode in either exploration (rand=True) or exploitation mode."""
        state = to_tensor_variable([self.env.reset()])
        t = 0
        done = False
        sum_of_rewards = 0
        while not done:
            t += 1
            with torch.no_grad():
                if rand:
                    action = np.array([self.env.action_space.sample()])
                else:
                    self.actor.eval()
                    action = self.actor(state) + noise(self.env.action_space.shape)
                    self.actor.train()
                    action = action.data.clamp(self.min_act_value, self.max_act_value).to("cpu").numpy()

            next_state, reward, done, _ = self.env.step(action[0])
            next_state = to_tensor_variable([next_state])
            # fix this and the action (we want the type to be List[float])
            self.replay_buffer.store(Transition(state, action, next_state, reward, done))
            sum_of_rewards += reward
            state = next_state
            
            if self.replay_buffer.occupied > self.minibatch_size:
                states, actions, next_states, rewards, dones = self.get_minibatch()
                self.update_critic(states, actions, next_states, rewards, dones)
                self.update_actor(states)
                self.update_target_networks()

            if done:
                self.log(rand, t, sum_of_rewards)
                if (self.exploitation_episodes + self.exploration_episodes) % 100 == 0:
                    self.save_models()

    def explore(self, n_episodes):
        """Explores for n_episodes"""
        for i_episode in range(0, n_episodes):
            self.exploration_episodes += 1
            self.run_episode(rand=True)

    def train(self, initial_exploration=False):
        """Train the four networks"""
        if initial_exploration:
            self.explore(256)
        for i_episode in range(1, self.max_episodes):
            if 0 <= random() <= 0.05:
                self.explore(1)
            else:
                self.exploitation_episodes += 1
                self.run_episode(rand=False)
        self.env.close()
