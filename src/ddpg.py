from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

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
        gamma: float,
        minibatch_size: int,
        device: torch.device,
        max_episodes: int,
        tau: int,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        replay_buffer_size: int,
        models_path: str,
        runs_path: Optional[str],
    ):
        self.env = env
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.device = device
        self.max_episodes = max_episodes
        self.tau = tau
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay,
        )
        self.critic_loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.models_path = models_path
        self.min_act_value = env.action_space.low[0]
        self.max_act_value = env.action_space.high[0]
        self.writer = SummaryWriter(log_dir=runs_path)
        self.actor_target.eval()
        self.critic_target.eval()
        self.episode_i = 0

    def compute_expected_return_target(self, rewards, next_states, dones):
        """
        Compute the expected return obtained by evaluating
        the critic_target on the actor_target's policy.
        """
        with torch.no_grad():
            target_expectation = self.critic_target(
                next_states, self.actor_target(next_states)
            )
            expected_return_target = (
                rewards + (1 - dones) * self.gamma * target_expectation
            )
            return expected_return_target

    def update_critic(self, states, actions, next_states, rewards, dones):
        """Update the critic network by minimizing the difference
        with the target critic"""
        self.critic_optimizer.zero_grad()
        expected_return_target = self.compute_expected_return_target(
            rewards, next_states, dones
        )
        expected_return_pred = self.critic(states, actions)
        critic_loss = self.critic_loss_fn(
            expected_return_pred, expected_return_target
        )
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def update_actor(self, states):
        """Update the actor by maximizing the expected return."""
        self.actor_optimizer.zero_grad()
        self.critic.eval()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.critic.train()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def update_target_networks(self):
        """Soft update the target networks"""
        with torch.no_grad():
            for p_a, p_a_target in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                p_a_target.data.mul_((1.0 - self.tau))
                p_a_target.data.add_(self.tau * p_a.data)

            for p_c, p_c_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_c_target.data.mul_(1.0 - self.tau)
                p_c_target.data.add_(self.tau * p_c.data)

    def get_minibatch(self):
        """Return `minibatch_size` (state, action, next_state, reward, done)
        tuples as Tensors."""
        minibatch = self.replay_buffer.get(self.minibatch_size)
        states = torch.stack([mb.state for mb in minibatch])[:, -1, :]
        actions = [mb.action for mb in minibatch]
        actions = torch.tensor(actions, device=self.device)[:, -1, :]
        next_states = [mb.next_state for mb in minibatch]
        next_states = torch.stack(next_states)[:, -1, :]
        rewards = [[mb.reward] for mb in minibatch]
        rewards = torch.tensor(rewards, device=self.device)
        dones = [[int(mb.done)] for mb in minibatch]
        dones = torch.tensor(dones, device=self.device)
        return states, actions, next_states, rewards, dones

    def save_models(self):
        """Save the current models to disk."""
        torch.save(self.critic, self.models_path + "critic")
        torch.save(self.actor, self.models_path + "actor")
        torch.save(self.critic_target, self.models_path + "critic_target")
        torch.save(self.actor_target, self.models_path + "actor_target")

    def select_action(self, state, explore=True):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).to("cpu").data.numpy()
        self.actor.train()
        if explore:
            action += noise(self.env.action_space.shape)
            action = action.clip(self.min_act_value, self.max_act_value)
        return action

    def log(self, sum_of_actor_losses, sum_of_critic_losses, reward):
        if self.episode_i % 20 == 0:
            self.save_models()
        self.writer.add_scalar(
            "ActorLoss/train", sum_of_actor_losses, self.episode_i
        )
        self.writer.add_scalar(
            "CriticLoss/train", sum_of_critic_losses, self.episode_i
        )
        self.writer.add_scalar("Reward/train", reward, self.episode_i)

    def run_episode(self, explore=True):
        """Run a single episode in either exploration (explore=True)
        or exploitation (explore=False) mode."""
        self.episode_i += 1
        state = to_tensor_variable([self.env.reset()])
        t = 0
        done = False
        sum_of_actor_losses = 0
        sum_of_critic_losses = 0

        while not done:
            t += 1
            with torch.no_grad():
                action = self.select_action(state, explore)
            next_state, reward, done, _ = self.env.step(action[0])
            next_state = to_tensor_variable([next_state])
            self.replay_buffer.store(
                Transition(state, action, next_state, reward, done)
            )
            state = next_state

            if explore and self.replay_buffer.occupied > self.minibatch_size:
                (
                    states,
                    actions,
                    next_states,
                    rewards,
                    dones,
                ) = self.get_minibatch()
                sum_of_critic_losses += self.update_critic(
                    states, actions, next_states, rewards, dones
                )
                self.critic.eval()
                sum_of_actor_losses += self.update_actor(states)
                self.critic.train()
                self.update_target_networks()

            if done:
                self.log(sum_of_actor_losses, sum_of_critic_losses, reward)
                return reward

    def run_random_episodes(self, n_episodes):
        for _ in range(n_episodes):
            state = to_tensor_variable([self.env.reset()])
            done = False
            while not done:
                action = np.array([self.env.action_space.sample()])
                next_state, reward, done, _ = self.env.step(action[0])
                next_state = to_tensor_variable([next_state])
                self.replay_buffer.store(
                    Transition(state, action, next_state, reward, done)
                )

    def exploit(self, n_episodes):
        """Exploits for n_episodes"""
        rewards = [self.run_episode(explore=False) for _ in range(n_episodes)]
        return rewards

    def explore(self, n_episodes):
        """Explores for n_episodes"""
        for _ in range(n_episodes):
            self.run_episode(explore=True)

    def train(self):
        """Train the four networks"""
        self.run_random_episodes(100)
        count_exploit = 0
        count_explore = 0
        while self.episode_i < self.max_episodes:
            self.explore(50)
            count_explore += 50
            rewards = self.exploit(10)
            count_exploit += 10
            s = f"Average final reward after 10 exploitations and "
            s += f"{count_explore} explorations: {sum(rewards)/len(rewards)}\n"
            print(s)
            if sum(rewards) / len(rewards) > 180:
                break
        self.env.close()
        return
