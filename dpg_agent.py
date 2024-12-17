# dpg_agent.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Deterministic Policy Gradient (DPG) Agent
class DPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001):
        # Initialize actor, critic, and target networks
        self.actor = self.build_actor(state_dim, action_dim).to(device)
        self.critic = self.build_critic(state_dim, action_dim).to(device)
        self.target_actor = self.build_actor(state_dim, action_dim).to(device)
        self.target_critic = self.build_critic(state_dim, action_dim).to(device)
        self.update_target_networks(1.0)
        self.history_buffer = []

        # Define optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Parameters
        self.gamma = gamma
        self.tau = tau

    # Build the actor network to output actions
    def build_actor(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    # Build the critic network to evaluate state-action pairs
    def build_critic(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),  # Increase hidden layer size
            nn.ReLU(),
            nn.Linear(256, 256),  # Add another layer
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    # Apply softmax with temperature for action probabilities
    def softmax_with_temperature(self, logits, temperature=1.0):
        exp_logits = torch.exp(logits / temperature)
        return exp_logits / torch.sum(exp_logits)

    # Apply softmax and clip the values to a range
    def clip_softmax(self, logits, min_val=0.0, max_val=1.0):
        logits = torch.tensor(logits, dtype=torch.float32, device=device)
        softmaxed = F.softmax(logits, dim=0)
        clipped = torch.clamp(softmaxed, min_val, max_val)
        return clipped

    # Select an action with optional temperature scaling for exploration
    def act(self, state, episode=None):
        logits = self.actor(state.unsqueeze(0)).detach().cpu().numpy()[0]

        action = self.clip_softmax(logits).numpy()
        temperature = max(0.1, 1 - (episode / 2000)) if episode else 1.0
        action /= temperature
        return np.clip(action, 0, 1)

    # Perform soft update of target networks with a given tau
    def update_target_networks(self, tau=None):
        tau = tau if tau is not None else self.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # Compute the historical average of past states
    def compute_historical_average(self, history_buffer):
        if len(history_buffer) < 2:
            return torch.zeros_like(history_buffer[0])  # No historical average for the first step
        return torch.sum(torch.stack(history_buffer[:-1]), dim=0) / (len(history_buffer) - 1)

    # Update the current state using historical average and recent contributions.
    def update_state(self, historical_avg, current_state, action, prediction, t):
        decay_factor = (1 - 1 / t)
        recent_contribution = (1 / t) * prediction * action
        updated_state = decay_factor * historical_avg + recent_contribution
        return updated_state

    # Update actor and critic networks using experiences and historical averages
    def update(self, state, action, reward, next_state, episode, max_episodes):
        if len(self.history_buffer) >= max_episodes:
            self.history_buffer.pop(0)
        self.history_buffer.append(state)

        historical_avg = self.compute_historical_average(self.history_buffer)

        prediction = self.actor(state).detach()
        updated_state = self.update_state(historical_avg, state, action, prediction, episode + 1)

        updated_state = torch.FloatTensor(updated_state).to(device) if not isinstance(updated_state,
                                                                                      torch.Tensor) else updated_state
        action = torch.FloatTensor(action).to(device) if not isinstance(action, torch.Tensor) else action
        reward = torch.clamp(torch.FloatTensor([reward]).to(device), min=-1, max=1)
        next_state = torch.FloatTensor(next_state).to(device) if not isinstance(next_state,
                                                                                torch.Tensor) else next_state

        updated_state = updated_state.unsqueeze(0) if updated_state.dim() == 1 else updated_state
        action = action.unsqueeze(0) if action.dim() == 1 else action
        next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state

        # Compute target Q-value
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_state_action = torch.cat([next_state, next_action], dim=1)
            target_q_value = reward + self.gamma * self.target_critic(next_state_action)
            target_q_value = torch.clamp(target_q_value, min=-10, max=10)

        # Compute critic loss and update critic
        state_action = torch.cat([updated_state, action], dim=1)
        predicted_q_value = self.critic(state_action)

        critic_loss = F.mse_loss(predicted_q_value, target_q_value.detach())
        critic_loss = torch.clamp(critic_loss, min=1e-6, max=10)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()

        # Compute actor loss and update actor
        predicted_action = self.actor(updated_state)
        state_action = torch.cat([updated_state, predicted_action], dim=1)
        actor_loss = -self.critic(state_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks(tau=0.005)

        if torch.isnan(actor_loss).any() or torch.isnan(critic_loss).any():
            print("Nan encountered in actor/critic loss!")

        return actor_loss.item(), critic_loss.item()
