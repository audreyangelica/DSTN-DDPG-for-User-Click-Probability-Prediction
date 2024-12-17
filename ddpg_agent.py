# ddpg_agent.py


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

# Set GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Ornstein-Uhlenbeck (OU) Noise for action exploration in continuous spaces
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    # Reset noise to its mean value
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    # Generate noise sample based on OU process dynamics
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# Noisy Linear Layer for exploration in neural networks
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters for the mean of weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for noise values
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    # Initialize parameters for mean and standard deviation
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    # Reset noise buffers with random values
    def reset_noise(self):
        device = self.weight_mu.device

        epsilon_in = torch.randn(self.in_features, device=device).sign() * torch.sqrt(
            torch.abs(torch.randn(self.in_features, device=device)))
        epsilon_out = torch.randn(self.out_features, device=device).sign() * torch.sqrt(
            torch.abs(torch.randn(self.out_features, device=device)))

        self.weight_epsilon = epsilon_out.ger(epsilon_in).to(device)
        self.bias_epsilon = epsilon_out.to(device)

    # Forward pass with noise applied to weights and biases
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

# Deep Deterministic Policy Gradient (DDPG) Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, buffer_size, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.actor_model = self.build_actor().to(device)
        self.target_actor_model = self.build_actor().to(device)
        self.critic_model = self.build_critic().to(device)
        self.target_critic_model = self.build_critic().to(device)
        self.train_step = 0
        self.noise = OUNoise(action_dim=self.action_dim)

        # Optimizers for the actor and critic networks
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)

    # Build the actor network
    def build_actor(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        return model

    # Build the critic network to evaluate Q-values
    def build_critic(self):
        class Critic(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(Critic, self).__init__()
                self.fc1 = nn.Linear(state_dim, 256)
                self.fc2 = nn.Linear(256 + action_dim, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 1)

            def forward(self, state, action):
                x = F.relu(self.fc1(state))
                x = torch.cat([x, action], dim=1)
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                return self.fc4(x)

        return Critic(self.state_dim, self.action_dim)

    # Update the state using historical average and recent contributions
    def update_state(self, historical_avg, state, action, prediction, t):
        if action.dim() == 1:
            action = action.unsqueeze(0).repeat(prediction.size(0), 1)

        recent_contribution = (1 / t) * prediction.unsqueeze(1) * action
        new_state = (1 - 1 / t) * historical_avg + recent_contribution.mean(dim=0)
        return new_state

    # Compute the historical average of past states
    def compute_historical_average(self, past_states, t):
        if len(past_states) == 0 or t < 2:
            return torch.zeros(self.state_dim, dtype=torch.float32, device=device)
        historical_avg = torch.stack(past_states[:t - 2]).mean(dim=0)
        return historical_avg

    # Store experience in the replay buffer with state updates
    def remember(self, state, action, reward, next_state, prediction, t):
        if len(self.buffer) > 0:
            historical_avg = sum([s[0] for s in self.buffer]) / len(self.buffer)
        else:
            historical_avg = torch.zeros_like(state)

        updated_state = self.update_state(historical_avg, state, action, prediction, t)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        clipped_reward = torch.clamp(reward, -1.0, 1.0)

        self.buffer.append((updated_state.detach(), action.detach(), clipped_reward.detach(), next_state.detach()))

    # Apply softmax and clip values to the specified range
    def clip_softmax(self, logits, min_val, max_val):
        logits = torch.tensor(logits, dtype=torch.float32, device=device)
        softmaxed = F.softmax(logits, dim=0)
        clipped = torch.clamp(softmaxed, min_val, max_val)
        return clipped

    # Generate an action with exploration noise
    def act(self, state, episode=None):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits = self.actor_model(state).detach().cpu().numpy()[0]

        if episode is not None:
            noise_scale = max(0.1, 1 - (episode / 2000))
            self.noise.sigma = 0.2 * noise_scale
            self.noise.theta = 0.15 * noise_scale
        else:
            noise_scale = 1.0

        self.noise.reset() if episode == 0 else None
        noise = noise_scale * self.noise.sample()

        action = logits + noise
        action = self.clip_softmax(action, min_val=0.0, max_val=1.0).numpy()

        return np.clip(action, 0, 1)

    # Perform soft update of target network parameters
    def soft_update(self, target_model, source_model, tau=0.001):
        for target_param, param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # Train actor and critic networks using replay buffer
    def train(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(
            [r.mean().item() if isinstance(r, torch.Tensor) else float(r) for r in rewards],
            dtype=torch.float32, device=device
        ).view(-1, 1)
        next_states = torch.stack(next_states).to(device)

        with torch.no_grad():
            target_actions = self.target_actor_model(next_states)
            target_q_values = self.target_critic_model(next_states, target_actions).view(-1, 1)
            target_q_values = rewards + self.gamma * target_q_values
            target_q_values = target_q_values.detach()

        # Update critic network
        predicted_q_values = self.critic_model(states, actions).view(-1, 1)

        critic_loss = F.mse_loss(predicted_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update actor network
        predicted_actions = self.actor_model(states)
        actor_loss = -self.critic_model(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.target_actor_model, self.actor_model)
        self.soft_update(self.target_critic_model, self.critic_model)

        return actor_loss.item(), critic_loss.item()

    # Evaluate the actor loss for a given state
    def test_actor_loss(self, state):
        self.actor_model.eval()
        self.critic_model.eval()

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = self.actor_model(state_tensor)
            q_value = self.critic_model(state_tensor, action)
            actor_loss = -q_value.mean().item()
        return actor_loss
