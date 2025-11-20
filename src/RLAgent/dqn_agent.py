"""
DQN (Deep Q-Network) Agent for CXL Memory Fault Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class ConvQNetwork(nn.Module):
    """Convolutional Q-Network for processing memory maps"""

    def __init__(self, input_channels: int = 4, hidden_dim: int = 256):
        super().__init__()

        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Calculate conv output size
        conv_output_size = 64 * 4 * 4  # 1024

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 64, 64, 4) -> (batch_size, 4, 64, 64)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension

        x = x.permute(0, 3, 1, 2)  # Move channels to dim 1
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class MultiHeadQNetwork(nn.Module):
    """Multi-head Q-Network for multi-discrete action space"""

    def __init__(self, feature_dim: int = 256, action_dims: List[int] = [14, 6, 64, 64]):
        super().__init__()

        self.action_dims = action_dims
        self.feature_extractor = ConvQNetwork(hidden_dim=feature_dim)

        # Separate Q-value heads for each action dimension
        self.algorithm_head = nn.Linear(feature_dim, action_dims[0])  # 14 algorithms
        self.pattern_head = nn.Linear(feature_dim, action_dims[1])    # 6 patterns
        self.start_head = nn.Linear(feature_dim, action_dims[2])      # 64 start regions
        self.end_head = nn.Linear(feature_dim, action_dims[3])        # 64 end regions

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass

        Returns:
            Tuple of Q-values for each action dimension
        """
        features = self.feature_extractor(state)

        algorithm_q = self.algorithm_head(features)
        pattern_q = self.pattern_head(features)
        start_q = self.start_head(features)
        end_q = self.end_head(features)

        return algorithm_q, pattern_q, start_q, end_q


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """Store experience"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for memory fault detection optimization"""

    def __init__(self,
                 state_shape: Tuple[int, ...] = (64, 64, 4),
                 action_dims: List[int] = [14, 6, 64, 64],
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_frequency: int = 1000,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = torch.device(device)
        self.state_shape = state_shape
        self.action_dims = action_dims
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.batch_size = batch_size

        # Networks
        self.q_network = MultiHeadQNetwork(action_dims=action_dims).to(self.device)
        self.target_network = MultiHeadQNetwork(action_dims=action_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training stats
        self.training_stats = {
            'loss': deque(maxlen=100),
            'q_values': deque(maxlen=100),
            'epsilon': deque(maxlen=100)
        }

        self.steps = 0

        logger.info(f"DQN Agent initialized on device: {self.device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action as numpy array [algorithm, pattern, start, end]
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random action
            action = np.array([
                random.randint(0, self.action_dims[0] - 1),
                random.randint(0, self.action_dims[1] - 1),
                random.randint(0, self.action_dims[2] - 1),
                random.randint(0, self.action_dims[3] - 1)
            ])
            # Ensure end >= start
            action[3] = max(action[3], action[2])
            return action

        # Greedy action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            algorithm_q, pattern_q, start_q, end_q = self.q_network(state_tensor)

            algorithm = torch.argmax(algorithm_q, dim=1).item()
            pattern = torch.argmax(pattern_q, dim=1).item()
            start = torch.argmax(start_q, dim=1).item()
            end = torch.argmax(end_q, dim=1).item()

            # Ensure end >= start
            end = max(end, start)

            action = np.array([algorithm, pattern, start, end])

        return action

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict]:
        """
        Update Q-network using experience replay

        Returns:
            Dictionary with training metrics, or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_algorithm, current_q_pattern, current_q_start, current_q_end = self.q_network(states)

        # Get Q-values for selected actions
        current_q_algorithm = current_q_algorithm.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
        current_q_pattern = current_q_pattern.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)
        current_q_start = current_q_start.gather(1, actions[:, 2].unsqueeze(1)).squeeze(1)
        current_q_end = current_q_end.gather(1, actions[:, 3].unsqueeze(1)).squeeze(1)

        # Target Q-values (using Double DQN)
        with torch.no_grad():
            # Use online network to select actions
            next_q_algorithm, next_q_pattern, next_q_start, next_q_end = self.q_network(next_states)
            next_actions_algorithm = torch.argmax(next_q_algorithm, dim=1)
            next_actions_pattern = torch.argmax(next_q_pattern, dim=1)
            next_actions_start = torch.argmax(next_q_start, dim=1)
            next_actions_end = torch.argmax(next_q_end, dim=1)

            # Use target network to evaluate actions
            target_q_algorithm, target_q_pattern, target_q_start, target_q_end = self.target_network(next_states)

            next_q_algorithm = target_q_algorithm.gather(1, next_actions_algorithm.unsqueeze(1)).squeeze(1)
            next_q_pattern = target_q_pattern.gather(1, next_actions_pattern.unsqueeze(1)).squeeze(1)
            next_q_start = target_q_start.gather(1, next_actions_start.unsqueeze(1)).squeeze(1)
            next_q_end = target_q_end.gather(1, next_actions_end.unsqueeze(1)).squeeze(1)

            # Average Q-values across action dimensions
            next_q = (next_q_algorithm + next_q_pattern + next_q_start + next_q_end) / 4.0

            # Calculate target
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calculate loss (average across action dimensions)
        current_q = (current_q_algorithm + current_q_pattern + current_q_start + current_q_end) / 4.0
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Target network updated at step {self.steps}")

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Store stats
        self.training_stats['loss'].append(loss.item())
        self.training_stats['q_values'].append(current_q.mean().item())
        self.training_stats['epsilon'].append(self.epsilon)

        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'epsilon': self.epsilon
        }

    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)

        logger.info(f"DQN model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)

        logger.info(f"DQN model loaded from {filepath}")

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        stats = {}

        for key, values in self.training_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_last'] = values[-1]

        return stats
