"""
Simple DQN Agent for CXL Memory Fault Detection
Designed for Discrete(1536) action space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleQNetwork(nn.Module):
    """Simple Q-Network for discrete action space (1536 actions)"""

    def __init__(self, state_dim: int = 6, num_actions: int = 1536, hidden_dim: int = 256):
        """
        Args:
            state_dim: Flattened state dimension
            num_actions: Number of discrete actions (1536)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values for all actions (batch_size, num_actions)
        """
        return self.network(state)


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store experience"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def sample(self, batch_size: int) -> Dict:
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

    def __len__(self):
        return len(self.buffer)


class SimpleDQNAgent:
    """Simple DQN Agent for Discrete(1536) action space"""

    def __init__(
        self,
        state_dim: int = 6,
        num_actions: int = 1536,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Simple DQN Agent

        Args:
            state_dim: State dimension
            num_actions: Number of actions (1536)
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            device: Device (cuda/cpu)
        """
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-Networks
        self.q_network = SimpleQNetwork(state_dim, num_actions).to(self.device)
        self.target_network = SimpleQNetwork(state_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training stats
        self.train_step = 0
        self.episode_count = 0

        logger.info(f"SimpleDQN initialized on {self.device}")
        logger.info(f"State dim: {state_dim}, Actions: {num_actions}")

    def flatten_state(self, state: Dict) -> np.ndarray:
        """
        Flatten dictionary state to vector

        Args:
            state: Dict observation from environment

        Returns:
            Flattened state vector
        """
        # Extract key features from dict state
        features = []

        # Last action from sequence history (if exists)
        if 'sequence_history' in state and len(state['sequence_history']) > 0:
            features.append(float(state['sequence_history'][-1]))
        else:
            features.append(0.0)

        # Last test result
        features.append(float(state.get('last_test_result', 0)))

        # Sequence length
        features.append(float(state.get('sequence_length', 0)))

        # Cumulative errors
        if 'cumulative_errors' in state:
            features.append(float(state['cumulative_errors'][0]))
        else:
            features.append(0.0)

        # Add some derived features
        features.append(float(len(state.get('sequence_history', []))))  # History length
        features.append(1.0 if state.get('last_test_result', 0) == 1 else 0.0)  # FAIL flag

        return np.array(features, dtype=np.float32)

    def select_action(self, state: Dict, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Environment state
            training: If True, use epsilon-greedy; else greedy

        Returns:
            Selected action (0-1535)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Greedy action
        state_vector = self.flatten_state(state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_experience(self, state: Dict, action: int, reward: float,
                        next_state: Dict, done: bool):
        """Store experience in replay buffer"""
        state_vector = self.flatten_state(state)
        next_state_vector = self.flatten_state(next_state)
        self.replay_buffer.push(state_vector, action, reward, next_state_vector, done)

    def train(self) -> Optional[float]:
        """
        Train on batch from replay buffer

        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Target network updated at step {self.train_step}")

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'epsilon': self.epsilon
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']
        self.epsilon = checkpoint['epsilon']
        logger.info(f"Model loaded from {path}")
