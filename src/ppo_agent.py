"""
PPO (Proximal Policy Optimization) Agent for CXL Memory Fault Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ConvNet(nn.Module):
    """Convolutional neural network for processing memory maps"""
    
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
            nn.AdaptiveAvgPool2d((4, 4))  # Ensure consistent output size
        )
        
        # Calculate the size after conv layers
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
        x = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        x = self.fc(x)
        
        return x


class PolicyNetwork(nn.Module):
    """Policy network for action selection"""
    
    def __init__(self, feature_dim: int = 256, action_dims: List[int] = [14, 6, 64, 64]):
        super().__init__()
        
        self.feature_extractor = ConvNet(hidden_dim=feature_dim)
        
        # Separate heads for each action dimension
        self.algorithm_head = nn.Linear(feature_dim, action_dims[0])  # 14 algorithms
        self.pattern_head = nn.Linear(feature_dim, action_dims[1])    # 6 patterns
        self.start_head = nn.Linear(feature_dim, action_dims[2])      # 64 start regions
        self.end_head = nn.Linear(feature_dim, action_dims[3])        # 64 end regions
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.feature_extractor(state)
        
        algorithm_logits = self.algorithm_head(features)
        pattern_logits = self.pattern_head(features)
        start_logits = self.start_head(features)
        end_logits = self.end_head(features)
        
        return algorithm_logits, pattern_logits, start_logits, end_logits


class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.feature_extractor = ConvNet(hidden_dim=feature_dim)
        self.value_head = nn.Linear(feature_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        value = self.value_head(features)
        
        return value


class PPOAgent:
    """PPO Agent for memory fault detection optimization"""
    
    def __init__(self, 
                 state_shape: Tuple[int, ...] = (64, 64, 4),
                 action_dims: List[int] = [14, 6, 64, 64],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = torch.device(device)
        self.state_shape = state_shape
        self.action_dims = action_dims
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy_net = PolicyNetwork(action_dims=action_dims).to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = PPOBuffer()
        
        # Training stats
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100)
        }
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action given current state"""

        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            algorithm_logits, pattern_logits, start_logits, end_logits = self.policy_net(state_tensor)

            # Get value estimate
            value = self.value_net(state_tensor).squeeze().item()

            # Create distributions
            algorithm_dist = torch.distributions.Categorical(logits=algorithm_logits)
            pattern_dist = torch.distributions.Categorical(logits=pattern_logits)
            start_dist = torch.distributions.Categorical(logits=start_logits)
            end_dist = torch.distributions.Categorical(logits=end_logits)

            if deterministic:
                algorithm = torch.argmax(algorithm_logits, dim=-1)
                pattern = torch.argmax(pattern_logits, dim=-1)
                start_region = torch.argmax(start_logits, dim=-1)
                end_region = torch.argmax(end_logits, dim=-1)
            else:
                algorithm = algorithm_dist.sample()
                pattern = pattern_dist.sample()
                start_region = start_dist.sample()
                end_region = end_dist.sample()

            # Ensure end_region >= start_region
            end_region = torch.max(end_region, start_region)

            # Calculate log probabilities
            log_prob_algorithm = algorithm_dist.log_prob(algorithm)
            log_prob_pattern = pattern_dist.log_prob(pattern)
            log_prob_start = start_dist.log_prob(start_region)
            log_prob_end = end_dist.log_prob(end_region)

            total_log_prob = log_prob_algorithm + log_prob_pattern + log_prob_start + log_prob_end

            action = np.array([
                algorithm.item(),
                pattern.item(),
                start_region.item(),
                end_region.item()
            ])

            action_info = {
                'log_prob': total_log_prob.item(),
                'value': value,  # Add value estimate
                'algorithm_prob': torch.softmax(algorithm_logits, dim=-1)[0, algorithm].item(),
                'pattern_prob': torch.softmax(pattern_logits, dim=-1)[0, pattern].item(),
                'entropy': (algorithm_dist.entropy() + pattern_dist.entropy() +
                           start_dist.entropy() + end_dist.entropy()).item()
            }

        return action, action_info
    
    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate"""
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            value = self.value_net(state_tensor)
            
        return value.squeeze().item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, action_info: Dict):
        """Store transition in buffer"""
        
        self.buffer.store(state, action, reward, next_state, done, action_info)
    
    def update(self, batch_size: int = 64, epochs: int = 4) -> Dict:
        """Update policy and value networks"""
        
        if len(self.buffer) < batch_size:
            return {}
        
        # Get batch data
        states, actions, rewards, next_states, dones, old_log_probs, values = self.buffer.get_batch()
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)
        
        # Calculate advantages using GAE
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            advantages, returns = self._compute_gae(rewards, old_values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for epoch in range(epochs):
            # Sample batch
            indices = torch.randperm(len(states))[:batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]
            
            # Policy update
            policy_loss, entropy, kl_div = self._update_policy(
                batch_states, batch_actions, batch_old_log_probs, batch_advantages
            )
            
            # Value update
            value_loss = self._update_value(batch_states, batch_returns)
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
            total_kl += kl_div
        
        # Clear buffer
        self.buffer.clear()
        
        # Store training stats
        avg_policy_loss = total_policy_loss / epochs
        avg_value_loss = total_value_loss / epochs
        avg_entropy = total_entropy / epochs
        avg_kl = total_kl / epochs
        
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy'].append(avg_entropy)
        self.training_stats['kl_divergence'].append(avg_kl)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     next_values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] * (1 - dones[t].float())
            else:
                next_value = values[t + 1] * (1 - dones[t].float())
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t].float())
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def _update_policy(self, states: torch.Tensor, actions: torch.Tensor,
                      old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float, float]:
        """Update policy network"""
        
        # Get current policy outputs
        algorithm_logits, pattern_logits, start_logits, end_logits = self.policy_net(states)
        
        # Calculate new log probabilities
        algorithm_dist = torch.distributions.Categorical(logits=algorithm_logits)
        pattern_dist = torch.distributions.Categorical(logits=pattern_logits)
        start_dist = torch.distributions.Categorical(logits=start_logits)
        end_dist = torch.distributions.Categorical(logits=end_logits)
        
        new_log_probs = (
            algorithm_dist.log_prob(actions[:, 0]) +
            pattern_dist.log_prob(actions[:, 1]) +
            start_dist.log_prob(actions[:, 2]) +
            end_dist.log_prob(actions[:, 3])
        )
        
        # Calculate ratio and clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate entropy bonus
        entropy = (algorithm_dist.entropy() + pattern_dist.entropy() + 
                  start_dist.entropy() + end_dist.entropy()).mean()
        
        # Total loss
        total_loss = policy_loss - self.entropy_coef * entropy
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        # Calculate KL divergence for monitoring
        kl_div = (old_log_probs - new_log_probs).mean()
        
        return policy_loss.item(), entropy.item(), kl_div.item()
    
    def _update_value(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """Update value network"""
        
        predicted_values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(predicted_values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        stats = {}
        
        for key, values in self.training_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_last'] = values[-1]
        
        return stats


class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.value_net = None  # Will be set by agent

    def store(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool, action_info: Dict):
        """Store experience"""

        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': action_info['log_prob'],
            'value': action_info.get('value', 0.0)  # Store value estimate
        }

        self.buffer.append(experience)
    
    def get_batch(self) -> Tuple:
        """Get all experiences as batch"""

        states = np.array([exp['state'] for exp in self.buffer])
        actions = np.array([exp['action'] for exp in self.buffer])
        rewards = np.array([exp['reward'] for exp in self.buffer])
        next_states = np.array([exp['next_state'] for exp in self.buffer])
        dones = np.array([exp['done'] for exp in self.buffer])
        log_probs = np.array([exp['log_prob'] for exp in self.buffer])
        values = np.array([exp['value'] for exp in self.buffer])

        return states, actions, rewards, next_states, dones, log_probs, values
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)