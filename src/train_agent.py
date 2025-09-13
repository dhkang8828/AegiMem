"""
Training script for CXL Memory Fault Detection RL Agent
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from rl_environment import CXLMemoryRLEnvironment
from ppo_agent import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration"""
    
    def __init__(self):
        # Environment config
        self.mbist_binary_path = "/home/dhkang/data2/mbist_sample_code-gen2_es/bin/mbist_smbus.exe"
        self.memory_size = 0x100000000  # 4GB
        self.max_episode_steps = 500
        self.safety_mode = True  # Use simulation mode for safety
        
        # Training config
        self.total_episodes = 1000
        self.max_steps_per_episode = 500
        self.update_frequency = 64  # Update every N steps
        self.evaluation_frequency = 100  # Evaluate every N episodes
        self.save_frequency = 100  # Save model every N episodes
        
        # PPO config
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.batch_size = 64
        self.epochs_per_update = 4
        
        # Exploration config
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon_decay_episodes = 500
        
        # Output config
        self.model_save_dir = "/home/dhkang/cxl_memory_rl_project/models"
        self.log_dir = "/home/dhkang/cxl_memory_rl_project/logs"
        self.experiment_name = f"cxl_mbist_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class Trainer:
    """Main training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create directories
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize environment
        self.env = CXLMemoryRLEnvironment(
            mbist_binary_path=config.mbist_binary_path,
            memory_size=config.memory_size,
            max_episode_steps=config.max_episode_steps,
            safety_mode=config.safety_mode
        )
        
        # Initialize agent
        self.agent = PPOAgent(
            state_shape=(64, 64, 4),
            action_dims=[14, 6, 64, 64],
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_faults_found = []
        self.training_losses = []
        self.evaluation_results = []
        
        # Best performance tracking
        self.best_reward = float('-inf')
        self.best_faults_found = 0
        
        logger.info("Trainer initialized successfully")
        
    def train(self):
        """Main training loop"""
        
        logger.info(f"Starting training for {self.config.total_episodes} episodes")
        
        step_count = 0
        
        for episode in range(self.config.total_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_faults = 0
            
            done = False
            
            while not done and episode_length < self.config.max_steps_per_episode:
                # Select action with exploration
                epsilon = self._get_epsilon(episode)
                
                if np.random.random() < epsilon:
                    # Random exploration
                    action = np.array([
                        np.random.randint(0, 14),  # algorithm
                        np.random.randint(0, 6),   # pattern
                        np.random.randint(0, 64),  # start region
                        np.random.randint(0, 64)   # end region
                    ])
                    # Ensure end >= start
                    action[3] = max(action[3], action[2])
                    
                    action_info = {'log_prob': 0.0}  # Dummy for exploration
                else:
                    # Agent action
                    action, action_info = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done, action_info)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                step_count += 1
                
                if 'test_result' in step_info:
                    episode_faults += step_info['test_result'].get('faults_found', 0)
                
                state = next_state
                
                # Update agent periodically
                if step_count % self.config.update_frequency == 0:
                    update_info = self.agent.update(
                        batch_size=self.config.batch_size,
                        epochs=self.config.epochs_per_update
                    )
                    if update_info:
                        self.training_losses.append(update_info)
            
            # Record episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_faults_found.append(episode_faults)
            
            # Update best performance
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_best_model()
            
            if episode_faults > self.best_faults_found:
                self.best_faults_found = episode_faults
            
            episode_time = time.time() - episode_start_time
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_faults = np.mean(self.episode_faults_found[-10:])
                
                logger.info(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:8.2f} | "
                    f"Avg Reward: {avg_reward:8.2f} | "
                    f"Faults: {episode_faults:3d} | "
                    f"Steps: {episode_length:3d} | "
                    f"Time: {episode_time:.2f}s | "
                    f"Epsilon: {epsilon:.3f}"
                )
            
            # Evaluation
            if episode % self.config.evaluation_frequency == 0 and episode > 0:
                eval_result = self._evaluate()
                self.evaluation_results.append({
                    'episode': episode,
                    'eval_reward': eval_result['avg_reward'],
                    'eval_faults': eval_result['avg_faults'],
                    'eval_coverage': eval_result['avg_coverage']
                })
                
                logger.info(
                    f"Evaluation at episode {episode}: "
                    f"Avg Reward: {eval_result['avg_reward']:.2f}, "
                    f"Avg Faults: {eval_result['avg_faults']:.2f}, "
                    f"Avg Coverage: {eval_result['avg_coverage']:.2f}"
                )
            
            # Save model periodically
            if episode % self.config.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Save training progress
            if episode % 50 == 0:
                self._save_training_progress()
        
        logger.info("Training completed!")
        self._save_final_results()
    
    def _get_epsilon(self, episode: int) -> float:
        """Get exploration epsilon for current episode"""
        if episode >= self.config.epsilon_decay_episodes:
            return self.config.final_epsilon
        
        decay_ratio = episode / self.config.epsilon_decay_episodes
        return self.config.initial_epsilon - (self.config.initial_epsilon - self.config.final_epsilon) * decay_ratio
    
    def _evaluate(self, num_episodes: int = 5) -> Dict:
        """Evaluate agent performance"""
        
        eval_rewards = []
        eval_faults = []
        eval_coverages = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_faults = 0
            done = False
            
            while not done:
                # Deterministic action selection
                action, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                if 'test_result' in info:
                    episode_faults += info['test_result'].get('faults_found', 0)
                
                state = next_state
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            eval_faults.append(episode_faults)
            eval_coverages.append(info.get('test_coverage', 0))
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_faults': np.mean(eval_faults),
            'avg_coverage': np.mean(eval_coverages)
        }
    
    def _save_best_model(self):
        """Save best performing model"""
        filepath = os.path.join(self.config.model_save_dir, f"{self.config.experiment_name}_best.pt")
        self.agent.save_model(filepath)
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        filepath = os.path.join(self.config.model_save_dir, f"{self.config.experiment_name}_ep_{episode}.pt")
        self.agent.save_model(filepath)
        
        # Save training state
        checkpoint_data = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_faults_found': self.episode_faults_found,
            'best_reward': self.best_reward,
            'best_faults_found': self.best_faults_found,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.log_dir, f"{self.config.experiment_name}_checkpoint_{episode}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _save_training_progress(self):
        """Save training progress plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.config.experiment_name}')
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Faults found
        axes[1, 0].plot(self.episode_faults_found)
        axes[1, 0].set_title('Faults Found per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Faults')
        
        # Moving averages
        if len(self.episode_rewards) >= 50:
            window_size = 50
            moving_avg_rewards = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_faults = np.convolve(self.episode_faults_found, np.ones(window_size)/window_size, mode='valid')
            
            axes[1, 1].plot(moving_avg_rewards, label='Avg Reward')
            axes[1, 1].plot(moving_avg_faults, label='Avg Faults')
            axes[1, 1].set_title(f'Moving Averages (window={window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.log_dir, f"{self.config.experiment_name}_progress.png")
        plt.savefig(plot_path)
        plt.close()
    
    def _save_final_results(self):
        """Save final training results"""
        
        results = {
            'config': self.config.__dict__,
            'final_performance': {
                'best_reward': self.best_reward,
                'best_faults_found': self.best_faults_found,
                'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                'avg_faults_last_100': np.mean(self.episode_faults_found[-100:]) if len(self.episode_faults_found) >= 100 else np.mean(self.episode_faults_found)
            },
            'training_data': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'episode_faults_found': self.episode_faults_found,
                'evaluation_results': self.evaluation_results
            },
            'agent_stats': self.agent.get_training_stats()
        }
        
        results_path = os.path.join(self.config.log_dir, f"{self.config.experiment_name}_final_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final plots
        self._save_training_progress()
        
        logger.info(f"Final results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CXL Memory Fault Detection RL Agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--safety-mode', action='store_true', default=True, help='Use simulation mode for safety')
    parser.add_argument('--mbist-path', type=str, 
                       default="/home/dhkang/data2/mbist_sample_code-gen2_es/bin/mbist_smbus.exe",
                       help='Path to MBIST binary')
    parser.add_argument('--experiment-name', type=str, help='Custom experiment name')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.total_episodes = args.episodes
    config.safety_mode = args.safety_mode
    config.mbist_binary_path = args.mbist_path
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    logger.info(f"Starting training with config:")
    logger.info(f"  Episodes: {config.total_episodes}")
    logger.info(f"  Safety Mode: {config.safety_mode}")
    logger.info(f"  MBIST Path: {config.mbist_binary_path}")
    logger.info(f"  Experiment: {config.experiment_name}")
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()