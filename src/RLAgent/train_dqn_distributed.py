"""
Training script for DQN on distributed CXL memory environment
"""

import argparse
import logging
import time
from pathlib import Path
import json
import requests

from phase1_environment_distributed import Phase1EnvironmentDistributed
from dqn_simple import SimpleDQNAgent


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def train_dqn(
    memory_agent_url: str,
    num_episodes: int = 100,
    max_steps_per_episode: int = 50,
    save_dir: str = "models",
    save_interval: int = 10,
    log_file: str = None
):
    """
    Train DQN agent on distributed environment

    Args:
        memory_agent_url: URL of Memory Agent REST API
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_dir: Directory to save models
        save_interval: Save model every N episodes
        log_file: Log file path
    """
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Test connection first
    logger.info("Testing connection to Memory Agent...")
    session = requests.Session()
    session.trust_env = False  # Disable proxy

    try:
        response = session.get(f"{memory_agent_url}/health", timeout=5)
        response.raise_for_status()
        logger.info(f"Connected to Memory Agent: {response.json()}")
    except Exception as e:
        logger.error(f"Failed to connect to Memory Agent: {e}")
        return

    # Create environment
    logger.info("Creating environment...")
    env = Phase1EnvironmentDistributed(
        memory_agent_url=memory_agent_url,
        max_sequence_length=max_steps_per_episode,
        verbose=True
    )

    # Create DQN agent
    logger.info("Creating DQN agent...")
    agent = SimpleDQNAgent(
        state_dim=6,
        num_actions=1536,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=1000
    )

    # Training statistics
    episode_rewards = []
    episode_losses = []
    episode_ce_counts = []
    best_reward = float('-inf')

    logger.info("="*60)
    logger.info("Starting DQN Training")
    logger.info("="*60)
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Max steps per episode: {max_steps_per_episode}")
    logger.info(f"Action space: {env.action_space.n}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info("="*60)

    # Training loop
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        # Reset environment
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss = []
        steps = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode}/{num_episodes}")
        logger.info(f"Epsilon: {agent.epsilon:.4f}")

        # Episode loop
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            logger.info(f"Step {step+1}/{max_steps_per_episode}: Action {action}")
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            steps += 1

            logger.info(f"  Reward: {reward:.2f}, CE total: {info.get('ce_total', 0)}")

            if done:
                logger.info(f"Episode terminated at step {steps}")
                break

            state = next_state

        # Decay epsilon
        agent.decay_epsilon()

        # Episode statistics
        episode_duration = time.time() - episode_start_time
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0

        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        episode_ce_counts.append(info.get('ce_total', 0))

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode} Summary:")
        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(f"  Steps: {steps}")
        logger.info(f"  Avg loss: {avg_loss:.4f}")
        logger.info(f"  CE detected: {info.get('ce_total', 0)}")
        logger.info(f"  Duration: {episode_duration:.2f}s")
        logger.info(f"  Epsilon: {agent.epsilon:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = save_path / "best_model.pt"
            agent.save(str(best_model_path))
            logger.info(f"  ⭐ New best model saved! Reward: {best_reward:.2f}")

        # Periodic save
        if episode % save_interval == 0:
            checkpoint_path = save_path / f"checkpoint_episode_{episode}.pt"
            agent.save(str(checkpoint_path))

            # Save training stats
            stats = {
                'episode': episode,
                'episode_rewards': episode_rewards,
                'episode_losses': episode_losses,
                'episode_ce_counts': episode_ce_counts,
                'best_reward': best_reward
            }
            stats_path = save_path / f"training_stats_episode_{episode}.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"  Checkpoint and stats saved at episode {episode}")

        # Running average
        if len(episode_rewards) >= 10:
            avg_reward_10 = sum(episode_rewards[-10:]) / 10
            logger.info(f"  Last 10 episodes avg reward: {avg_reward_10:.2f}")

    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Final epsilon: {agent.epsilon:.4f}")
    logger.info(f"Total episodes: {num_episodes}")
    logger.info("="*60)

    # Save final model
    final_model_path = save_path / "final_model.pt"
    agent.save(str(final_model_path))

    # Save final stats
    final_stats = {
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'episode_ce_counts': episode_ce_counts,
        'best_reward': best_reward,
        'final_epsilon': agent.epsilon
    }
    final_stats_path = save_path / "final_training_stats.json"
    with open(final_stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)

    logger.info(f"Final model and stats saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train DQN on distributed CXL memory environment'
    )
    parser.add_argument(
        '--memory-agent-url',
        type=str,
        default='http://192.168.3.20:5000',
        help='Memory Agent REST API URL'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=50,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='Save model every N episodes'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path (optional)'
    )

    args = parser.parse_args()

    train_dqn(
        memory_agent_url=args.memory_agent_url,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_file=args.log_file
    )


if __name__ == '__main__':
    main()
