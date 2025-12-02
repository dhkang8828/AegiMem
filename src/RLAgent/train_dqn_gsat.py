"""
Training script for DQN on GSAT-style CXL memory environment

Action Space: 64 actions (4 operations × 16 key patterns)
Based on Google StressAppTest: https://github.com/stressapptest/stressapptest
"""

import argparse
import logging
import time
from pathlib import Path
import json
import requests

from phase1_gsat_like_environment_distributed import Phase1GSATEnvironmentDistributed
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


def train_dqn_gsat(
    memory_agent_url: str,
    num_episodes: int = 100,
    max_steps_per_episode: int = 50,
    save_dir: str = "models_gsat",
    save_interval: int = 10,
    log_file: str = None
):
    """
    Train DQN agent on GSAT-style distributed environment

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
    logger.info("Creating GSAT-style environment...")
    env = Phase1GSATEnvironmentDistributed(
        memory_agent_url=memory_agent_url,
        max_sequence_length=max_steps_per_episode,
        verbose=True
    )

    # Create DQN agent (64 actions instead of 1536)
    logger.info("Creating DQN agent...")
    agent = SimpleDQNAgent(
        state_dim=6,           # 6 observation features
        num_actions=64,        # 4 ops × 16 patterns = 64 actions
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,         # Smaller batch for 64 actions
        target_update_freq=500  # More frequent updates
    )

    # Training statistics
    episode_rewards = []
    episode_losses = []
    episode_ce_counts = []
    best_reward = float('-inf')
    best_ce_count = 0

    logger.info("="*60)
    logger.info("Starting GSAT-style DQN Training")
    logger.info("="*60)
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Max steps per episode: {max_steps_per_episode}")
    logger.info(f"Action space: {env.action_space.n} (4 ops × 16 patterns)")
    logger.info(f"Observation space: {env.observation_space.shape}")
    logger.info("="*60)
    logger.info("\nKey Patterns (StressAppTest):")
    logger.info("  0x00, 0xFF, 0x55, 0xAA, 0xF0, 0x0F, 0xCC, 0x33,")
    logger.info("  0x01, 0x80, 0x16, 0xB5, 0x4A, 0x57, 0x02, 0xFD")
    logger.info("\nRecommended Actions for CE:")
    logger.info("  Action 17: INVERT with 0x55 (01010101)")
    logger.info("  Action 19: INVERT with 0xAA (10101010)")
    logger.info("  Action 16: INVERT with 0xFF (11111111)")
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

            # Decode action for logging
            operation = action // 16
            pattern_idx = action % 16
            pattern_byte = env.KEY_PATTERNS[pattern_idx]
            op_name = env.OPERATION_NAMES[operation]

            # Execute action
            logger.info(f"Step {step+1}/{max_steps_per_episode}: "
                       f"Action {action} ({op_name} with 0x{pattern_byte:02X})")
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

            ce_total = info.get('ce_total', 0)
            logger.info(f"  Reward: {reward:.2f}, CE: {ce_total}, "
                       f"Temp: {info.get('temperature', 0)}°C")

            # Celebrate CE detection
            if ce_total > 0:
                logger.info(f"  *** CE DETECTED! Total CE in episode: {info.get('total_ce_episode', 0)} ***")

            if done:
                logger.info(f"Episode terminated at step {steps}")
                break

            state = next_state

        # Decay epsilon
        agent.decay_epsilon()

        # Episode statistics
        episode_duration = time.time() - episode_start_time
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0
        total_ce_episode = info.get('total_ce_episode', 0)

        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        episode_ce_counts.append(total_ce_episode)

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode} Summary:")
        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(f"  Steps: {steps}")
        logger.info(f"  Avg loss: {avg_loss:.4f}")
        logger.info(f"  CE detected (episode): {total_ce_episode}")
        logger.info(f"  Unique actions: {info.get('unique_actions', 0)}/64")
        logger.info(f"  Duration: {episode_duration:.2f}s")
        logger.info(f"  Epsilon: {agent.epsilon:.4f}")

        # Save best model (by reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = save_path / "best_model.pt"
            agent.save(str(best_model_path))
            logger.info(f"  ⭐ New best model saved! Reward: {best_reward:.2f}")

        # Save best CE detector model
        if total_ce_episode > best_ce_count:
            best_ce_count = total_ce_episode
            best_ce_model_path = save_path / "best_ce_model.pt"
            agent.save(str(best_ce_model_path))
            logger.info(f"  🎯 New best CE detector! CE count: {best_ce_count}")

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
                'best_reward': best_reward,
                'best_ce_count': best_ce_count
            }
            stats_path = save_path / f"training_stats_episode_{episode}.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"  Checkpoint and stats saved at episode {episode}")

        # Running average
        if len(episode_rewards) >= 10:
            avg_reward_10 = sum(episode_rewards[-10:]) / 10
            avg_ce_10 = sum(episode_ce_counts[-10:]) / 10
            logger.info(f"  Last 10 episodes avg reward: {avg_reward_10:.2f}")
            logger.info(f"  Last 10 episodes avg CE: {avg_ce_10:.2f}")

    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Best CE count: {best_ce_count}")
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
        'best_ce_count': best_ce_count,
        'final_epsilon': agent.epsilon
    }
    final_stats_path = save_path / "final_training_stats.json"
    with open(final_stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)

    logger.info(f"Final model and stats saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train DQN on GSAT-style CXL memory environment'
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
        default='models_gsat',
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

    train_dqn_gsat(
        memory_agent_url=args.memory_agent_url,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_file=args.log_file
    )


if __name__ == '__main__':
    main()
