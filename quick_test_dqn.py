#!/usr/bin/env python3
"""
Quick integration test for DQN
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rl_environment import CXLMemoryRLEnvironment
from dqn_agent import DQNAgent


def quick_test_dqn():
    """Run a quick integration test for DQN"""

    print("Running DQN quick integration test...")
    print("=" * 60)

    # Create environment and agent
    env = CXLMemoryRLEnvironment(safety_mode=True, max_episode_steps=20)
    agent = DQNAgent(
        state_shape=(64, 64, 4),
        action_dims=[14, 6, 64, 64],
        learning_rate=1e-4,
        epsilon_start=1.0,
        batch_size=32
    )

    print(f"Environment: {env}")
    print(f"Agent: DQN with epsilon={agent.epsilon:.3f}")
    print(f"Device: {agent.device}")
    print("")

    # Run one episode
    state, _ = env.reset()
    total_reward = 0
    faults_found = 0

    print(f"Initial state shape: {state.shape}")
    print("")

    for step in range(20):
        print(f"Step {step+1}:")

        # Select action
        action = agent.select_action(state, training=True)
        print(f"  Action: {action}")
        print(f"  Epsilon: {agent.epsilon:.3f}")

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)

        print(f"  Reward: {reward:.2f}")
        print(f"  Test result: {info.get('test_result', {}).get('status', 'unknown')}")
        print(f"  Faults: {info.get('test_result', {}).get('faults_found', 0)}")

        total_reward += reward
        faults_found += info.get('test_result', {}).get('faults_found', 0)

        # Store transition
        agent.store_transition(state, action, reward, next_state, terminated or truncated)

        # Try to update (will only work if enough samples)
        update_info = agent.update()
        if update_info:
            print(f"  Loss: {update_info['loss']:.4f}")
            print(f"  Q-value: {update_info['q_value']:.4f}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break

        state = next_state
        print("")

    print("=" * 60)
    print(f"Episode Summary:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total faults found: {faults_found}")
    print(f"  Steps taken: {step+1}")
    print(f"  Replay buffer size: {len(agent.replay_buffer)}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print("")

    # Test a few more updates to fill buffer
    print("Filling replay buffer with random experiences...")
    for _ in range(50):
        state, _ = env.reset()
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, terminated or truncated)

    print(f"Replay buffer size after filling: {len(agent.replay_buffer)}")
    print("")

    # Test batch update
    print("Testing batch update...")
    if len(agent.replay_buffer) >= agent.batch_size:
        update_info = agent.update()
        if update_info:
            print(f"  Loss: {update_info['loss']:.4f}")
            print(f"  Q-value: {update_info['q_value']:.4f}")
            print(f"  Epsilon: {update_info['epsilon']:.3f}")
    else:
        print("  Not enough samples for update")

    print("")
    print("✓ DQN quick integration test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        quick_test_dqn()
        print("\n🎉 All DQN systems working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
