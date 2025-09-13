#!/usr/bin/env python3
"""
Quick integration test
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rl_environment import CXLMemoryRLEnvironment
from ppo_agent import PPOAgent


def quick_test():
    """Run a quick integration test"""
    
    print("Running quick integration test...")
    
    # Create environment and agent
    env = CXLMemoryRLEnvironment(safety_mode=True, max_episode_steps=10)
    agent = PPOAgent()
    
    # Run one episode
    state, _ = env.reset()
    total_reward = 0
    faults_found = 0
    
    print(f"Initial state shape: {state.shape}")
    
    for step in range(10):
        print(f"\nStep {step+1}:")
        
        # Select action
        action, action_info = agent.select_action(state)
        print(f"  Action: {action}")
        print(f"  Log prob: {action_info['log_prob']:.4f}")
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Reward: {reward:.2f}")
        print(f"  Test result: {info.get('test_result', {}).get('status', 'unknown')}")
        print(f"  Faults: {info.get('test_result', {}).get('faults_found', 0)}")
        
        total_reward += reward
        faults_found += info.get('test_result', {}).get('faults_found', 0)
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, 
                             terminated or truncated, action_info)
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
        
        state = next_state
    
    print(f"\nEpisode Summary:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total faults found: {faults_found}")
    print(f"  Steps taken: {step+1}")
    
    # Test agent update
    print(f"\nTesting agent update...")
    update_info = agent.update(batch_size=min(len(agent.buffer), 32), epochs=1)
    if update_info:
        print(f"  Policy loss: {update_info.get('policy_loss', 0):.4f}")
        print(f"  Value loss: {update_info.get('value_loss', 0):.4f}")
        print(f"  Entropy: {update_info.get('entropy', 0):.4f}")
    else:
        print("  Not enough data for update")
    
    print("\n✓ Quick integration test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        quick_test()
        print("\n🎉 All systems working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)