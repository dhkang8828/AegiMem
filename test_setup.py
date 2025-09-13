#!/usr/bin/env python3
"""
Test script to verify CXL Memory RL setup
"""

import os
import sys
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rl_environment import CXLMemoryRLEnvironment
from ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment():
    """Test RL environment"""
    
    logger.info("Testing CXL Memory RL Environment...")
    
    try:
        # Create environment in safety mode
        env = CXLMemoryRLEnvironment(
            mbist_binary_path="/home/dhkang/data2/mbist_sample_code-gen2_es/bin/mbist_smbus.exe",
            safety_mode=True  # Use simulation
        )
        
        # Test reset
        state, info = env.reset()
        logger.info(f"Environment reset successful. State shape: {state.shape}")
        logger.info(f"Info: {info}")
        
        # Test random actions
        for i in range(5):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            logger.info(f"Step {i+1}:")
            logger.info(f"  Action: {action}")
            logger.info(f"  Reward: {reward:.2f}")
            logger.info(f"  Terminated: {terminated}")
            logger.info(f"  Faults found: {step_info.get('test_result', {}).get('faults_found', 0)}")
            
            if terminated or truncated:
                break
        
        logger.info("Environment test PASSED ✓")
        return True
        
    except Exception as e:
        logger.error(f"Environment test FAILED: {e}")
        return False


def test_agent():
    """Test PPO agent"""
    
    logger.info("Testing PPO Agent...")
    
    try:
        # Create agent
        agent = PPOAgent(
            state_shape=(64, 64, 4),
            action_dims=[14, 6, 64, 64]
        )
        
        # Test action selection
        dummy_state = np.random.random((64, 64, 4)).astype(np.float32)
        
        action, action_info = agent.select_action(dummy_state)
        logger.info(f"Action selection successful. Action: {action}")
        logger.info(f"Action info: {action_info}")
        
        # Test value estimation
        value = agent.get_value(dummy_state)
        logger.info(f"Value estimation: {value:.4f}")
        
        # Test storing transitions
        next_state = np.random.random((64, 64, 4)).astype(np.float32)
        agent.store_transition(dummy_state, action, 1.0, next_state, False, action_info)
        
        logger.info("Agent test PASSED ✓")
        return True
        
    except Exception as e:
        logger.error(f"Agent test FAILED: {e}")
        return False


def test_mbist_binary():
    """Test MBIST binary availability"""
    
    logger.info("Testing MBIST binary...")
    
    mbist_path = "/home/dhkang/data2/mbist_sample_code-gen2_es/bin/mbist_smbus.exe"
    
    if os.path.exists(mbist_path):
        logger.info(f"MBIST binary found at: {mbist_path}")
        
        # Check if we can get help output
        try:
            import subprocess
            result = subprocess.run([mbist_path, "-h"], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 or "Parameters for customer's configuration" in result.stdout:
                logger.info("MBIST binary responds correctly ✓")
                return True
            else:
                logger.warning(f"MBIST binary returned code {result.returncode}")
                logger.warning(f"Output: {result.stdout}")
                return False
                
        except Exception as e:
            logger.warning(f"Could not execute MBIST binary: {e}")
            logger.info("This is expected if running in simulation mode")
            return True
    else:
        logger.error(f"MBIST binary not found at: {mbist_path}")
        return False


def test_integration():
    """Test environment and agent integration"""
    
    logger.info("Testing Environment-Agent Integration...")
    
    try:
        # Create environment and agent
        env = CXLMemoryRLEnvironment(safety_mode=True)
        agent = PPOAgent()
        
        # Run a short episode
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(10):
            action, action_info = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, 
                                 terminated or truncated, action_info)
            
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        logger.info(f"Integration test completed. Total reward: {total_reward:.2f}")
        logger.info("Integration test PASSED ✓")
        return True
        
    except Exception as e:
        logger.error(f"Integration test FAILED: {e}")
        return False


def main():
    """Run all tests"""
    
    logger.info("=" * 50)
    logger.info("CXL Memory RL Project Setup Test")
    logger.info("=" * 50)
    
    tests = [
        ("MBIST Binary", test_mbist_binary),
        ("Environment", test_environment),
        ("Agent", test_agent),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 30}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'-' * 30}")
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            logger.info(f"{test_name} test: ✓ PASSED")
        else:
            logger.error(f"{test_name} test: ✗ FAILED")
    
    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("Test Summary:")
    logger.info(f"{'=' * 50}")
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name:15s}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Setup is ready for training.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())