#!/usr/bin/env python3
"""
Test CUDA availability and performance
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ppo_agent import PPOAgent
from rl_environment import CXLMemoryRLEnvironment


def test_cuda_basic():
    """Test basic CUDA functionality"""
    
    print("Testing CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Test basic tensor operations
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Wait for completion
        cuda_time = time.time() - start_time
        
        # CPU comparison
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CUDA time: {cuda_time:.4f}s")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"Speedup: {cpu_time/cuda_time:.2f}x")
        
        return True
    else:
        print("CUDA not available")
        return False


def test_agent_cuda():
    """Test agent with CUDA"""
    
    print("\nTesting PPO Agent with CUDA...")
    
    try:
        # Test with CUDA if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        agent = PPOAgent(device=device)
        
        # Test action selection
        dummy_state = np.random.random((64, 64, 4)).astype(np.float32)
        
        start_time = time.time()
        for _ in range(10):
            action, action_info = agent.select_action(dummy_state)
        selection_time = time.time() - start_time
        
        print(f"10 action selections took: {selection_time:.4f}s")
        print(f"Average per action: {selection_time/10:.4f}s")
        
        # Test value estimation
        start_time = time.time()
        for _ in range(10):
            value = agent.get_value(dummy_state)
        value_time = time.time() - start_time
        
        print(f"10 value estimations took: {value_time:.4f}s")
        print(f"Average per value: {value_time/10:.4f}s")
        
        print("Agent CUDA test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"Agent CUDA test FAILED: {e}")
        return False


def test_training_step():
    """Test a full training step with CUDA"""
    
    print("\nTesting training step with CUDA...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        env = CXLMemoryRLEnvironment(safety_mode=True)
        agent = PPOAgent(device=device)
        
        # Run a few steps
        state, _ = env.reset()
        
        total_time = 0
        for i in range(5):
            start_time = time.time()
            
            action, action_info = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, 
                                 terminated or truncated, action_info)
            
            step_time = time.time() - start_time
            total_time += step_time
            
            print(f"Step {i+1}: {step_time:.4f}s, reward: {reward:.2f}")
            
            if terminated or truncated:
                break
            
            state = next_state
        
        print(f"Total time for 5 steps: {total_time:.4f}s")
        print(f"Average per step: {total_time/5:.4f}s")
        
        print("Training step test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"Training step test FAILED: {e}")
        return False


def main():
    """Run CUDA tests"""
    
    print("=" * 50)
    print("CXL Memory RL CUDA Test")
    print("=" * 50)
    
    tests = [
        ("CUDA Basic", test_cuda_basic),
        ("Agent CUDA", test_agent_cuda),
        ("Training Step", test_training_step)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        print(f"Running {test_name} test...")
        print(f"{'-' * 30}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("CUDA Test Summary:")
    print(f"{'=' * 50}")
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:15s}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if torch.cuda.is_available() and passed == len(results):
        print("🚀 CUDA is working perfectly! Ready for GPU acceleration.")
    elif not torch.cuda.is_available():
        print("💻 CUDA not available, will use CPU for training.")
    else:
        print("⚠️  Some CUDA tests failed, check the issues above.")


if __name__ == "__main__":
    main()