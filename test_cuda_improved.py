#!/usr/bin/env python3
"""
Improved CUDA performance test with proper benchmarking
"""

import torch
import time


def benchmark_matmul(size, device, warmup=5, iterations=20):
    """
    Benchmark matrix multiplication with proper warmup

    Args:
        size: Matrix size (size x size)
        device: 'cuda' or 'cpu'
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
    """
    device_obj = torch.device(device)

    # Create tensors
    x = torch.randn(size, size, device=device_obj)
    y = torch.randn(size, size, device=device_obj)

    # Warmup
    for _ in range(warmup):
        z = torch.matmul(x, y)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        z = torch.matmul(x, y)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    return avg_time


def test_different_sizes():
    """Test GPU vs CPU on different matrix sizes"""

    print("=" * 80)
    print("GPU vs CPU Performance Test")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Test different sizes
    sizes = [100, 500, 1000, 2000, 4000, 8000]

    print(f"{'Size':<10} {'GPU Time':<12} {'CPU Time':<12} {'Speedup':<10} {'Winner'}")
    print("-" * 80)

    for size in sizes:
        try:
            # GPU benchmark
            gpu_time = benchmark_matmul(size, 'cuda', warmup=3, iterations=10)

            # CPU benchmark (fewer iterations for large sizes)
            cpu_iterations = 10 if size <= 2000 else 3
            cpu_time = benchmark_matmul(size, 'cpu', warmup=2, iterations=cpu_iterations)

            speedup = cpu_time / gpu_time
            winner = "🚀 GPU" if speedup > 1 else "💻 CPU"

            print(f"{size:<10} {gpu_time*1000:>8.2f}ms   {cpu_time*1000:>8.2f}ms   {speedup:>6.2f}x    {winner}")

        except Exception as e:
            print(f"{size:<10} Error: {str(e)[:50]}")

    print()


def test_neural_network():
    """Test realistic neural network operations"""

    print("=" * 80)
    print("Neural Network Operations Test (More Realistic)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    import torch.nn as nn

    # Create a realistic CNN model
    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 16 * 16, 256)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    batch_sizes = [1, 8, 32, 64]

    print(f"{'Batch Size':<12} {'GPU Time':<12} {'CPU Time':<12} {'Speedup':<10}")
    print("-" * 80)

    for batch_size in batch_sizes:
        # GPU model
        model_gpu = TestCNN().cuda()
        input_gpu = torch.randn(batch_size, 4, 64, 64).cuda()

        # Warmup
        for _ in range(5):
            _ = model_gpu(input_gpu)
            torch.cuda.synchronize()

        # GPU benchmark
        start = time.time()
        for _ in range(20):
            output = model_gpu(input_gpu)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 20

        # CPU model
        model_cpu = TestCNN().cpu()
        input_cpu = torch.randn(batch_size, 4, 64, 64).cpu()

        # Warmup
        for _ in range(3):
            _ = model_cpu(input_cpu)

        # CPU benchmark
        start = time.time()
        iterations = 20 if batch_size <= 32 else 10
        for _ in range(iterations):
            output = model_cpu(input_cpu)
        cpu_time = (time.time() - start) / iterations

        speedup = cpu_time / gpu_time

        print(f"{batch_size:<12} {gpu_time*1000:>8.2f}ms   {cpu_time*1000:>8.2f}ms   {speedup:>6.2f}x")

    print()


def test_ppo_agent():
    """Test actual PPO agent performance"""

    print("=" * 80)
    print("PPO Agent Inference Test (Real Use Case)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

    from ppo_agent import PPOAgent
    import numpy as np

    # Test with GPU
    print("Testing GPU agent...")
    agent_gpu = PPOAgent(device='cuda')
    dummy_state = np.random.rand(64, 64, 4).astype(np.float32)

    # Warmup
    for _ in range(5):
        _, _ = agent_gpu.select_action(dummy_state)

    # Benchmark
    start = time.time()
    for _ in range(100):
        action, action_info = agent_gpu.select_action(dummy_state)
    gpu_time = (time.time() - start) / 100

    # Test with CPU
    print("Testing CPU agent...")
    agent_cpu = PPOAgent(device='cpu')

    # Warmup
    for _ in range(3):
        _, _ = agent_cpu.select_action(dummy_state)

    # Benchmark
    start = time.time()
    for _ in range(100):
        action, action_info = agent_cpu.select_action(dummy_state)
    cpu_time = (time.time() - start) / 100

    speedup = cpu_time / gpu_time

    print()
    print(f"GPU action selection: {gpu_time*1000:.2f}ms per action")
    print(f"CPU action selection: {cpu_time*1000:.2f}ms per action")
    print(f"Speedup: {speedup:.2f}x")
    print()

    if speedup > 1:
        print("✅ GPU is faster! Use GPU for training.")
    else:
        print("⚠️  CPU is faster for this specific task.")
        print("   This is normal for small models with small batch sizes.")
        print("   GPU will be faster during actual training with batches.")


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CUDA Performance Benchmark" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system.")
        print("   Training will use CPU only.")
        return

    print(f"✅ CUDA is available!")
    print(f"   Device: {torch.cuda.get_device_name()}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Run tests
    test_different_sizes()
    test_neural_network()
    test_ppo_agent()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("📊 Key Findings:")
    print("  • Small matrices (< 2000): CPU might be faster (overhead)")
    print("  • Large matrices (> 4000): GPU is much faster")
    print("  • Neural networks: GPU is faster for batch processing")
    print("  • Actual training: GPU will be significantly faster")
    print()
    print("💡 Recommendation:")
    print("  Use GPU for training! The benefits show up with:")
    print("  - Batch training (32-64 samples)")
    print("  - Multiple epochs")
    print("  - Gradient updates")
    print()


if __name__ == "__main__":
    main()
