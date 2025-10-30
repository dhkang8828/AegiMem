#!/usr/bin/env python3
"""
Generate training report from logs
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_results(log_dir: str, experiment_name: str = None) -> Dict:
    """Load training results from log directory"""

    if experiment_name:
        # Load specific experiment
        result_file = os.path.join(log_dir, f"{experiment_name}_final_results.json")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Result file not found: {result_file}")

        with open(result_file, 'r') as f:
            return json.load(f)
    else:
        # Find most recent result file
        result_files = glob.glob(os.path.join(log_dir, "*_final_results.json"))
        if not result_files:
            raise FileNotFoundError(f"No result files found in {log_dir}")

        # Sort by modification time
        result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = result_files[0]

        print(f"Loading latest results from: {os.path.basename(latest_file)}")

        with open(latest_file, 'r') as f:
            return json.load(f)


def generate_summary_text(results: Dict) -> str:
    """Generate text summary of results"""

    summary = []
    summary.append("=" * 80)
    summary.append("CXL Memory Fault Detection RL Training Report")
    summary.append("=" * 80)
    summary.append("")

    # Configuration
    config = results.get('config', {})
    summary.append("Configuration:")
    summary.append(f"  Total Episodes: {config.get('training', {}).get('total_episodes', 'N/A')}")
    summary.append(f"  Safety Mode: {config.get('environment', {}).get('safety_mode', 'N/A')}")
    summary.append(f"  Learning Rate: {config.get('ppo', {}).get('learning_rate', 'N/A')}")
    summary.append(f"  Batch Size: {config.get('ppo', {}).get('batch_size', 'N/A')}")
    summary.append("")

    # Performance Summary
    perf = results.get('final_performance', {})
    summary.append("Performance Summary:")
    summary.append(f"  Best Reward: {perf.get('best_reward', 0):.2f}")
    summary.append(f"  Best Faults Found: {perf.get('best_faults_found', 0)}")
    summary.append(f"  Avg Reward (Last 100): {perf.get('avg_reward_last_100', 0):.2f}")
    summary.append(f"  Avg Faults (Last 100): {perf.get('avg_faults_last_100', 0):.2f}")
    summary.append("")

    # Training Statistics
    training_data = results.get('training_data', {})
    episode_rewards = training_data.get('episode_rewards', [])
    episode_faults = training_data.get('episode_faults_found', [])

    if episode_rewards:
        summary.append("Training Statistics:")
        summary.append(f"  Total Episodes Completed: {len(episode_rewards)}")
        summary.append(f"  Average Reward: {np.mean(episode_rewards):.2f}")
        summary.append(f"  Reward Std Dev: {np.std(episode_rewards):.2f}")
        summary.append(f"  Total Faults Found: {sum(episode_faults)}")
        summary.append(f"  Average Faults per Episode: {np.mean(episode_faults):.2f}")
        summary.append("")

    # Agent Statistics
    agent_stats = results.get('agent_stats', {})
    if agent_stats:
        summary.append("Agent Learning Statistics:")
        for key, value in agent_stats.items():
            summary.append(f"  {key}: {value:.4f}")
        summary.append("")

    # Evaluation Results
    eval_results = training_data.get('evaluation_results', [])
    if eval_results:
        summary.append("Evaluation Checkpoints:")
        for eval_point in eval_results[-5:]:  # Last 5 evaluations
            summary.append(
                f"  Episode {eval_point['episode']:4d}: "
                f"Reward={eval_point['eval_reward']:8.2f}, "
                f"Faults={eval_point['eval_faults']:6.2f}, "
                f"Coverage={eval_point['eval_coverage']:6.2f}"
            )
        summary.append("")

    summary.append("=" * 80)
    summary.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 80)

    return "\n".join(summary)


def generate_plots(results: Dict, output_dir: str):
    """Generate visualization plots"""

    training_data = results.get('training_data', {})
    episode_rewards = training_data.get('episode_rewards', [])
    episode_lengths = training_data.get('episode_lengths', [])
    episode_faults = training_data.get('episode_faults_found', [])

    if not episode_rewards:
        print("No training data available for plotting")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CXL Memory RL Training Report', fontsize=16, fontweight='bold')

    # 1. Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, linewidth=0.5)
    if len(episode_rewards) >= 50:
        window = 50
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='MA(50)')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode Lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Faults Found
    axes[0, 2].plot(episode_faults, alpha=0.6, linewidth=0.5)
    if len(episode_faults) >= 50:
        window = 50
        moving_avg = np.convolve(episode_faults, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(range(window-1, len(episode_faults)), moving_avg, 'g-', linewidth=2, label='MA(50)')
        axes[0, 2].legend()
    axes[0, 2].set_title('Faults Found per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Faults')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Reward Distribution
    axes[1, 0].hist(episode_rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 5. Cumulative Faults
    cumulative_faults = np.cumsum(episode_faults)
    axes[1, 1].plot(cumulative_faults, linewidth=2)
    axes[1, 1].set_title('Cumulative Faults Discovered')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Faults')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Learning Progress (Reward vs Faults)
    if len(episode_rewards) >= 100:
        # Use last 100 episodes
        recent_rewards = episode_rewards[-100:]
        recent_faults = episode_faults[-100:]

        axes[1, 2].scatter(recent_faults, recent_rewards, alpha=0.5)
        axes[1, 2].set_title('Reward vs Faults (Last 100 Episodes)')
        axes[1, 2].set_xlabel('Faults Found')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(recent_faults, recent_rewards, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(recent_faults), max(recent_faults), 100)
        axes[1, 2].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'Not enough data\n(< 100 episodes)',
                       ha='center', va='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'training_report.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training report')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--experiment', type=str, help='Experiment name (default: most recent)')
    parser.add_argument('--output-dir', type=str, default='./reports', help='Output directory for report')
    parser.add_argument('--format', type=str, choices=['text', 'plot', 'both'], default='both',
                       help='Report format')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load results
        print(f"Loading results from {args.log_dir}...")
        results = load_results(args.log_dir, args.experiment)

        # Generate text report
        if args.format in ['text', 'both']:
            print("\nGenerating text report...")
            summary = generate_summary_text(results)
            print(summary)

            # Save to file
            report_path = os.path.join(args.output_dir, 'training_report.txt')
            with open(report_path, 'w') as f:
                f.write(summary)
            print(f"\nText report saved to: {report_path}")

        # Generate plots
        if args.format in ['plot', 'both']:
            print("\nGenerating plots...")
            generate_plots(results, args.output_dir)

        print("\nReport generation completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
