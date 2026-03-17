"""
Quick hyperparameter testing script.
Useful for rapid experimentation with specific configurations.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from src.rl_games.agents.dqn import DQNAgent
from src.rl_games.agents.qlearning import QLearningAgent

ENV_ID = "LunaLander-v3"  # Using LunarLander for easier setup
RESULTS_DIR = Path("quick_tests")
RESULTS_DIR.mkdir(exist_ok=True)


def test_qlearning_config(config: dict, episodes: int = 500, name: str = "test") -> dict:
    """Test a specific Q-Learning configuration."""
    print(f"\nTesting Q-Learning: {name}")
    print(f"Config: {config}")
    
    agent = QLearningAgent(ENV_ID, **config)
    rewards = agent.train(total_episodes=episodes, log_interval=100)
    
    metrics = {
        "name": name,
        "config": config,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "final_avg": float(np.mean(rewards[-100:])),
        "rewards": [float(r) for r in rewards],
    }
    
    print(f"  Mean: {metrics['mean_reward']:.2f}, Std: {metrics['std_reward']:.2f}")
    print(f"  Max: {metrics['max_reward']:.2f}, Final Avg: {metrics['final_avg']:.2f}")
    
    return metrics


def test_dqn_config(config: dict, episodes: int = 200, name: str = "test") -> dict:
    """Test a specific DQN configuration."""
    print(f"\nTesting DQN: {name}")
    print(f"Config: {config}")
    
    agent = DQNAgent(ENV_ID, **config)
    rewards = agent.train(total_episodes=episodes, log_interval=40)
    
    metrics = {
        "name": name,
        "config": config,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "final_avg": float(np.mean(rewards[-40:])),
        "rewards": [float(r) for r in rewards],
    }
    
    print(f"  Mean: {metrics['mean_reward']:.2f}, Std: {metrics['std_reward']:.2f}")
    print(f"  Max: {metrics['max_reward']:.2f}, Final Avg: {metrics['final_avg']:.2f}")
    
    return metrics


def compare_configs(results: list[dict], metric: str = "final_avg") -> None:
    """Compare multiple configurations and rank them."""
    print("\n" + "="*70)
    print("RANKING BY PERFORMANCE")
    print("="*70 + "\n")
    
    sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank}. {result['name']:20s} | {metric}: {result[metric]:.2f}")
        for k, v in result['config'].items():
            print(f"   - {k}: {v}")
        print()


def plot_training_curves(results: list[dict], output_file: str = "training_curves.png") -> None:
    """Plot training curves for multiple configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for result in results:
        rewards = result['rewards']
        ax.plot(rewards, label=result['name'], linewidth=2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Example configurations to test
# ─────────────────────────────────────────────────────────────────────

QLEARNING_CONFIGS = {
    "baseline": {
        "n_bins": 10,
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
    },
    "high_exploration": {
        "n_bins": 10,
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon_decay": 0.99,  # Decays slower, more exploration
    },
    "high_learning_rate": {
        "n_bins": 10,
        "lr": 0.2,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
    },
    "fine_discretization": {
        "n_bins": 20,  # More bins = finer state representation
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
    },
    "coarse_discretization": {
        "n_bins": 5,  # Fewer bins = coarser state representation
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
    },
}

DQN_CONFIGS = {
    "baseline": {
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "hidden": 128,
        "target_update_freq": 10,
    },
    "smaller_network": {
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "hidden": 64,
        "target_update_freq": 10,
    },
    "larger_network": {
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "hidden": 256,
        "target_update_freq": 10,
    },
    "lower_learning_rate": {
        "lr": 5e-4,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "hidden": 128,
        "target_update_freq": 10,
    },
    "longer_target_update": {
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "hidden": 128,
        "target_update_freq": 20,
    },
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick hyperparameter testing")
    parser.add_argument(
        "--agent",
        choices=["qlearning", "dqn", "both"],
        default="both",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to test (e.g., 'baseline', 'high_learning_rate')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training curves",
    )

    args = parser.parse_args()

    results = []

    # Test Q-Learning
    if args.agent in ["qlearning", "both"]:
        configs = QLEARNING_CONFIGS
        if args.config:
            configs = {args.config: QLEARNING_CONFIGS[args.config]}
        
        for name, config in configs.items():
            result = test_qlearning_config(config, episodes=args.episodes, name=name)
            results.append(result)

    # Test DQN
    if args.agent in ["dqn", "both"]:
        configs = DQN_CONFIGS
        if args.config:
            configs = {args.config: DQN_CONFIGS[args.config]}
        
        for name, config in configs.items():
            result = test_dqn_config(config, episodes=args.episodes // 2, name=name)
            results.append(result)

    # Compare and visualize
    if results:
        compare_configs(results, metric="final_avg")
        
        if args.plot and len(results) > 1:
            plot_training_curves(results)
        
        # Save results
        output_file = RESULTS_DIR / "quick_test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
