"""
Hyperparameter tuning for RL agents using grid search and random search.
This script helps find optimal hyperparameters for Q-Learning and DQN agents.
"""

import json
from itertools import product
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from optuna import create_study, Trial
from optuna.samplers import TPESampler

from src.rl_games.agents.dqn import DQNAgent
from src.rl_games.agents.qlearning import QLearningAgent

ENV_ID = "CartPole-v1"  # Using CartPole instead of LunarLander for easier setup
RESULTS_DIR = Path("tuning_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Q-Learning Hyperparameter Search
# ─────────────────────────────────────────────────────────────────────


class QLearningTuner:
    """Grid search for Q-Learning agents."""

    param_grid = {
        "n_bins": [5, 10, 15, 20],
        "lr": [0.05, 0.1, 0.2],
        "gamma": [0.95, 0.99, 0.999],
        "epsilon_decay": [0.99, 0.995, 0.999],
    }

    def __init__(self, env_id: str = ENV_ID, episodes: int = 500, runs: int = 2):
        self.env_id = env_id
        self.episodes = episodes
        self.runs = runs
        self.results = []

    def evaluate(self, params: dict, run: int) -> dict:
        """Train an agent with given hyperparameters and return performance."""
        agent = QLearningAgent(
            self.env_id,
            n_bins=params["n_bins"],
            lr=params["lr"],
            gamma=params["gamma"],
            epsilon_decay=params["epsilon_decay"],
        )
        
        rewards = agent.train(total_episodes=self.episodes, log_interval=50)
        
        # Compute metrics
        final_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        max_reward = np.max(rewards)
        
        return {
            "params": params,
            "run": run,
            "final_avg_reward": float(final_reward),
            "max_reward": float(max_reward),
            "all_rewards": [float(r) for r in rewards],
        }

    def search_grid(self) -> list[dict]:
        """Perform exhaustive grid search."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        all_results = []
        total = np.prod([len(v) for v in values])
        current = 0
        
        for combo in product(*values):
            params = dict(zip(keys, combo))
            current += 1
            print(f"\n{'='*70}")
            print(f"Testing Q-Learning: {current}/{total}")
            print(f"Parameters: {params}")
            print(f"{'='*70}")
            
            # Multiple runs for the same hyperparameters
            run_results = []
            for run in range(self.runs):
                print(f"  Run {run + 1}/{self.runs}...")
                result = self.evaluate(params, run)
                run_results.append(result)
            
            # Aggregate results across runs
            avg_final = np.mean([r["final_avg_reward"] for r in run_results])
            std_final = np.std([r["final_avg_reward"] for r in run_results])
            
            aggregated = {
                "params": params,
                "avg_final_reward": float(avg_final),
                "std_final_reward": float(std_final),
                "run_results": run_results,
            }
            all_results.append(aggregated)
            
            print(f"  Results: {avg_final:.2f} ± {std_final:.2f}")
        
        return all_results


# ─────────────────────────────────────────────────────────────────────
# DQN Hyperparameter Search
# ─────────────────────────────────────────────────────────────────────


class DQNTuner:
    """Bayesian optimization for DQN agents using Optuna."""

    def __init__(self, env_id: str = ENV_ID, episodes: int = 200, n_trials: int = 20):
        self.env_id = env_id
        self.episodes = episodes
        self.n_trials = n_trials
        self.results = []

    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna."""
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
        batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
        hidden = trial.suggest_int("hidden", 64, 256, step=64)
        target_update_freq = trial.suggest_int("target_update_freq", 5, 50, step=5)

        try:
            agent = DQNAgent(
                self.env_id,
                lr=lr,
                gamma=gamma,
                epsilon_decay=epsilon_decay,
                batch_size=batch_size,
                hidden=hidden,
                target_update_freq=target_update_freq,
            )
            
            rewards = agent.train(total_episodes=self.episodes, log_interval=20)
            final_avg_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
            
            result = {
                "trial": trial.number,
                "params": {
                    "lr": lr,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay,
                    "batch_size": batch_size,
                    "hidden": hidden,
                    "target_update_freq": target_update_freq,
                },
                "final_avg_reward": float(final_avg_reward),
            }
            self.results.append(result)
            
            return final_avg_reward
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("-inf")

    def search_bayesian(self) -> list[dict]:
        """Perform Bayesian optimization."""
        sampler = TPESampler(seed=42)
        study = create_study(sampler=sampler, direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        return self.results


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────


def save_results(agent_type: str, results: list[dict]) -> None:
    """Save results to JSON file."""
    filepath = RESULTS_DIR / f"{agent_type}_tuning_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filepath}")


def print_best_results(agent_type: str, results: list[dict], top_k: int = 5) -> None:
    """Print the best hyperparameter combinations."""
    print(f"\n{'='*70}")
    print(f"Top {top_k} configurations for {agent_type.upper()}")
    print(f"{'='*70}\n")
    
    if agent_type == "qlearning":
        sorted_results = sorted(
            results, 
            key=lambda x: x["avg_final_reward"], 
            reverse=True
        )
        for rank, result in enumerate(sorted_results[:top_k], 1):
            print(f"{rank}. Final Reward: {result['avg_final_reward']:.2f} "
                  f"± {result['std_final_reward']:.2f}")
            print(f"   Parameters: {result['params']}\n")
    else:
        sorted_results = sorted(
            results, 
            key=lambda x: x["final_avg_reward"], 
            reverse=True
        )
        for rank, result in enumerate(sorted_results[:top_k], 1):
            print(f"{rank}. Final Reward: {result['final_avg_reward']:.2f}")
            print(f"   Parameters: {result['params']}\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RL agents")
    parser.add_argument(
        "--agent",
        choices=["qlearning", "dqn", "both"],
        default="both",
        help="Which agent to tune",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Episodes per training run",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials for DQN tuning",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of runs per Q-Learning configuration",
    )

    args = parser.parse_args()

    # Tune Q-Learning
    if args.agent in ["qlearning", "both"]:
        print("Starting Q-Learning hyperparameter tuning...")
        qlearning_tuner = QLearningTuner(episodes=args.episodes, runs=args.runs)
        qlearning_results = qlearning_tuner.search_grid()
        save_results("qlearning", qlearning_results)
        print_best_results("qlearning", qlearning_results)

    # Tune DQN
    if args.agent in ["dqn", "both"]:
        print("Starting DQN hyperparameter tuning...")
        dqn_tuner = DQNTuner(episodes=args.episodes, n_trials=args.trials)
        dqn_results = dqn_tuner.search_bayesian()
        save_results("dqn", dqn_results)
        print_best_results("dqn", dqn_results)
