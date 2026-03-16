import pickle
from collections import defaultdict
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np

# Definimos la constante global para que el método 'load' no falle al reconstruir la tabla
_N_ACTIONS = 4 

class QLearningAgent:
    def __init__(
        self,
        env_id: str,
        *,
        n_bins: int = 4,           # Hiperparámetro original
        lr: float = 0.015,            # Hiperparámetro original
        gamma: float = 0.99,        # Hiperparámetro original
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9999, # Hiperparámetro original
    ) -> None:
        self.env_id = env_id
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_episodes = 0

        # Ajuste dinámico de límites según el entorno (LunarLander-v3)
        env = gym.make(env_id)
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        
        # Forzamos los límites de la Luna para evitar que se desborde el índice de los bins
        # x, y, vx, vy, angle, angular_v
        obs_low[:6] = [-2.5, -2.5, -10.0, -10.0, -6.28, -10.0]
        obs_high[:6] = [2.5, 2.5, 10.0, 10.0, 6.28, 10.0]
        
        self.n_obs = len(obs_low)
        env.close()
        
        # Creación de bins para cada dimensión de observación 
        self._bins = [
            np.linspace(obs_low[i], obs_high[i], n_bins + 1)[1:-1]
            for i in range(self.n_obs)
        ]
        
        self.n_actions = _N_ACTIONS
        # Inicialización de la tabla Q con el tamaño correcto para la nave (4 acciones) 
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(_N_ACTIONS)
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def discretize(self, obs: np.ndarray) -> tuple:
        """Convert continuous observation to discrete state."""
        # Definimos los límites para el recorte (clipping) basados en los bins creados
        low_clip = [self._bins[i][0] for i in range(self.n_obs)]
        high_clip = [self._bins[i][-1] for i in range(self.n_obs)]
        
        # Recorte de valores para que siempre caigan dentro de los bins definidos
        clipped = np.clip(obs[:self.n_obs], low_clip, high_clip)
        
        # Conversión a índices discretos [cite: 81]
        indices = tuple(int(np.digitize(clipped[i], self._bins[i])) for i in range(self.n_obs))
        return indices

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        state = self.discretize(obs)
        return self.select_action(state, deterministic=deterministic), None

    # ------------------------------------------------------------------
    # core RL
    # ------------------------------------------------------------------

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train(self, total_episodes: int = 10_000, log_interval: int = 100) -> list[float]:
        env = gym.make(self.env_id)
        
        # Asegurar que n_actions esté sincronizado con el entorno
        if self.n_actions is None:
            self.n_actions = int(env.action_space.n)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = self.discretize(next_obs)
                done = truncated or terminated
                self._update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # Decaimiento de epsilon original 
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"States visited: {len(self.q_table)}"
                )

        env.close()
        return rewards_history

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "n_bins": self.n_bins,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved Q-Learning agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            env_id=data["env_id"],
            n_bins=data["n_bins"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
        )
        # Uso de la constante corregida para reconstruir entradas faltantes
        agent.q_table = defaultdict(
            lambda: np.zeros(_N_ACTIONS), data["q_table"]
        )
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        return (
            f"Q-Learning agent for {self.env_id}\n"
            f"   Episodes trained : {self.training_episodes}\n"
            f"   States visited   : {len(self.q_table)}\n"
            f"   Epsilon          : {self.epsilon:.4f}\n"
            f"   LR / Gamma       : {self.lr} / {self.gamma}"
        )