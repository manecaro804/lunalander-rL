import gymnasium as gym
import ale_py
import numpy as np
import torch

from wrappers import PreprocessFrame, FrameStack, FrameSkip
from dqn_agent import DQNAgent

gym.register_envs(ale_py)


def train():
    env = gym.make("ALE/Pacman-v5", render_mode=None)
    env = FrameSkip(env, skip=4)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    agent = DQNAgent(n_actions)

    episodes = 1000

    # 🔥 tracking
    reward_history = []
    best_reward = -float("inf")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        step_counter = 0
        step = 0
        max_steps = 2000

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 🔥 REWARD SHAPING MEJORADO
            reward_shaped = reward / 10.0

            reward_shaped += 0.2  # sobrevivir

            if reward > 0:
                reward_shaped += 3  # comer

            if done:
                reward_shaped -= 15  # morir

            reward_shaped = np.clip(reward_shaped, -5, 5)

            agent.memory.push(state, action, reward_shaped, next_state, done)

            # 🔥 entrenamiento cada 4 pasos
            step_counter += 1
            if step_counter % 4 == 0:
                agent.train_step()

            state = next_state
            total_reward += reward

            step += 1
            if step > max_steps:
                break

        # 🔥 tracking
        reward_history.append(total_reward)

        if len(reward_history) >= 20:
            avg_reward = np.mean(reward_history[-20:])
        else:
            avg_reward = np.mean(reward_history)

        # 🔥 print cada 10 episodios
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Avg(20): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        # 🔥 guardar mejor modelo
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_net.state_dict(), "results/dqn_best.pth")
            print("🏆 Nuevo mejor modelo guardado!")

        # 🔥 checkpoints
        if (episode + 1) % 100 == 0:
            torch.save(agent.q_net.state_dict(), f"results/dqn_ep_{episode+1}.pth")
            print(f"💾 Checkpoint guardado en episode {episode+1}")

    env.close()

    torch.save(agent.q_net.state_dict(), "results/dqn_pacman_final.pth")
    print("\n✅ Modelo final guardado en results/dqn_pacman_final.pth")


if __name__ == "__main__":
    train()